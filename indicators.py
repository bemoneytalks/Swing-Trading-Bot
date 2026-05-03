"""Technical indicator calculations and feature engineering for SPX."""

import os
import time

import numpy as np
import pandas as pd
import ta
import yfinance as yf
import config

# ---------------------------------------------------------------------------
# Cross-asset cache configuration
# ---------------------------------------------------------------------------
_CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache", "cross_asset.csv")
_CACHE_TTL_SECONDS = 4 * 3600  # 4 hours

_XA_TICKERS = {
    "xa_vix":   "^VIX",
    "xa_vix3m": "^VIX3M",
    "xa_tlt":   "TLT",
    "xa_gld":   "GLD",
    "xa_dxy":   "DX-Y.NYB",
}


def _fetch_cross_asset_data(start_date, end_date):
    """Fetch daily Close prices for VIX and cross-asset instruments.

    Uses a disk cache at ``cache/cross_asset.csv``.  The cache is reused
    unless it is older than 4 hours *or* the requested date range extends
    beyond what is already cached.  Individual ticker failures are skipped
    gracefully so a partial result is always returned.

    Parameters
    ----------
    start_date : str | date-like
        First date needed (inclusive).
    end_date : str | date-like
        Last date needed (inclusive).

    Returns
    -------
    pd.DataFrame
        Columns: xa_vix, xa_vix3m, xa_tlt, xa_gld, xa_dxy
        Index  : tz-naive dates
    """
    start_date = pd.Timestamp(start_date).normalize()
    end_date   = pd.Timestamp(end_date).normalize()

    # ---- Try to load cache ------------------------------------------------
    cached = None
    cache_is_fresh = False
    if os.path.exists(_CACHE_PATH):
        cache_age = time.time() - os.path.getmtime(_CACHE_PATH)
        if cache_age < _CACHE_TTL_SECONDS:
            try:
                cached = pd.read_csv(_CACHE_PATH, index_col="date", parse_dates=True)
                cached.index = cached.index.tz_localize(None)
                # Check that the cached range covers the requested range
                if (cached.index.min() <= start_date and
                        cached.index.max() >= end_date):
                    cache_is_fresh = True
            except Exception:
                cached = None

    if cache_is_fresh and cached is not None:
        return cached

    # ---- Fetch from Yahoo Finance -----------------------------------------
    # Download one extra day before start so pct_change is available
    fetch_start = (start_date - pd.Timedelta(days=30)).strftime("%Y-%m-%d")
    fetch_end   = (end_date   + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    frames = {}
    for col, ticker in _XA_TICKERS.items():
        try:
            raw = yf.download(
                ticker,
                start=fetch_start,
                end=fetch_end,
                auto_adjust=True,
                progress=False,
            )
            if raw.empty:
                continue
            # Flatten MultiIndex columns if present (yfinance ≥ 0.2)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            series = raw["Close"].copy()
            series.index = pd.to_datetime(series.index).tz_localize(None)
            series.name = col
            frames[col] = series
        except Exception as exc:
            print(f"[indicators] WARNING: could not fetch {ticker}: {exc}")

    if not frames:
        return pd.DataFrame()

    xa = pd.concat(frames.values(), axis=1)
    xa.index.name = "date"

    # ---- Persist cache -----------------------------------------------------
    try:
        os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
        xa.to_csv(_CACHE_PATH, index=True)
    except Exception as exc:
        print(f"[indicators] WARNING: could not write cross-asset cache: {exc}")

    return xa


def add_all_features(df):
    """Add all technical indicators and engineered features to the dataframe."""
    df = df.copy()
    df.index = pd.to_datetime(df.index, utc=True).tz_localize(None).normalize()

    # --- Price Action Features ---
    df["returns_1d"] = df["Close"].pct_change()
    df["returns_2d"] = df["Close"].pct_change(2)
    df["returns_5d"] = df["Close"].pct_change(5)
    df["returns_10d"] = df["Close"].pct_change(10)

    # Intraday range and body
    df["daily_range"] = (df["High"] - df["Low"]) / df["Close"]
    df["body"] = (df["Close"] - df["Open"]) / df["Close"]
    df["upper_wick"] = (df["High"] - df[["Open", "Close"]].max(axis=1)) / df["Close"]
    df["lower_wick"] = (df[["Open", "Close"]].min(axis=1) - df["Low"]) / df["Close"]

    # Gap (open vs previous close)
    df["gap"] = (df["Open"] - df["Close"].shift(1)) / df["Close"].shift(1)
    df["gap_filled"] = ((df["gap"] > 0) & (df["Low"] <= df["Close"].shift(1)) |
                        (df["gap"] < 0) & (df["High"] >= df["Close"].shift(1))).astype(int)

    # --- Moving Averages ---
    for period in config.LOOKBACK_PERIODS:
        df[f"sma_{period}"] = df["Close"].rolling(period).mean()
        df[f"ema_{period}"] = df["Close"].ewm(span=period).mean()
        df[f"dist_sma_{period}"] = (df["Close"] - df[f"sma_{period}"]) / df[f"sma_{period}"]
        df[f"dist_ema_{period}"] = (df["Close"] - df[f"ema_{period}"]) / df[f"ema_{period}"]

    # MA crossovers
    df["sma_5_20_cross"] = (df["sma_5"] > df["sma_20"]).astype(int)
    df["sma_10_50_cross"] = (df["sma_10"] > df["sma_50"]).astype(int)
    df["sma_50_200_cross"] = (df["sma_50"] > df["sma_200"]).astype(int)
    df["ema_5_20_cross"] = (df["ema_5"] > df["ema_20"]).astype(int)

    # --- Momentum Indicators ---
    df["rsi"] = ta.momentum.rsi(df["Close"], window=config.RSI_PERIOD)
    df["rsi_5"] = ta.momentum.rsi(df["Close"], window=5)

    macd_ind = ta.trend.MACD(df["Close"], window_slow=config.MACD_SLOW,
                              window_fast=config.MACD_FAST, window_sign=config.MACD_SIGNAL)
    df["macd"] = macd_ind.macd()
    df["macd_signal"] = macd_ind.macd_signal()
    df["macd_hist"] = macd_ind.macd_diff()
    df["macd_hist_change"] = df["macd_hist"].diff()

    stoch = ta.momentum.StochasticOscillator(df["High"], df["Low"], df["Close"],
                                              window=config.STOCH_PERIOD)
    df["stoch_k"] = stoch.stoch()
    df["stoch_d"] = stoch.stoch_signal()

    df["williams_r"] = ta.momentum.williams_r(df["High"], df["Low"], df["Close"],
                                               lbp=config.WILLIAMS_PERIOD)
    df["cci"] = ta.trend.cci(df["High"], df["Low"], df["Close"], window=config.CCI_PERIOD)
    df["roc_10"] = ta.momentum.roc(df["Close"], window=10)
    df["roc_20"] = ta.momentum.roc(df["Close"], window=20)

    # --- Volatility Indicators ---
    bb = ta.volatility.BollingerBands(df["Close"], window=config.BB_PERIOD, window_dev=config.BB_STD)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["bb_mid"]
    df["bb_pct"] = (df["Close"] - df["bb_lower"]) / (df["bb_upper"] - df["bb_lower"])

    df["atr"] = ta.volatility.average_true_range(df["High"], df["Low"], df["Close"],
                                                  window=config.ATR_PERIOD)
    df["atr_pct"] = df["atr"] / df["Close"]

    # Historical volatility
    df["hvol_10"] = df["returns_1d"].rolling(10).std() * np.sqrt(252)
    df["hvol_20"] = df["returns_1d"].rolling(20).std() * np.sqrt(252)

    # --- Trend Indicators ---
    adx = ta.trend.ADXIndicator(df["High"], df["Low"], df["Close"], window=config.ADX_PERIOD)
    df["adx"] = adx.adx()
    df["adx_pos"] = adx.adx_pos()
    df["adx_neg"] = adx.adx_neg()

    # --- Volume Features ---
    df["vol_sma_20"] = df["Volume"].rolling(20).mean()
    df["vol_ratio"] = df["Volume"] / df["vol_sma_20"]
    df["vol_change"] = df["Volume"].pct_change()

    # --- Day-of-Week and Calendar Features ---
    df["dow"] = df.index.dayofweek  # Mon=0, Fri=4
    df["dom"] = df.index.day
    df["month"] = df.index.month
    df["is_monday"] = (df["dow"] == 0).astype(int)
    df["is_friday"] = (df["dow"] == 4).astype(int)
    df["is_month_start"] = (df["dom"] <= 3).astype(int)
    df["is_month_end"] = (df["dom"] >= 27).astype(int)

    # --- Pattern Features ---
    # Consecutive up/down days
    up = (df["Close"] > df["Close"].shift(1)).astype(int)
    down = (df["Close"] < df["Close"].shift(1)).astype(int)
    df["consec_up"] = up.groupby((up != up.shift()).cumsum()).cumsum()
    df["consec_down"] = down.groupby((down != down.shift()).cumsum()).cumsum()

    # Distance from N-day high/low
    for period in [10, 20, 50]:
        df[f"dist_high_{period}"] = (df["Close"] - df["High"].rolling(period).max()) / df["Close"]
        df[f"dist_low_{period}"] = (df["Close"] - df["Low"].rolling(period).min()) / df["Close"]

    # --- Mean Reversion Features ---
    df["zscore_20"] = (df["Close"] - df["Close"].rolling(20).mean()) / df["Close"].rolling(20).std()
    df["zscore_50"] = (df["Close"] - df["Close"].rolling(50).mean()) / df["Close"].rolling(50).std()

    # --- Swing / Trend-Specific Features (5-day model) ---
    # Longer-horizon returns
    df["returns_20d"] = df["Close"].pct_change(20)
    df["returns_50d"] = df["Close"].pct_change(50)

    # Longer RSI — better for swing trend identification
    df["rsi_21"] = ta.momentum.rsi(df["Close"], window=21)

    # MA spread: how far is 50 SMA from 200 SMA (trend strength/direction)
    df["ma_spread_50_200"] = (df["sma_50"] - df["sma_200"]) / df["sma_200"]

    # Trend regime: is the market in a trending or ranging environment
    df["trend_regime"] = (df["adx"] > 25).astype(int)
    df["strong_trend_regime"] = (df["adx"] > 40).astype(int)

    # Higher highs / higher lows over 10 and 20 days (swing structure)
    df["higher_high_10"] = (df["High"] > df["High"].shift(10)).astype(int)
    df["higher_low_10"] = (df["Low"] > df["Low"].shift(10)).astype(int)
    df["higher_high_20"] = (df["High"] > df["High"].shift(20)).astype(int)
    df["higher_low_20"] = (df["Low"] > df["Low"].shift(20)).astype(int)

    # Trend score: sum of higher highs + higher lows over recent swings
    df["swing_trend_score"] = (df["higher_high_10"] + df["higher_low_10"] +
                                df["higher_high_20"] + df["higher_low_20"])

    # Momentum acceleration: is the 5d return accelerating vs 10d?
    df["momentum_accel"] = df["returns_5d"] - df["returns_10d"]

    # Volume trend: is volume higher on up days vs down days (accumulation)?
    # Use a simpler calculation: rolling sum of signed volume
    df["signed_volume"] = df["Volume"] * np.sign(df["returns_1d"].fillna(0))
    df["vol_accumulation"] = df["signed_volume"].rolling(10).sum() / df["Volume"].rolling(10).mean().replace(0, np.nan)
    df["vol_accumulation"] = df["vol_accumulation"].fillna(0)

    # Clean up inf values
    df = df.replace([np.inf, -np.inf], np.nan)

    # --- Cross-Asset Features (VIX, rates, gold, dollar) ---
    xa = _fetch_cross_asset_data(df.index.min(), df.index.max())

    if not xa.empty:
        # Left-join on date index so df shape is preserved; ffill gaps
        df = df.join(xa, how="left")
        df[list(xa.columns)] = df[list(xa.columns)].ffill()

        # -- VIX features --------------------------------------------------
        if "xa_vix" in df.columns:
            df["xa_vix_chg"]      = df["xa_vix"].pct_change()
            df["xa_vix_ma20"]     = df["xa_vix"].rolling(20).mean()
            df["xa_vix_dist_ma20"] = (df["xa_vix"] - df["xa_vix_ma20"]) / df["xa_vix_ma20"]
            df["xa_vix_high"]     = (df["xa_vix"] > 25).astype(int)
            df["xa_vix_low"]      = (df["xa_vix"] < 15).astype(int)

        # -- VIX term structure (contango = front VIX < 3-month VIX) -------
        if "xa_vix" in df.columns and "xa_vix3m" in df.columns:
            df["xa_vix_ts"]       = df["xa_vix"] / df["xa_vix3m"]
            df["xa_vix_contango"] = (df["xa_vix_ts"] < 1).astype(int)

        # -- TLT features --------------------------------------------------
        if "xa_tlt" in df.columns:
            df["xa_tlt_chg"]   = df["xa_tlt"].pct_change()
            df["xa_tlt_chg5"]  = df["xa_tlt"].pct_change(5)
            df["xa_tlt_trend"] = (
                df["xa_tlt"] > df["xa_tlt"].ewm(span=10).mean()
            ).astype(int)

        # -- GLD features --------------------------------------------------
        if "xa_gld" in df.columns:
            df["xa_gld_chg"]  = df["xa_gld"].pct_change()
            df["xa_gld_chg5"] = df["xa_gld"].pct_change(5)

        # -- DXY features --------------------------------------------------
        if "xa_dxy" in df.columns:
            df["xa_dxy_chg"]  = df["xa_dxy"].pct_change()
            df["xa_dxy_chg5"] = df["xa_dxy"].pct_change(5)

        # -- SPX / TLT rolling correlation (risk-on / risk-off signal) -----
        if "xa_tlt" in df.columns:
            tlt_ret = df["xa_tlt"].pct_change()
            df["xa_spx_tlt_corr20"] = df["returns_1d"].rolling(20).corr(tlt_ret)

        # Clean up any inf values introduced by the division above
        df = df.replace([np.inf, -np.inf], np.nan)

        # Drop raw price-level columns — keep only derived features
        raw_cols = [c for c in _XA_TICKERS.keys() if c in df.columns]
        df = df.drop(columns=raw_cols)

    return df


def get_feature_columns(df):
    """Return list of feature column names (excludes OHLCV and target)."""
    exclude = {"Open", "High", "Low", "Close", "Volume", "target",
               "sma_5", "sma_10", "sma_20", "sma_50", "sma_100", "sma_200",
               "ema_5", "ema_10", "ema_20", "ema_50", "ema_100", "ema_200",
               "bb_upper", "bb_lower", "bb_mid", "vol_sma_20", "atr"}
    return [c for c in df.columns if c not in exclude]
