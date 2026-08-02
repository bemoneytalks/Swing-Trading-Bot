"""
Options Contract Grader — A–F scoring of swing-trade option candidates.

Direction comes from the Trend Confluence system; contracts come from the
shared chain service (same cached fetch that powers the GEX/VEX/DEX ladder).
Every contract in the swing DTE window is scored against the preset rules in
config (GRADER_*), then the best candidate per risk tier (conservative /
balanced / aggressive) is surfaced with Recommendation, Quality, and
Tier-fit scores.

All estimates use the standard Black-Scholes machinery from gex.py and are
educational — quotes are delayed and OI updates once daily.
"""

import math
import time
from datetime import datetime

import yfinance as yf
from scipy.stats import norm

import config
from chain_service import get_chain, chain_stats
from earnings import get_next_earnings as _get_next_earnings
from gex import black_scholes_delta, RISK_FREE_RATE

# In-memory caches (per process)
_hist_cache = {}       # symbol -> (dict, ts)
_CACHE_TTL = 6 * 3600


def _letter(score):
    for cutoff, letter in config.GRADER_LETTERS:
        if score >= cutoff:
            return letter
    return "F"


def _bs_theta_daily(S, K, T, r, sigma, option_type):
    """Black-Scholes theta per calendar day (negative for long options)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    try:
        sqrt_t = math.sqrt(T)
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt_t)
        d2 = d1 - sigma * sqrt_t
        term1 = -(S * norm.pdf(d1) * sigma) / (2 * sqrt_t)
        if option_type == "call":
            term2 = -r * K * math.exp(-r * T) * norm.cdf(d2)
        else:
            term2 = r * K * math.exp(-r * T) * norm.cdf(-d2)
        return (term1 + term2) / 365.0
    except (ValueError, ZeroDivisionError):
        return 0.0


def _prob_beyond_breakeven(S, breakeven, T, sigma, option_type):
    """Risk-neutral probability the underlying finishes beyond breakeven."""
    if T <= 0 or sigma <= 0 or breakeven <= 0 or S <= 0:
        return 0.0
    try:
        z = (math.log(S / breakeven) + (RISK_FREE_RATE - 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        if option_type == "call":
            return float(norm.cdf(z))
        return float(norm.cdf(-z))
    except (ValueError, ZeroDivisionError):
        return 0.0


def _get_underlying_stats(symbol):
    """20-day HV and 14-day ATR for the underlying. Cached 6h."""
    now = time.time()
    if symbol in _hist_cache:
        val, ts = _hist_cache[symbol]
        if now - ts < _CACHE_TTL:
            return val
    stats = {"hv20": None, "atr14": None}
    try:
        df = yf.Ticker(symbol).history(period="4mo")
        if len(df) >= 20:
            rets = df["Close"].pct_change().dropna()
            stats["hv20"] = float(rets.tail(20).std() * math.sqrt(252))
            hi, lo, cl = df["High"], df["Low"], df["Close"]
            tr = (hi - lo).combine((hi - cl.shift()).abs(), max).combine(
                (lo - cl.shift()).abs(), max)
            stats["atr14"] = float(tr.tail(14).mean())
    except Exception:
        pass
    _hist_cache[symbol] = (stats, now)
    return stats


def _score_contract(row, spot, dte, option_type, hv20, atr14):
    """Score one contract. Returns None if ineligible, else a dict."""
    strike = float(row.get("strike") or 0)
    oi = int(row.get("openInterest") or 0)
    volume = int(row.get("volume") or 0)
    iv = float(row.get("impliedVolatility") or 0)
    bid = float(row.get("bid") or 0)
    ask = float(row.get("ask") or 0)
    last = float(row.get("lastPrice") or 0)

    if strike <= 0 or oi < config.GRADER_MIN_OI or iv <= 0.005:
        return None

    # --- Mid price: live quote preferred, lastPrice fallback (flagged) ---
    estimated = False
    if bid > 0 and ask > bid:
        mid = (bid + ask) / 2
        spread_pct = (ask - bid) / mid
        if spread_pct > config.GRADER_MAX_SPREAD_PCT:
            return None
    elif last >= config.GRADER_MIN_MID:
        mid = last
        spread_pct = None  # unknown; liquidity spread component scores 0
        estimated = True
    else:
        return None
    if mid < config.GRADER_MIN_MID:
        return None

    T = max(dte / 365.0, 1 / 365.0)
    delta = black_scholes_delta(spot, strike, T, RISK_FREE_RATE, iv, option_type)
    abs_delta = abs(delta)
    if abs_delta < config.GRADER_DELTA_MIN or abs_delta > config.GRADER_DELTA_MAX:
        return None

    breakeven = strike + mid if option_type == "call" else strike - mid
    pop = _prob_beyond_breakeven(spot, breakeven, T, iv, option_type)
    theta_daily = _bs_theta_daily(spot, strike, T, RISK_FREE_RATE, iv, option_type)
    theta_pct_daily = abs(theta_daily) / mid if mid > 0 else 1.0

    # --- Component scores (0–100) ---
    # Liquidity: OI (50) + volume (25) + spread tightness (25)
    oi_score = min(50.0, 50.0 * math.log10(max(oi, 1)) / 4.0)          # 10k OI = 50
    vol_score = min(25.0, 25.0 * math.log10(volume + 1) / math.log10(5001))
    if spread_pct is None:
        spread_score = 0.0
    else:
        spread_score = 25.0 * max(0.0, 1.0 - spread_pct / config.GRADER_MAX_SPREAD_PCT)
    liquidity = oi_score + vol_score + spread_score

    # Value: contract IV vs underlying 20d HV (cheap vol scores high)
    if hv20 and hv20 > 0.01:
        ratio = iv / hv20
        value = 100.0 * max(0.0, min(1.0, (2.5 - ratio) / 1.5))  # <=1.0 -> 100, >=2.5 -> 0
    else:
        value = 50.0  # no HV context: neutral

    # Probability: risk-neutral prob of profit at expiry
    probability = max(0.0, min(100.0, pop * 100.0 * 1.3))  # PoP 77%+ maxes out

    # Risk efficiency: theta burn + breakeven distance vs ATR
    theta_score = 100.0 * max(0.0, min(1.0, (0.03 - theta_pct_daily) / 0.025))  # <=0.5%/d -> 100
    if atr14 and atr14 > 0:
        be_dist_atr = abs(breakeven - spot) / atr14
        be_score = 100.0 * max(0.0, min(1.0, (5.0 - be_dist_atr) / 4.0))  # <=1 ATR -> 100
    else:
        be_score = 50.0
    risk_eff = (theta_score + be_score) / 2

    quality = (config.GRADER_W_LIQUIDITY * liquidity
               + config.GRADER_W_VALUE * value
               + config.GRADER_W_PROBABILITY * probability
               + config.GRADER_W_RISK * risk_eff)

    return {
        "strike": strike, "dte": dte, "type": option_type,
        "mid": round(mid, 2), "estimated": estimated,
        "spread_pct": round(spread_pct * 100, 1) if spread_pct is not None else None,
        "oi": oi, "volume": volume, "iv": round(iv * 100, 1),
        "delta": round(delta, 2), "breakeven": round(breakeven, 2),
        "prob_profit": round(pop * 100), "max_risk": round(mid * 100),
        "theta_day": round(theta_daily * 100, 2),
        "moneyness": "ITM" if ((option_type == "call" and strike < spot)
                               or (option_type == "put" and strike > spot)) else "OTM",
        "liquidity_score": round(liquidity),
        "value_score": round(value),
        "probability_score": round(probability),
        "risk_score": round(risk_eff),
        "quality": round(quality, 1),
        "quality_letter": _letter(quality),
    }


def _tier_fit(contract, tier_cfg):
    """0–100 fit of a contract to a tier's delta band + DTE preference."""
    lo, hi = tier_cfg["delta"]
    d = abs(contract["delta"])
    if lo <= d <= hi:
        delta_score = 100.0
    else:
        gap = (lo - d) if d < lo else (d - hi)
        delta_score = 100.0 * max(0.0, 1.0 - gap / 0.15)

    dlo, dhi = tier_cfg["dte"]
    dte = contract["dte"]
    if dlo <= dte <= dhi:
        dte_score = 100.0
    else:
        gap = (dlo - dte) if dte < dlo else (dte - dhi)
        dte_score = 100.0 * max(0.0, 1.0 - gap / 20.0)

    return 0.7 * delta_score + 0.3 * dte_score


def _why_this_contract(tier_name, c):
    if tier_name == "conservative":
        return ("Higher-delta, generally in-the-money exposure with more "
                "stock sensitivity and a higher estimated probability of "
                "finishing beyond breakeven.")
    if tier_name == "balanced":
        return ("Near-the-money exposure balancing premium cost, directional "
                "sensitivity, time, and upside potential.")
    return ("Lower-delta, generally out-of-the-money exposure with lower "
            "premium, higher leverage, and a lower estimated probability "
            "of profit.")


def grade_chain(symbol, allow_earnings=False):
    """Grade a symbol's option chain for swing trades.

    Returns dict: direction block, chain stats, earnings guard state, and
    three tier picks (or a reason each is empty).
    """
    symbol = symbol.upper()

    # ── 1. Direction from Trend Confluence ──
    from confluence import analyze_ticker
    yahoo_symbol = {"SPX": "^GSPC", "NDX": "^NDX"}.get(symbol, symbol)
    conf = analyze_ticker(yahoo_symbol)
    if conf is None:
        raise RuntimeError(f"Could not run confluence analysis for {symbol}")

    signal = conf.get("signal", "NO SIGNAL")
    long_count = conf.get("long_count", 0)
    short_count = conf.get("short_count", 0)
    total_ind = conf.get("total_indicators", 12)

    if "LONG" in signal or (signal == "NO SIGNAL" and long_count >= short_count):
        side, option_type = "CALLS", "call"
        lean_count = long_count
    else:
        side, option_type = "PUTS", "put"
        lean_count = short_count
    directional = signal != "NO SIGNAL"
    confidence_pct = round(100 * lean_count / max(total_ind, 1))

    # Top confluence reasons aligned with the chosen direction
    want = 1 if option_type == "call" else -1
    reasons = [v["label"] for v in conf.get("scores", {}).values()
               if v.get("score") == want][:3]

    # ── 2. Earnings guard ──
    chain_symbol = {"SPX": "^SPX", "NDX": "QQQ"}.get(symbol, symbol)
    # Indices have no earnings; only look up real tickers
    earnings = None if yahoo_symbol.startswith("^") else _get_next_earnings(yahoo_symbol)
    earnings_in_days = None
    earnings_blocked = False
    if earnings:
        earnings_in_days = (earnings - datetime.now()).days
        if 0 <= earnings_in_days <= config.GRADER_EARNINGS_EXCLUDE_DAYS and not allow_earnings:
            earnings_blocked = True

    # ── 3. Chain fetch + stats (shared cache with the GEX ladder) ──
    stats = chain_stats(chain_symbol, max_dte=config.GRADER_DTE_MAX)
    chain = get_chain(chain_symbol, max_dte=config.GRADER_DTE_MAX)
    spot = chain["spot"]
    und = _get_underlying_stats(yahoo_symbol)

    # ── 4. Score every contract in the swing window ──
    candidates = []
    quoted = 0
    for exp_str, v in chain["expirations"].items():
        dte = v["dte"]
        if dte < config.GRADER_DTE_MIN or dte > config.GRADER_DTE_MAX:
            continue
        for row in v["calls" if option_type == "call" else "puts"]:
            quoted += 1
            c = _score_contract(row, spot, dte, option_type,
                                und["hv20"], und["atr14"])
            if c is not None:
                c["expiration"] = exp_str
                if earnings_in_days is not None and 0 <= earnings_in_days <= dte:
                    c["earnings_risk"] = True
                candidates.append(c)

    # ── 5. Pick best per tier ──
    tiers = {}
    for tier_name, tier_cfg in config.GRADER_TIERS.items():
        fits = []
        for c in candidates:
            fit = _tier_fit(c, tier_cfg)
            if fit < 40:
                continue
            rec = (config.GRADER_W_QUALITY_IN_REC * c["quality"]
                   + (1 - config.GRADER_W_QUALITY_IN_REC) * fit)
            fits.append((rec, fit, c))
        fits.sort(key=lambda x: -x[0])
        if fits and not earnings_blocked:
            rec, fit, best = fits[0]
            pick = dict(best)
            pick["tier_fit"] = round(fit)
            pick["recommendation"] = round(rec, 1)
            pick["rec_letter"] = _letter(rec)
            pick["tagline"] = tier_cfg["tagline"]
            pick["why"] = _why_this_contract(tier_name, pick)
            pick["rank_note"] = (
                f"Ranks first among {len(fits)} recommendation-eligible "
                f"{tier_name} candidates using "
                f"{round(config.GRADER_W_QUALITY_IN_REC * 100)}% contract "
                f"quality and {round((1 - config.GRADER_W_QUALITY_IN_REC) * 100)}% "
                f"risk-tier fit.")
            tiers[tier_name] = pick
        else:
            tiers[tier_name] = None

    # ── 6. GEX context + narrative ──
    gex_line = ""
    try:
        from gex import get_gex_signal
        custom = symbol not in ("SPX", "NDX")
        gs = get_gex_signal(index="NDX" if symbol == "NDX" else "SPX",
                            symbol=symbol if custom else None)
        if gs.get("regime") not in (None, "UNKNOWN"):
            walls = ""
            if gs.get("call_wall") and gs.get("put_wall"):
                walls = (f" Dealer walls sit at {gs['put_wall']:g} (put) and "
                         f"{gs['call_wall']:g} (call).")
            gex_line = (f" Dealers are {gs['regime'].lower()} on "
                        f"{symbol}.{walls}")
    except Exception:
        pass

    med_iv = f"{stats['median_iv'] * 100:.1f}%" if stats.get("median_iv") else "n/a"
    take = (
        f"{symbol} confluence currently reads {signal} "
        f"({long_count} bullish / {short_count} bearish of {total_ind} "
        f"indicators), so the grader is evaluating {side.lower()}. "
        f"The {config.GRADER_DTE_MIN}–{config.GRADER_DTE_MAX} DTE chain "
        f"contains {quoted} quoted contracts, with {len(candidates)} meeting "
        f"eligibility filters (OI ≥ {config.GRADER_MIN_OI}, spread ≤ "
        f"{round(config.GRADER_MAX_SPREAD_PCT * 100)}%, |delta| "
        f"{config.GRADER_DELTA_MIN}–{config.GRADER_DELTA_MAX}); median "
        f"implied volatility is {med_iv} and aggregate open interest is "
        f"{stats['total_oi']:,}.{gex_line} This is a model-based research "
        f"view, not a guarantee; confirm live quotes, Greeks, earnings "
        f"timing, and position size with your broker.")

    if earnings_blocked:
        take = (f"⚠ {symbol} reports earnings in {earnings_in_days} day(s). "
                f"Long-option swing entries this close to earnings are exposed "
                f"to IV crush, so contract recommendations are suppressed. ") + take

    return {
        "symbol": symbol,
        "spot": round(spot, 2),
        "direction": side,
        "directional": directional,
        "signal": signal,
        "confidence_pct": confidence_pct,
        "reasons": reasons,
        "earnings_date": earnings.strftime("%Y-%m-%d") if earnings else None,
        "earnings_in_days": earnings_in_days,
        "earnings_blocked": earnings_blocked,
        "chain": {"quoted": quoted, "eligible": len(candidates),
                  "median_iv": stats.get("median_iv"),
                  "total_oi": stats.get("total_oi"),
                  "expirations": stats.get("expirations")},
        "tiers": tiers,
        "our_take": take,
        "dte_window": [config.GRADER_DTE_MIN, config.GRADER_DTE_MAX],
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
