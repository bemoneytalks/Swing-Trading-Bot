"""Fetch and cache SPX/NDX historical data."""

import os
import time
import json
import tempfile
import subprocess
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import config

# Absolute path to the project directory, derived from __file__ so it works
# in the Launch Agent context where os.getcwd() raises EPERM (launchd runs
# with iCloud Drive as CWD but doesn't grant getcwd access to Python.app).
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _safe_remove(path):
    """Delete a file before rewriting it.
    macOS com.apple.provenance xattr cannot be stripped — the only reliable
    fix is to remove the old file so the new write creates a clean one.
    """
    try:
        if os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _abs(path):
    """Resolve a relative path against the project directory without os.getcwd().
    os.getcwd() raises EPERM in the Launch Agent context (iCloud Drive CWD)."""
    if os.path.isabs(path):
        return path
    return os.path.join(_BASE_DIR, path)


def _atomic_write_csv(df, path):
    """Write DataFrame to CSV atomically via temp file + os.replace().
    Bypasses macOS iCloud provenance (EPERM) and avoids os.getcwd()."""
    path = _abs(path)
    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        os.close(fd)
        df.to_csv(tmp)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def _atomic_write_json(data, path):
    """Write JSON atomically via temp file + os.replace().
    Bypasses macOS iCloud provenance (EPERM) and avoids os.getcwd()."""
    path = _abs(path)
    dir_ = os.path.dirname(path)
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        os.close(fd)
        with open(tmp, "w") as f:
            json.dump(data, f)
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError:
            pass
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except Exception:
            pass
        raise


def _cache_path(cache_name="spx_daily.csv"):
    os.makedirs(config.CACHE_DIR, exist_ok=True)
    return os.path.join(config.CACHE_DIR, cache_name)


def _cache_meta_path(cache_name="spx_daily.csv"):
    # Strip .csv extension and append _meta.json
    base = cache_name.replace(".csv", "")
    return os.path.join(config.CACHE_DIR, f"{base}_meta.json")


def _cache_is_fresh(ticker, cache_name="spx_daily.csv"):
    meta = _cache_meta_path(cache_name)
    if not os.path.exists(meta):
        return False
    with open(meta) as f:
        info = json.load(f)
    fetched = datetime.fromisoformat(info["fetched_at"])

    # Check if cache includes the most recent trading day's close
    # If it's after 4:30pm ET on a weekday and cache was fetched before today, it's stale
    now = datetime.now()
    age_hours = (now - fetched).total_seconds() / 3600

    # Always stale if older than cache expiry
    if age_hours > config.CACHE_EXPIRY_HOURS:
        return False

    # Check if a new trading day has closed since cache was fetched
    # Load cached data to check the last date
    cache = _cache_path(cache_name)
    if os.path.exists(cache):
        try:
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            last_cached_date = df.index[-1].date()
            # Fetch minimal fresh data to see if there's a newer close
            fresh = yf.Ticker(ticker).history(period="5d")
            if not fresh.empty:
                latest_date = fresh.index[-1]
                if hasattr(latest_date, 'tz') and latest_date.tz:
                    latest_date = latest_date.tz_localize(None) if hasattr(latest_date, 'tz_localize') else latest_date.replace(tzinfo=None)
                if latest_date.date() > last_cached_date:
                    return False  # New trading day available
        except Exception:
            pass

    return True


def fetch_index_data(ticker, cache_name, force_refresh=False):
    """Fetch 5 years of daily OHLCV data for any ticker. Uses cache if fresh."""
    cache = _cache_path(cache_name)

    if not force_refresh and _cache_is_fresh(ticker, cache_name) and os.path.exists(cache):
        print(f"[DATA] Loading {cache_name} from cache...")
        df = pd.read_csv(cache, index_col=0, parse_dates=True)
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        print(f"[DATA] {len(df)} trading days loaded (cached)")
        return df

    print(f"[DATA] Fetching {ticker} data from Yahoo Finance...")
    end = datetime.now()
    start = end - timedelta(days=config.DATA_PERIOD_YEARS * 365)

    tk = yf.Ticker(ticker)
    df = tk.history(start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"))

    if df.empty:
        raise RuntimeError(f"Failed to fetch {ticker} data. Check your internet connection.")

    # Keep only OHLCV columns
    df = df[["Open", "High", "Low", "Close", "Volume"]]
    df.index.name = "Date"

    # Drop incomplete rows — yfinance sometimes returns the latest day with
    # Volume but NaN OHLC, which poisons predictions (NaN close → invalid JSON)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # Use atomic temp-file + rename writes — macOS com.apple.provenance blocks
    # in-place overwrites from Python.app, but os.replace() (rename) is not
    # affected by xattrs on the destination file.
    _atomic_write_csv(df, cache)
    _atomic_write_json({"fetched_at": datetime.now().isoformat(), "rows": len(df)},
                       _cache_meta_path(cache_name))

    print(f"[DATA] {len(df)} trading days fetched ({df.index[0].date()} to {df.index[-1].date()})")
    return df


def fetch_spx_data(force_refresh=False):
    """Fetch 5 years of daily SPX OHLCV data. Uses cache if fresh."""
    return fetch_index_data(config.TICKER, "spx_daily.csv", force_refresh)
