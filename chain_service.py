"""
Shared per-symbol options-chain fetch + cache.

One chain fetch powers the GEX/VEX/DEX ladder, per-ticker dealer
positioning, and (v1.3) the options contract grader. Chains are cached to
disk for 30 minutes per symbol with atomic writes, plus an in-memory layer,
so repeated calls within a session cost nothing.

The canonical cache window is 120 DTE; callers request any max_dte <= 120
and get a filtered view of the same cached data.
"""

import os
import json
import time
import tempfile
from datetime import datetime

import pandas as pd
import yfinance as yf

_BASE = os.path.dirname(os.path.abspath(__file__))
_CACHE_DIR = os.path.join(_BASE, "cache")

CHAIN_TTL_SECONDS = 30 * 60
CANONICAL_MAX_DTE = 120

_KEEP_COLS = ["strike", "bid", "ask", "lastPrice", "openInterest",
              "volume", "impliedVolatility"]

_mem_cache = {}  # symbol -> (data, ts)


def _safe_name(symbol):
    return symbol.lower().replace("^", "idx_").replace("=", "_").replace("/", "_")


def _cache_path(symbol):
    return os.path.join(_CACHE_DIR, f"chain_{_safe_name(symbol)}.json")


def _atomic_write_json(data, path):
    """Atomic JSON write (temp + os.replace) — same pattern as the rest of
    the codebase, bypasses macOS iCloud provenance issues."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path), suffix=".tmp")
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


def _records(df):
    """Reduce a yfinance chain DataFrame to the columns we keep, as plain
    JSON-safe records (NaN -> None)."""
    cols = [c for c in _KEEP_COLS if c in df.columns]
    return json.loads(df[cols].to_json(orient="records"))


def get_chain(symbol, max_dte=45, force_refresh=False):
    """Fetch (or serve cached) options chain data for any optionable symbol.

    Returns::

        {"symbol", "spot", "fetched_at",
         "expirations": {exp_str: {"dte", "calls": [...], "puts": [...]}}}

    where calls/puts rows have: strike, bid, ask, lastPrice, openInterest,
    volume, impliedVolatility. Raises RuntimeError when the symbol has no
    usable options data.
    """
    symbol = symbol.upper()

    data = None
    # ---- In-memory cache ----
    if not force_refresh and symbol in _mem_cache:
        cached, ts = _mem_cache[symbol]
        if time.time() - ts < CHAIN_TTL_SECONDS:
            data = cached

    # ---- Disk cache ----
    if data is None and not force_refresh:
        path = _cache_path(symbol)
        if os.path.exists(path):
            try:
                with open(path) as f:
                    cached = json.load(f)
                fetched = datetime.fromisoformat(cached["fetched_at"])
                if (datetime.now() - fetched).total_seconds() < CHAIN_TTL_SECONDS:
                    data = cached
                    _mem_cache[symbol] = (data, time.time())
            except Exception:
                data = None

    # ---- Fresh fetch (canonical 120-DTE window) ----
    if data is None:
        data = _fetch_fresh(symbol)
        _mem_cache[symbol] = (data, time.time())
        try:
            _atomic_write_json(data, _cache_path(symbol))
        except Exception:
            pass  # cache write failure is non-fatal

    # ---- Filter to requested DTE window ----
    if max_dte >= CANONICAL_MAX_DTE:
        return data
    view = dict(data)
    view["expirations"] = {e: v for e, v in data["expirations"].items()
                           if v["dte"] <= max_dte}
    return view


def _fetch_fresh(symbol):
    tk = yf.Ticker(symbol)
    hist = tk.history(period="1d")
    if hist.empty:
        raise RuntimeError(f"Could not fetch price for {symbol}")
    spot = float(hist["Close"].iloc[-1])

    try:
        expirations = tk.options
    except Exception:
        expirations = None
    if not expirations:
        raise RuntimeError(f"No options available for {symbol}")

    now = datetime.now()
    exps = {}
    for exp_str in expirations:
        try:
            dte = (datetime.strptime(exp_str, "%Y-%m-%d") - now).days
        except ValueError:
            continue
        if dte <= 0 or dte > CANONICAL_MAX_DTE:
            continue
        try:
            chain = tk.option_chain(exp_str)
        except Exception:
            continue
        exps[exp_str] = {"dte": dte,
                         "calls": _records(chain.calls),
                         "puts": _records(chain.puts)}

    if not exps:
        raise RuntimeError(
            f"No options data for {symbol} within {CANONICAL_MAX_DTE} DTE "
            "(market may be closed or chain is empty)")

    return {"symbol": symbol, "spot": spot,
            "fetched_at": now.isoformat(), "expirations": exps}


def get_frames(symbol, max_dte=45, force_refresh=False):
    """Chain data with calls/puts as DataFrames — the shape the GEX engine
    consumes. Returns (spot, {exp_str: {"dte", "calls": df, "puts": df}})."""
    data = get_chain(symbol, max_dte=max_dte, force_refresh=force_refresh)
    frames = {}
    for exp_str, v in data["expirations"].items():
        frames[exp_str] = {
            "dte": v["dte"],
            "calls": pd.DataFrame(v["calls"]),
            "puts": pd.DataFrame(v["puts"]),
        }
    return data["spot"], frames


def chain_stats(symbol, max_dte=45):
    """Aggregate liquidity stats for a symbol's chain (used for the
    thin-chain guard and, later, the grader's 'Our Take' summary)."""
    data = get_chain(symbol, max_dte=max_dte)
    n_contracts = 0
    total_oi = 0
    total_volume = 0
    ivs = []
    for v in data["expirations"].values():
        for side in ("calls", "puts"):
            for row in v[side]:
                n_contracts += 1
                total_oi += int(row.get("openInterest") or 0)
                total_volume += int(row.get("volume") or 0)
                iv = row.get("impliedVolatility")
                if iv and iv > 0.005:
                    ivs.append(float(iv))
    ivs.sort()
    median_iv = ivs[len(ivs) // 2] if ivs else None
    return {"symbol": data["symbol"], "spot": data["spot"],
            "contracts": n_contracts, "total_oi": total_oi,
            "total_volume": total_volume, "median_iv": median_iv,
            "expirations": len(data["expirations"]),
            "fetched_at": data["fetched_at"]}
