"""
Shared earnings-date lookup — used by the options grader (IV-crush guard)
and the per-ticker ML signal (earnings-gap warning).

yfinance calendar lookups are cached in-memory for 6 hours. Index symbols
(^GSPC, ^NDX, ...) have no earnings and always return None.
"""

import time
from datetime import datetime

import yfinance as yf

_cache = {}  # symbol -> (datetime|None, ts)
_CACHE_TTL = 6 * 3600


def get_next_earnings(symbol):
    """Next earnings datetime for symbol, or None."""
    if not symbol or symbol.startswith("^"):
        return None
    symbol = symbol.upper()
    now = time.time()
    if symbol in _cache:
        val, ts = _cache[symbol]
        if now - ts < _CACHE_TTL:
            return val
    result = None
    try:
        cal = yf.Ticker(symbol).calendar
        dates = None
        if isinstance(cal, dict):
            dates = cal.get("Earnings Date")
        elif cal is not None and hasattr(cal, "loc"):
            try:
                dates = list(cal.loc["Earnings Date"])
            except Exception:
                dates = None
        if dates:
            first = dates[0] if isinstance(dates, (list, tuple)) else dates
            result = datetime(first.year, first.month, first.day)
    except Exception:
        result = None
    _cache[symbol] = (result, now)
    return result


def days_to_earnings(symbol):
    """Days until next earnings, or None when unknown / not applicable."""
    e = get_next_earnings(symbol)
    if e is None:
        return None
    return (e - datetime.now()).days
