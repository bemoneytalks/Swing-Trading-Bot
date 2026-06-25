"""
Trade Card — stores and retrieves trade cards from trade_log.json.
Every trade logged here before entry. No written plan = no entry.
"""

import json
import os
import tempfile
from datetime import datetime


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _atomic_write_json(data, path):
    """Atomic JSON write via temp+rename — avoids os.getcwd() (EPERM in Launch Agent)."""
    if not os.path.isabs(path):
        path = os.path.join(_BASE_DIR, path)
    dir_ = os.path.dirname(path) or _BASE_DIR
    os.makedirs(dir_, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix=".tmp")
    try:
        os.close(fd)
        with open(tmp, "w") as f:
            json.dump(data, f, indent=2, default=str)
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

TRADE_LOG_PATH = "trade_log.json"


def save_trade_card(card):
    """Append a trade card to trade_log.json. Creates file if missing."""
    card["logged_at"] = datetime.now().isoformat()

    log = []
    if os.path.exists(TRADE_LOG_PATH):
        try:
            with open(TRADE_LOG_PATH, "r") as f:
                log = json.load(f)
            if not isinstance(log, list):
                log = []
        except Exception:
            log = []

    log.append(card)

    _atomic_write_json(log, TRADE_LOG_PATH)

    return {"saved": True, "total_trades": len(log), "path": TRADE_LOG_PATH}


def get_trade_log():
    """Return full trade log array."""
    if not os.path.exists(TRADE_LOG_PATH):
        return []
    try:
        with open(TRADE_LOG_PATH, "r") as f:
            log = json.load(f)
        return log if isinstance(log, list) else []
    except Exception:
        return []


def get_recent_trades(n=20):
    """Return last N trades formatted for dashboard display."""
    log = get_trade_log()
    recent = log[-n:] if len(log) > n else log
    recent.reverse()  # newest first

    formatted = []
    for t in recent:
        formatted.append({
            "date":       t.get("logged_at", "")[:10],
            "contract":   t.get("contract", ""),
            "direction":  t.get("direction", ""),
            "entry_price": t.get("entry_price", ""),
            "contracts":  t.get("contracts", ""),
            "tier":       t.get("entry_tier", ""),
            "spx_entry":  t.get("spx_at_entry", ""),
            "stop_spx":   t.get("stop_spx_level", ""),
            "t1_spx":     t.get("t1_spx_level", ""),
            "t2_spx":     t.get("t2_spx_level", ""),
            "thesis":     t.get("thesis", ""),
            "score":      t.get("checklist_score", ""),
            "max_loss":   t.get("max_loss_dollars", ""),
        })
    return formatted
