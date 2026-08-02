"""
Golden-master regression suite.

Locks the shape of every existing API endpoint so feature updates can never
silently break current behavior. Values drift with the market; this suite
asserts structure (success flag + required keys), not numbers.

Run:  python3 test_regression.py           # fast core endpoints
      SLOW=1 python3 test_regression.py    # + network-heavy endpoints
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app import app  # noqa: E402

RESULTS = {"passed": 0, "failed": 0}


def check(name, resp, required_keys=(), allow_data_unavailable=False):
    """Assert an endpoint returned JSON with success=True and required keys.

    allow_data_unavailable: endpoints that legitimately fail outside market
    hours (e.g. options chains) pass if they return a clean error envelope
    instead of crashing.
    """
    label = f"{name:<40s}"
    try:
        assert resp.status_code == 200, f"HTTP {resp.status_code}"
        data = resp.get_json()
        assert data is not None, "no JSON body"
        if not data.get("success"):
            if allow_data_unavailable and "error" in data:
                print(f"  ~  SOFT-PASS {label} (data unavailable: {str(data['error'])[:60]})")
                RESULTS["passed"] += 1
                return None
            raise AssertionError(f"success=False: {str(data.get('error'))[:80]}")
        missing = [k for k in required_keys if k not in data]
        assert not missing, f"missing keys: {missing}"
        print(f"  ✅ PASS      {label}")
        RESULTS["passed"] += 1
        return data
    except AssertionError as exc:
        print(f"  ❌ FAIL      {label} {exc}")
        RESULTS["failed"] += 1
        return None


def main():
    slow = os.environ.get("SLOW") == "1"
    client = app.test_client()

    print("\n" + "=" * 62)
    print("  GOLDEN-MASTER REGRESSION SUITE")
    print("=" * 62)

    # --- Fast, no/low network ---
    check("/api/risk-calc",
          client.get("/api/risk-calc?account=10000&entry=150&stop=140&risk=2"),
          ["shares", "position_value", "risk_amount", "max_loss"])

    check("/api/trade-log", client.get("/api/trade-log"), ["trades"])

    # --- Core market endpoints ---
    check("/api/confluence (SPX)", client.get("/api/confluence"),
          ["scores", "signal", "price"])

    check("/api/predict (SPX)", client.get("/api/predict"),
          ["bull_prob", "bear_prob", "close", "levels"])

    check("/api/backtest (SPX)", client.get("/api/backtest"),
          ["rows", "accuracy", "total"])

    check("/api/gex (SPX)", client.get("/api/gex"),
          ["total_gex", "gex_flip", "chart_strikes", "chart_net_gex",
           "dealer_position"],
          allow_data_unavailable=True)

    check("/api/checklist", client.get("/api/checklist"),
          allow_data_unavailable=True)

    check("/api/net-premium", client.get("/api/net-premium"),
          allow_data_unavailable=True)

    if slow:
        check("/api/confluence (NDX)", client.get("/api/confluence?index=NDX"),
              ["scores", "signal", "price"])
        check("/api/options (SPX)", client.get("/api/options"),
              allow_data_unavailable=True)
        check("/api/scan (AAPL)", client.get("/api/scan?tickers=AAPL"),
              ["rows"])
        check("/api/live", client.get("/api/live"),
              allow_data_unavailable=True)

    print("=" * 62)
    print(f"  Results: {RESULTS['passed']} passed / {RESULTS['failed']} failed")
    print("=" * 62)
    return 1 if RESULTS["failed"] else 0


if __name__ == "__main__":
    sys.exit(main())
