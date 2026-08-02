"""
Unit tests for the options grader — synthetic contracts, no network.

Run: python3 test_grader.py
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from options_grader import _score_contract, _tier_fit, _letter

PASS = {"n": 0}
FAIL = {"n": 0}


def check(name, cond, detail=""):
    if cond:
        PASS["n"] += 1
        print(f"  ✅ PASS  {name}")
    else:
        FAIL["n"] += 1
        print(f"  ❌ FAIL  {name} {detail}")


def row(**kw):
    """A liquid, tight-spread ATM-ish call row unless overridden."""
    base = {"strike": 100.0, "bid": 5.0, "ask": 5.4, "lastPrice": 5.2,
            "openInterest": 5000, "volume": 800, "impliedVolatility": 0.40}
    base.update(kw)
    return base


SPOT, DTE, HV, ATR = 100.0, 60, 0.35, 2.0


def main():
    print("\n" + "=" * 62)
    print("  OPTIONS GRADER UNIT TESTS")
    print("=" * 62)

    # ── Eligibility filters ──
    good = _score_contract(row(), SPOT, DTE, "call", HV, ATR)
    check("liquid ATM call is eligible", good is not None)

    check("OI below minimum rejected",
          _score_contract(row(openInterest=config.GRADER_MIN_OI - 1),
                          SPOT, DTE, "call", HV, ATR) is None)

    check("wide spread rejected",
          _score_contract(row(bid=4.0, ask=6.0), SPOT, DTE, "call", HV, ATR) is None)

    check("zero IV rejected",
          _score_contract(row(impliedVolatility=0), SPOT, DTE, "call", HV, ATR) is None)

    check("deep OTM (delta < min) rejected",
          _score_contract(row(strike=180.0, bid=0.15, ask=0.17, lastPrice=0.16),
                          SPOT, DTE, "call", HV, ATR) is None)

    check("no-quote contract falls back to lastPrice as estimated",
          (_score_contract(row(bid=0, ask=0), SPOT, DTE, "call", HV, ATR) or {}).get("estimated") is True)

    check("worthless no-quote contract rejected",
          _score_contract(row(bid=0, ask=0, lastPrice=0.05),
                          SPOT, DTE, "call", HV, ATR) is None)

    # ── Score sanity ──
    check("quality in 0-100", good and 0 <= good["quality"] <= 100,
          f"quality={good and good['quality']}")
    check("component scores in 0-100",
          good and all(0 <= good[k] <= 100 for k in
                       ("liquidity_score", "value_score",
                        "probability_score", "risk_score")))
    check("ITM call flagged ITM",
          (_score_contract(row(strike=85.0, bid=16.0, ask=16.6, lastPrice=16.2),
                           SPOT, DTE, "call", HV, ATR) or {}).get("moneyness") == "ITM")
    check("call breakeven = strike + mid",
          good and abs(good["breakeven"] - (100.0 + good["mid"])) < 0.01)

    put = _score_contract(row(), SPOT, DTE, "put", HV, ATR)
    check("put delta negative", put and put["delta"] < 0)
    check("put breakeven = strike - mid",
          put and abs(put["breakeven"] - (100.0 - put["mid"])) < 0.01)

    # ── Cheap vol beats rich vol on the value component ──
    cheap = _score_contract(row(impliedVolatility=0.30), SPOT, DTE, "call", HV, ATR)
    rich = _score_contract(row(impliedVolatility=0.80, bid=10.0, ask=10.6, lastPrice=10.3),
                           SPOT, DTE, "call", HV, ATR)
    check("cheaper IV scores higher value",
          cheap and rich and cheap["value_score"] > rich["value_score"])

    # ── Letter boundaries ──
    for score, want in [(95, "A"), (90, "A"), (89.9, "B"), (75, "B"),
                        (74.9, "C"), (60, "C"), (59.9, "D"), (45, "D"),
                        (44.9, "F"), (0, "F")]:
        check(f"letter({score}) == {want}", _letter(score) == want,
              f"got {_letter(score)}")

    # ── Tier fit ──
    bal = config.GRADER_TIERS["balanced"]
    in_band = {"delta": 0.52, "dte": 60}
    check("in-band contract fits 100", _tier_fit(in_band, bal) == 100.0)
    off_delta = {"delta": 0.90, "dte": 60}
    check("far-off delta scores low", _tier_fit(off_delta, bal) < 40,
          f"got {_tier_fit(off_delta, bal)}")
    near = {"delta": 0.62, "dte": 60}
    check("near-band beats far-band",
          _tier_fit(near, bal) > _tier_fit(off_delta, bal))

    print("=" * 62)
    print(f"  Results: {PASS['n']} passed / {FAIL['n']} failed")
    print("=" * 62)
    return 1 if FAIL["n"] else 0


if __name__ == "__main__":
    sys.exit(main())
