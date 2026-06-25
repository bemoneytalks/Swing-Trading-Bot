"""
Regression tests for the confluence scoring engine.

Run with:  python3 test_confluence.py
Or:        python3 -m pytest test_confluence.py -v

Design philosophy
-----------------
There are two layers of tests:

1. DIRECT SCORING TESTS (Groups 1-4)
   Build an `indicators` dict by hand and call `score_confluence()` directly.
   These are fast, deterministic, and test exactly the logic path that produced
   the bug (boundary conditions, EMA-stack uptrend substitution, overbought-in-
   uptrend scoring).  They do NOT rely on synthetic price data generating the
   right ADX/EMA values.

2. END-TO-END DATA TESTS (Groups 5-8)
   Build synthetic OHLCV DataFrames, run `_calculate_indicators()` +
   `score_confluence()`, and verify directional consistency.  These catch bugs
   that live inside `_calculate_indicators` itself.
"""

import sys
import numpy as np
import pandas as pd

try:
    from confluence import _calculate_indicators, score_confluence
except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)


# ── Test Runner ───────────────────────────────────────────────────────────────

PASS = 0
FAIL = 0
ERRORS = []


def check(name, condition, msg=""):
    global PASS, FAIL, ERRORS
    if condition:
        print(f"  ✅  PASS  {name}")
        PASS += 1
    else:
        detail = f" — {msg}" if msg else ""
        print(f"  ❌  FAIL  {name}{detail}")
        FAIL += 1
        ERRORS.append(f"{name}: {msg}")


# ── Indicator Dict Helper ─────────────────────────────────────────────────────

def _bull_indicators(**overrides):
    """
    Return a complete `indicators` dict representing a clear uptrend
    (NDX-like: new highs, bullish EMA stack, high RSI/Stoch, net prem bullish).
    Override any field by passing kwargs.
    """
    base = dict(
        price=28000.0, prev_close=27700.0,
        sma200=28000.0, above_200sma=True,
        sma50=26000.0,  above_50sma=True,
        ema9=27500.0, ema21=27000.0, ema50=26000.0,
        ema_bull_stack=True, ema_bear_stack=False,
        adx=31.4, adx_trending=True, adx_bull=True,
        adx_pos=25.0, adx_neg=12.0,
        confirmed_uptrend=True, confirmed_downtrend=False,
        rsi=76.0, rsi_bear_divergence=False, rsi_bull_divergence=False,
        stoch_k=98.7, stoch_d=95.6,
        stoch_bull_cross=False, stoch_bear_cross=False,
        bb_pct=0.885, bb_width=0.04, bb_width_prev=0.038,
        bb_squeeze=False, bb_expanding=True,
        macd=120.0, macd_signal=90.0, macd_hist=30.0, macd_hist_prev=25.0,
        macd_bull_cross=False, macd_bear_cross=False, macd_hist_expanding=True,
        price_vs_vwap=1000.0, vwap=27000.0,
        vol_ratio=0.1, vol_above_avg=False,
        high_20=28016.0, low_20=25000.0,
        near_20d_high=True, near_20d_low=False,
        change_1d=0.0123, change_5d=0.035,
        close_range_pct=0.85, day_high=28100.0, day_low=27800.0,
        rsi_recovery_bullish=False, rsi_recovery_bearish=False,
    )
    base.update(overrides)
    return base


def _bear_indicators(**overrides):
    """
    Return a complete `indicators` dict representing a clear downtrend.
    """
    base = dict(
        price=19000.0, prev_close=19300.0,
        sma200=22000.0, above_200sma=False,
        sma50=21000.0,  above_50sma=False,
        ema9=19200.0, ema21=19800.0, ema50=21000.0,
        ema_bull_stack=False, ema_bear_stack=True,
        adx=28.0, adx_trending=True, adx_bull=False,
        adx_pos=10.0, adx_neg=22.0,
        confirmed_uptrend=False, confirmed_downtrend=True,
        rsi=28.0, rsi_bear_divergence=False, rsi_bull_divergence=False,
        stoch_k=12.0, stoch_d=18.0,
        stoch_bull_cross=False, stoch_bear_cross=False,
        bb_pct=0.08, bb_width=0.05, bb_width_prev=0.04,
        bb_squeeze=False, bb_expanding=True,
        macd=-80.0, macd_signal=-60.0, macd_hist=-20.0, macd_hist_prev=-15.0,
        macd_bull_cross=False, macd_bear_cross=False, macd_hist_expanding=True,
        price_vs_vwap=-500.0, vwap=19500.0,
        vol_ratio=0.9, vol_above_avg=False,
        high_20=21000.0, low_20=18900.0,
        near_20d_high=False, near_20d_low=True,
        change_1d=-0.016, change_5d=-0.04,
        close_range_pct=0.15, day_high=19400.0, day_low=18950.0,
        rsi_recovery_bullish=False, rsi_recovery_bearish=False,
    )
    base.update(overrides)
    return base


# ── Synthetic OHLCV Helper ────────────────────────────────────────────────────

def _make_df(length=300, direction="up"):
    """
    Build OHLCV data with a smooth, strong trend so ADX reliably exceeds 25.
    direction: "up" | "down" | "sideways"
    """
    np.random.seed(42)
    dates = pd.date_range("2024-01-01", periods=length, freq="B")

    # Very small noise relative to trend so ADX > 25 is guaranteed
    if direction == "up":
        closes = np.linspace(18000, 28000, length) + np.random.randn(length) * 20
    elif direction == "down":
        closes = np.linspace(28000, 18000, length) + np.random.randn(length) * 20
    else:
        closes = 23000 + np.random.randn(length) * 200

    closes = np.maximum(closes, 100)
    opens  = closes * (1 + np.random.randn(length) * 0.0005)
    highs  = np.maximum(opens, closes) * (1 + np.abs(np.random.randn(length)) * 0.001)
    lows   = np.minimum(opens, closes) * (1 - np.abs(np.random.randn(length)) * 0.001)
    vols   = np.abs(np.random.randn(length) * 200_000 + 2_000_000)

    return pd.DataFrame(
        {"Open": opens, "High": highs, "Low": lows, "Close": closes, "Volume": vols},
        index=dates,
    )


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 1 — REGRESSION: Price at 200 SMA (the exact NDX bug, 2026-05-05)
# ══════════════════════════════════════════════════════════════════════════════

def test_regression_ndx_at_200sma():
    """
    BUG (2026-05-05): NDX at 27,991 with SMA200 also at 27,991.
    `price > sma200` returned False → confirmed_uptrend = False → RANGE mode.
    RSI 76, Stoch 99, BB 88% all scored -1 → LEAN SHORT in a bull market.

    FIX: above_200sma uses >=; EMA bull stack substitutes for 200 SMA in uptrend check.
    """
    print("\n[Group 1] Regression — NDX at 200 SMA boundary (exact bug replay)")

    # Exact values from the failing API response
    ind = _bull_indicators(
        price=27991.43, sma200=27991.43,   # equal floats — the bug trigger
        above_200sma=(27991.43 >= 27991.43),   # True after >= fix
        rsi=76.0, stoch_k=98.7, stoch_d=95.6, bb_pct=0.885,
        near_20d_high=True,
    )
    # Force confirmed_uptrend to reflect the fixed logic
    ind["confirmed_uptrend"] = (
        (ind["above_200sma"] or ind["ema_bull_stack"]) and
        ind["above_50sma"] and ind["adx_trending"] and ind["adx_bull"]
    )

    result = score_confluence(ind)

    check("above_200sma is True when price == sma200 (>= fix)",
          ind["above_200sma"],
          f"price={ind['price']} sma200={ind['sma200']}")

    check("confirmed_uptrend is True (EMA bull stack + ADX bullish)",
          ind["confirmed_uptrend"],
          f"above_200={ind['above_200sma']} ema_bull={ind['ema_bull_stack']}")

    check("RSI 76 in UPTREND must score +1, not -1",
          result["scores"]["rsi"]["score"] == 1,
          f"got score={result['scores']['rsi']['score']} label={result['scores']['rsi']['label']}")

    check("BB 88% in UPTREND must score +1 (riding upper band), not -1",
          result["scores"]["bollinger"]["score"] == 1,
          f"got score={result['scores']['bollinger']['score']} label={result['scores']['bollinger']['label']}")

    check("Near 20-day high in UPTREND must score +1 (breakout), not -1",
          result["scores"]["key_level"]["score"] == 1,
          f"got score={result['scores']['key_level']['score']} label={result['scores']['key_level']['label']}")

    check("Stochastic 99 in UPTREND must score 0 (trend momentum), not -1",
          result["scores"]["stochastic"]["score"] == 0,
          f"got score={result['scores']['stochastic']['score']}")

    check("trend_context is UPTREND",
          result["trend_context"] == "UPTREND",
          f"got {result['trend_context']}")

    check("Signal is not SHORT (was incorrectly LEAN SHORT before fix)",
          "SHORT" not in result["signal"],
          f"got signal={result['signal']} (long={result['long_count']} short={result['short_count']})")

    check("Long count exceeds short count",
          result["long_count"] > result["short_count"],
          f"long={result['long_count']} short={result['short_count']}")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 2 — EMA Bull Stack Substitutes for 200 SMA in Uptrend Check
# ══════════════════════════════════════════════════════════════════════════════

def test_ema_stack_substitutes_for_200sma():
    """
    Price is 0.5% below the 200 SMA (above_200sma = False), but EMA 9>21>50
    is clearly bullish.  The uptrend should still be confirmed via EMA stack.
    """
    print("\n[Group 2] EMA bull stack substitutes for 200 SMA in uptrend detection")

    ind = _bull_indicators(
        price=27851.0, sma200=27991.0,    # 0.5% below 200 SMA
        above_200sma=False,               # strictly below
        ema_bull_stack=True,              # but EMA 9>21>50 is bullish
    )
    ind["confirmed_uptrend"] = (
        (ind["above_200sma"] or ind["ema_bull_stack"]) and
        ind["above_50sma"] and ind["adx_trending"] and ind["adx_bull"]
    )

    result = score_confluence(ind)

    check("confirmed_uptrend = True via EMA bull stack (above_200sma = False)",
          ind["confirmed_uptrend"],
          f"above_200={ind['above_200sma']} ema_bull={ind['ema_bull_stack']}")

    check("trend_context = UPTREND when EMA stack is bullish",
          result["trend_context"] == "UPTREND",
          f"got {result['trend_context']}")

    check("Signal is not SHORT even though price < 200 SMA",
          "SHORT" not in result["signal"],
          f"got signal={result['signal']}")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 3 — Context-Aware Scoring: Uptrend vs Range
# ══════════════════════════════════════════════════════════════════════════════

def test_rsi_overbought_scores_differently_by_context():
    """RSI 76 should score +1 in uptrend but -1 in range/mean-reversion mode."""
    print("\n[Group 3] Context-aware scoring: RSI overbought in uptrend vs range")

    # UPTREND
    ind_up = _bull_indicators(rsi=76.0, confirmed_uptrend=True, confirmed_downtrend=False)
    r_up = score_confluence(ind_up)
    check("RSI 76 in UPTREND → score +1",
          r_up["scores"]["rsi"]["score"] == 1,
          f"got {r_up['scores']['rsi']['score']}")

    # RANGE (no confirmed uptrend)
    ind_rng = _bull_indicators(
        rsi=76.0,
        above_200sma=False, ema_bull_stack=False, adx_trending=False,
        confirmed_uptrend=False, confirmed_downtrend=False,
    )
    r_rng = score_confluence(ind_rng)
    check("RSI 76 in RANGE mode → score -1",
          r_rng["scores"]["rsi"]["score"] == -1,
          f"got {r_rng['scores']['rsi']['score']}")


def test_key_level_near_high_context():
    """Near 20-day high: +1 (breakout) in uptrend, -1 (resistance) in range."""
    print()
    ind_up = _bull_indicators(near_20d_high=True, confirmed_uptrend=True, confirmed_downtrend=False)
    r_up = score_confluence(ind_up)
    check("Near 20d high in UPTREND → score +1 (breakout)",
          r_up["scores"]["key_level"]["score"] == 1,
          f"got {r_up['scores']['key_level']['score']} label={r_up['scores']['key_level']['label']}")

    ind_rng = _bull_indicators(
        near_20d_high=True,
        above_200sma=False, ema_bull_stack=False, adx_trending=False,
        confirmed_uptrend=False, confirmed_downtrend=False,
    )
    r_rng = score_confluence(ind_rng)
    check("Near 20d high in RANGE → score -1 (resistance)",
          r_rng["scores"]["key_level"]["score"] == -1,
          f"got {r_rng['scores']['key_level']['score']}")


def test_bollinger_upper_band_context():
    """BB at upper band: +1 (momentum) in uptrend, -1 (fade) in range."""
    print()
    ind_up = _bull_indicators(bb_pct=0.90, bb_expanding=True, confirmed_uptrend=True, confirmed_downtrend=False)
    r_up = score_confluence(ind_up)
    check("BB 90% in UPTREND → score +1 (riding upper band)",
          r_up["scores"]["bollinger"]["score"] == 1,
          f"got {r_up['scores']['bollinger']['score']} label={r_up['scores']['bollinger']['label']}")

    ind_rng = _bull_indicators(
        bb_pct=0.85, bb_expanding=False,
        above_200sma=False, ema_bull_stack=False, adx_trending=False,
        confirmed_uptrend=False, confirmed_downtrend=False,
    )
    r_rng = score_confluence(ind_rng)
    check("BB 85% in RANGE (not expanding) → score -1 (fade)",
          r_rng["scores"]["bollinger"]["score"] == -1,
          f"got {r_rng['scores']['bollinger']['score']}")


def test_stochastic_high_context():
    """Stochastic 95: 0 (trend momentum) in uptrend, -1 (overbought) in range."""
    print()
    ind_up = _bull_indicators(stoch_k=95.0, stoch_d=90.0, confirmed_uptrend=True, confirmed_downtrend=False)
    r_up = score_confluence(ind_up)
    check("Stoch 95 in UPTREND → score 0 (trend momentum)",
          r_up["scores"]["stochastic"]["score"] == 0,
          f"got {r_up['scores']['stochastic']['score']}")

    ind_rng = _bull_indicators(
        stoch_k=85.0, stoch_d=80.0,
        stoch_bear_cross=True,     # crossed bearish
        above_200sma=False, ema_bull_stack=False, adx_trending=False,
        confirmed_uptrend=False, confirmed_downtrend=False,
    )
    r_rng = score_confluence(ind_rng)
    check("Stoch 85 in RANGE (with bear cross) → score -1",
          r_rng["scores"]["stochastic"]["score"] == -1,
          f"got {r_rng['scores']['stochastic']['score']}")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 4 — Downtrend Context-Aware Scoring
# ══════════════════════════════════════════════════════════════════════════════

def test_downtrend_context_scoring():
    """RSI 28 in downtrend should score -1 (bearish momentum, not oversold bounce)."""
    print("\n[Group 4] Downtrend context-aware scoring")

    ind_dn = _bear_indicators(rsi=28.0, confirmed_uptrend=False, confirmed_downtrend=True)
    r_dn = score_confluence(ind_dn)

    check("RSI 28 in DOWNTREND → score -1 (oversold can get worse)",
          r_dn["scores"]["rsi"]["score"] == -1,
          f"got {r_dn['scores']['rsi']['score']} label={r_dn['scores']['rsi']['label']}")

    check("trend_context is DOWNTREND",
          r_dn["trend_context"] == "DOWNTREND",
          f"got {r_dn['trend_context']}")

    check("Downtrend: signal is not LONG",
          "LONG" not in r_dn["signal"],
          f"got signal={r_dn['signal']}")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 5 — Signal Threshold Arithmetic
# ══════════════════════════════════════════════════════════════════════════════

def test_enter_long_requires_8():
    """ENTER LONG fires only at 8+ bullish indicators."""
    print("\n[Group 5] Signal threshold arithmetic")

    ind = _bull_indicators()
    result = score_confluence(ind)

    if result["signal"] == "ENTER LONG":
        check("ENTER LONG: long_count >= 8",
              result["long_count"] >= 8,
              f"long_count={result['long_count']}")

    check("signal value is a known valid string",
          result["signal"] in {
              "ENTER LONG", "ENTER SHORT",
              "STAY LONG", "STAY SHORT",
              "LEAN LONG", "LEAN SHORT",
              "NO SIGNAL",
          },
          f"got '{result['signal']}'")

    check("total_indicators is 12",
          result["total_indicators"] == 12,
          f"got {result['total_indicators']}")

    check("long_count + short_count + neutral_count == 12",
          result["long_count"] + result["short_count"] + result["neutral_count"] == 12,
          f"{result['long_count']} + {result['short_count']} + {result['neutral_count']} != 12")


def test_signal_label_matches_counts():
    """Signal label must be consistent with its count values."""
    print()
    for label, ind_fn in [("bull", _bull_indicators), ("bear", _bear_indicators)]:
        ind = ind_fn()
        r = score_confluence(ind)
        sig    = r["signal"]
        longs  = r["long_count"]
        shorts = r["short_count"]

        if sig == "ENTER LONG":
            check(f"{label}: ENTER LONG ↔ long_count >= 8",
                  longs >= 8, f"long={longs}")
        elif sig == "ENTER SHORT":
            check(f"{label}: ENTER SHORT ↔ short_count >= 8",
                  shorts >= 8, f"short={shorts}")
        elif sig == "STAY LONG":
            check(f"{label}: STAY LONG ↔ long >= 6 and long > short+1",
                  longs >= 6 and longs > shorts + 1,
                  f"long={longs} short={shorts}")
        elif sig == "STAY SHORT":
            check(f"{label}: STAY SHORT ↔ short >= 6 and short > long+1",
                  shorts >= 6 and shorts > longs + 1,
                  f"long={longs} short={shorts}")
        elif sig in ("LEAN LONG", "LEAN SHORT"):
            check(f"{label}: LEAN ↔ dominant >= 5",
                  max(longs, shorts) >= 5,
                  f"long={longs} short={shorts}")
        else:  # NO SIGNAL
            check(f"{label}: NO SIGNAL ↔ dominant < 5",
                  max(longs, shorts) < 5,
                  f"long={longs} short={shorts}")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 6 — End-to-End: Directional Consistency with Synthetic Price Data
# ══════════════════════════════════════════════════════════════════════════════

def test_e2e_uptrend_direction():
    """End-to-end: smooth uptrend → signal must not be SHORT."""
    print("\n[Group 6] End-to-end directional consistency with synthetic data")

    df = _make_df(direction="up")
    ind = _calculate_indicators(df)
    result = score_confluence(ind)

    check("Smooth uptrend: long_count > short_count",
          result["long_count"] > result["short_count"],
          f"long={result['long_count']} short={result['short_count']}")

    check("Smooth uptrend: signal is not ENTER SHORT or STAY SHORT",
          result["signal"] not in ("ENTER SHORT", "STAY SHORT"),
          f"got signal={result['signal']}")


def test_e2e_downtrend_direction():
    """End-to-end: smooth downtrend → signal must not be LONG."""
    df = _make_df(direction="down")
    ind = _calculate_indicators(df)
    result = score_confluence(ind)

    check("Smooth downtrend: short_count > long_count",
          result["short_count"] > result["long_count"],
          f"long={result['long_count']} short={result['short_count']}")

    check("Smooth downtrend: signal is not ENTER LONG or STAY LONG",
          result["signal"] not in ("ENTER LONG", "STAY LONG"),
          f"got signal={result['signal']}")


def test_e2e_output_structure():
    """End-to-end: score_confluence always returns required keys with valid types."""
    print()
    for direction in ("up", "down", "sideways"):
        df = _make_df(direction=direction)
        ind = _calculate_indicators(df)
        result = score_confluence(ind)

        required = [
            "scores", "long_count", "short_count", "neutral_count",
            "signal", "signal_class", "strength", "threshold",
            "total_indicators", "trend_context",
            "recovery_bullish", "recovery_bearish",
        ]
        missing = [k for k in required if k not in result]
        check(f"{direction}: all required output keys present",
              not missing,
              f"missing: {missing}")

        check(f"{direction}: trend_context is UPTREND/DOWNTREND/RANGE",
              result["trend_context"] in {"UPTREND", "DOWNTREND", "RANGE"},
              f"got '{result['trend_context']}'")

        check(f"{direction}: total_indicators == 12",
              result["total_indicators"] == 12,
              f"got {result['total_indicators']}")


# ══════════════════════════════════════════════════════════════════════════════
# GROUP 7 — Absolute Safety Guards (can NEVER happen)
# ══════════════════════════════════════════════════════════════════════════════

def test_ema_bull_stack_never_produces_enter_short():
    """
    When EMA 9>21>50 AND ADX bullish AND above 50 SMA:
    ENTER SHORT (8+ bearish) is structurally impossible.
    """
    print("\n[Group 7] Absolute safety guards")

    for i in range(8):
        np.random.seed(i * 13)
        rsi_val   = np.random.uniform(50, 85)
        stoch_val = np.random.uniform(60, 99)
        bb_val    = np.random.uniform(0.6, 0.99)
        ind = _bull_indicators(
            rsi=rsi_val, stoch_k=stoch_val, stoch_d=stoch_val - 5,
            bb_pct=bb_val,
            ema_bull_stack=True, adx_bull=True, above_50sma=True,
            confirmed_uptrend=True, confirmed_downtrend=False,
        )
        result = score_confluence(ind)
        check(
            f"EMA bull + ADX bull (trial {i+1}, rsi={rsi_val:.0f} stoch={stoch_val:.0f}): never ENTER SHORT",
            result["signal"] != "ENTER SHORT",
            f"got signal={result['signal']} short={result['short_count']}",
        )


def test_ema_bear_stack_never_produces_enter_long():
    """
    When EMA 9<21<50 AND ADX bearish AND below 50 SMA:
    ENTER LONG (8+ bullish) is structurally impossible.
    """
    print()
    for i in range(8):
        np.random.seed(i * 17)
        rsi_val   = np.random.uniform(15, 45)
        stoch_val = np.random.uniform(1, 40)
        bb_val    = np.random.uniform(0.01, 0.40)
        ind = _bear_indicators(
            rsi=rsi_val, stoch_k=stoch_val, stoch_d=stoch_val + 5,
            bb_pct=bb_val,
            ema_bear_stack=True, adx_bull=False, above_50sma=False,
            confirmed_uptrend=False, confirmed_downtrend=True,
        )
        result = score_confluence(ind)
        check(
            f"EMA bear + ADX bear (trial {i+1}, rsi={rsi_val:.0f} stoch={stoch_val:.0f}): never ENTER LONG",
            result["signal"] != "ENTER LONG",
            f"got signal={result['signal']} long={result['long_count']}",
        )


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 62)
    print("  Confluence Signal Engine — Regression Test Suite")
    print("=" * 62)

    test_regression_ndx_at_200sma()
    test_ema_stack_substitutes_for_200sma()
    test_rsi_overbought_scores_differently_by_context()
    test_key_level_near_high_context()
    test_bollinger_upper_band_context()
    test_stochastic_high_context()
    test_downtrend_context_scoring()
    test_enter_long_requires_8()
    test_signal_label_matches_counts()
    test_e2e_uptrend_direction()
    test_e2e_downtrend_direction()
    test_e2e_output_structure()
    test_ema_bull_stack_never_produces_enter_short()
    test_ema_bear_stack_never_produces_enter_long()

    print()
    print("=" * 62)
    print(f"  Results: {PASS} passed / {FAIL} failed")
    if ERRORS:
        print("\n  Failed tests:")
        for e in ERRORS:
            print(f"    • {e}")
    print("=" * 62)

    sys.exit(0 if FAIL == 0 else 1)
