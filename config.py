"""Configuration for SPX Predictive Trading Bot."""

TICKER = "^GSPC"  # S&P 500 Index
DATA_PERIOD_YEARS = 5
TRADING_WINDOW_HOURS = 2  # First 2 hours from open (9:30 - 11:30 ET)

# Feature engineering
LOOKBACK_PERIODS = [5, 10, 20, 50, 100, 200]
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
ATR_PERIOD = 14
STOCH_PERIOD = 14
WILLIAMS_PERIOD = 14
CCI_PERIOD = 20
ADX_PERIOD = 14

# Model — Daily (next-day candle direction)
MODEL_PATH = "model/spx_model.pkl"
SCALER_PATH = "model/spx_scaler.pkl"
FEATURE_PATH = "model/spx_features.pkl"

# Model — 5-Day Trend (will price be higher in 5 trading days?)
TREND_MODEL_PATH = "model/spx_trend_model.pkl"
TREND_SCALER_PATH = "model/spx_trend_scaler.pkl"
TREND_FEATURE_PATH = "model/spx_trend_features.pkl"
TREND_HORIZON = 5  # trading days forward
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_ESTIMATORS = 500
MIN_SAMPLES_SPLIT = 10
MIN_SAMPLES_LEAF = 5
MAX_DEPTH = 12

# Prediction thresholds
STRONG_BULL_THRESHOLD = 0.65
BULL_THRESHOLD = 0.55
BEAR_THRESHOLD = 0.45
STRONG_BEAR_THRESHOLD = 0.35

# High-confidence filter — only surface predictions in the high-conviction zone
# Bull prob > HIGH_CONF = high-confidence bullish call
# Bull prob < LOW_CONF  = high-confidence bearish call
# Predictions in the 35–65% zone are uncertain; filtered separately in backtest
HIGH_CONF_THRESHOLD = 0.65
LOW_CONF_THRESHOLD  = 0.35

# Data cache
CACHE_DIR = "cache"
CACHE_EXPIRY_HOURS = 4

# NDX (Nasdaq-100) config
NDX_TICKER = "^NDX"
NDX_OPTIONS_TICKER = "QQQ"   # QQQ options (more liquid than ^NDX options)
NDX_MODEL_PATH = "model/ndx_model.pkl"
NDX_SCALER_PATH = "model/ndx_scaler.pkl"
NDX_FEATURE_PATH = "model/ndx_features.pkl"
NDX_TREND_MODEL_PATH = "model/ndx_trend_model.pkl"
NDX_TREND_SCALER_PATH = "model/ndx_trend_scaler.pkl"
NDX_TREND_FEATURE_PATH = "model/ndx_trend_features.pkl"

# ─── Options Grader (v1.3.0) ─────────────────────────────────────────
# Preset rules for the A–F contract grader. All thresholds tunable here.
GRADER_DTE_MIN = 30            # swing horizon: min days to expiration
GRADER_DTE_MAX = 90            # swing horizon: max days to expiration
GRADER_MIN_OI = 100            # eligibility: minimum open interest
GRADER_MAX_SPREAD_PCT = 0.12   # eligibility: max bid-ask spread as % of mid
GRADER_MIN_MID = 0.10          # eligibility: minimum mid price ($)
GRADER_DELTA_MIN = 0.15        # eligibility: min |delta|
GRADER_DELTA_MAX = 0.90        # eligibility: max |delta|
GRADER_EARNINGS_EXCLUDE_DAYS = 7   # exclude all grades if earnings this close

# Quality score weights (sum to 1.0)
GRADER_W_LIQUIDITY = 0.40
GRADER_W_VALUE = 0.25
GRADER_W_PROBABILITY = 0.20
GRADER_W_RISK = 0.15

# Recommendation = quality * this + tier fit * (1 - this)
GRADER_W_QUALITY_IN_REC = 0.65

# Tier definitions: (delta band lo, hi, DTE pref lo, hi)
GRADER_TIERS = {
    "conservative": {"delta": (0.65, 0.85), "dte": (60, 90),
                     "tagline": "Lower relative risk"},
    "balanced":     {"delta": (0.45, 0.60), "dte": (45, 75),
                     "tagline": "Balanced risk/reward"},
    "aggressive":   {"delta": (0.25, 0.40), "dte": (30, 60),
                     "tagline": "Higher relative risk"},
}

# Letter grade boundaries
GRADER_LETTERS = [(90, "A"), (75, "B"), (60, "C"), (45, "D"), (0, "F")]
