# Changelog

## v2.0.4 — Honest NDX Grader Labeling (2026-08-03)

### Fixed
- Grading **NDX** showed QQQ's spot price ($694) under an "NDX" headline,
  and quoted dealer walls in NDX scale (28,781) next to QQQ-scale strikes.
  The grader has always deliberately recommended **QQQ contracts** for NDX
  (deeper liquidity than NDX options); it now says so: the banner reads
  **QQQ** with a proxy note ("Direction is scored on the NDX index;
  recommended contracts are QQQ options…"), and wall levels are quoted in
  QQQ scale to match the recommended strikes. SPX and custom tickers
  unchanged.

## v2.0.3 — Leading Indicator Source Labels (2026-08-02)

### Changed
- All six Leading Indicators cards now declare their data source in the
  title: News Sentiment, Dealer Positioning, Multi-Timeframe, and GEX
  Regime show the analyzed ticker (e.g. "(MU)"); Crude Oil Correlation is
  labeled "(macro)"; **Net Premium Flow is labeled "(SPX market flow)"** —
  making explicit that it is index-level flow until per-ticker streak
  history exists.

## v2.0.2 — Dynamic Confluence Tab + Tab Reorder (2026-08-02)

### Changed
- **The first tab now shows what it's analyzing.** It reads "SPX" by
  default and becomes the active ticker (e.g. "NVDA ✕") when you Analyze a
  custom symbol; clicking ✕ (or the header SPX button) returns to the SPX
  default. This makes the any-ticker Confluence feature — present since
  v1.1.0 — actually discoverable.
- **Tab order reorganized**: SPX / NDX / Options / ML Signal / GEX /
  Risk Calc / Scanner / Patterns / Backtest.
- Tab switching now uses data-tab attributes instead of a fragile
  index-based map (future tab changes can no longer break highlighting).

## v2.0.1 — NDX Dashboard Gets the Exposure Ladder (2026-08-02)

### Fixed
- The NDX parallel dashboard's **GEX (QQQ)** sub-tab was still rendering the
  old net-GEX bar chart. It now shows the same GEX/VEX/DEX dealer exposure
  ladder as the main GEX tab (spot row, CW/PW/DM walls, vanna bias),
  scaled from QQQ options to NDX levels. The ladder renderer is now shared
  (`renderGexDashboard`) so both tabs stay in sync going forward.

## v2.0.0 — The Any-Ticker Update (2026-08-02)

### Added
- **Per-ticker ML Signal + Backtest** — the final piece. Enter any ticker
  and the ML Signal tab trains a dedicated Random Forest + Gradient
  Boosting ensemble for it on first use (~1–3 minutes, then cached under
  model/<symbol>_*.pkl with its own daily-data cache), producing next-day
  and 5-day trend probabilities. The Backtest tab replays that model's
  last 30 days. SPX/NDX models and behavior are unchanged.
- **Earnings guard for ML signals**: single-stock predictions display the
  next earnings date, and signals within 5 days of earnings are flagged
  "SIGNAL UNRELIABLE THROUGH EARNINGS GAP". Shared earnings.py module now
  powers both the ML guard and the options grader's IV-crush guard.
- **Symbol banners** on the ML Signal and Backtest tabs, and the live
  price line is labeled with the analyzed ticker.
- **Housekeeping**: per-symbol models and caches untouched for 30+ days
  are cleaned up at startup (SPX/NDX artifacts are never touched).

### The Any-Ticker Update (v1.1 → v2.0) in one line
Type any ticker: Confluence says whether and which direction → the
GEX/VEX/DEX ladder says where the dealer walls are → the Grader says which
contract → the Contract Analyzer says your specific odds → the ML Signal
adds model probabilities → the Exit engine says when to get out.

### Notes
- Single-stock ML models are noisier than index models (single-name
  idiosyncratic risk, earnings gaps); treat probabilities as one input,
  never a sole basis. Index models have longer validation history.

## v1.3.2 — Options Dashboard Follows the Graded Ticker (2026-08-02)

### Changed
- The **0–30 DTE Options Dashboard** (IV rank, skew, technicals, suggested
  strikes, pivots, ATM greeks, strategy suggestions) is no longer
  SPX/NDX-only: grade any ticker and the dashboard reloads for that symbol
  ("NVDA Options Dashboard"), with strike increments inferred from price
  scale. `/api/options` accepts `?symbol=`.
- The Contract Analyzer's SPX/QQQ/NDX dropdown flow is unchanged.

## v1.3.1 — Grader Consolidation + Layout Cleanup (2026-08-02)

### Changed
- **Grader moved into the Options tab** (standalone Grader tab removed):
  the A–F contract grader now sits directly above the Contract Analyzer,
  with its **own ticker input** and "Grade Contracts" button — no more
  dependence on the global Analyze flow (which navigates to Confluence).
- **Prominent symbol headline** on grader results: a large SYMBOL · $spot ·
  direction banner makes it unmistakable which ticker was graded.
- **Scaled Entry Checklist moved to the Risk Calc tab**, grouping the
  pre-trade sizing/confirmation tools together.

## v1.3.0 — Options Contract Grader (2026-08-02)

### Added
- **Grader tab**: A–F contract grading for any optionable ticker. Direction
  comes from the Trend Confluence system; every contract in the 30–90 DTE
  swing window is scored against preset rules, and the best candidate per
  risk tier (Conservative 0.65–0.85Δ / Balanced 0.45–0.60Δ / Aggressive
  0.25–0.40Δ) is surfaced with Recommendation, Quality, and Tier-fit scores,
  delta, probability of profit, max risk, breakeven, IV, liquidity, theta,
  and a "why this contract" explanation.
- **Preset rules** (all tunable in config.py `GRADER_*`): OI ≥ 100, spread
  ≤ 12% of mid, |delta| 0.15–0.90; quality = 40% liquidity + 25% value
  (IV vs 20-day HV) + 20% probability + 15% risk efficiency (theta burn +
  breakeven distance vs ATR); recommendation = 65% quality + 35% tier fit;
  A ≥ 90, B ≥ 75, C ≥ 60, D ≥ 45, else F.
- **Earnings guard**: recommendations are suppressed within 7 days of
  earnings (IV crush protection), and any contract whose expiry spans an
  earnings date is tagged EARNINGS BEFORE EXPIRY.
- **"Our Take" narrative** including dealer gamma regime and wall levels
  from the v1.2 exposure engine.
- `/api/options/grade?symbol=` endpoint; `test_grader.py` (27 unit tests,
  synthetic contracts, no network).

### Notes
- Quotes are delayed and stale off-hours (mids labeled "estimated"); OI
  updates once daily. Educational estimates — verify with your broker.

## v1.2.0 — GEX/VEX/DEX Dealer Exposure Ladder (2026-08-02)

### Added
- **Dealer Exposure Ladder**: the GEX tab is now a synced per-strike ladder
  showing GEX (gamma), VEX (vanna), and DEX (delta) dealer exposure side by
  side, with a highlighted spot row, call wall / put wall / delta magnet
  badges, per-column net totals, and a thin-chain guard. Works for **any
  optionable ticker** via the global ticker input, plus SPX/NDX as before.
- **New levels**: Call Wall, Put Wall, Delta Magnet, and a Vanna
  Supportive/Fragile bias badge alongside the existing flip level and
  gamma support/resistance.
- **`chain_service.py`**: shared per-symbol options-chain fetch with a
  30-minute cache — one fetch powers the ladder, dealer positioning, and the
  upcoming v1.3 contract grader.
- **Per-ticker GEX Regime + Dealer Positioning**: the confluence
  `gex_regime` indicator and the Leading Indicators panel's GEX Regime and
  Dealer Positioning cards now use the analyzed ticker's own chain
  (put/call volume & OI ratios included). SPX/NDX behavior unchanged.
- `/api/gex` accepts `?symbol=` and returns per-strike `chart_net_vex` /
  `chart_net_dex` plus wall levels.

### Notes
- Net Premium Flow remains SPX/NDX: its streak/flip logic needs day-by-day
  history, which begins accumulating per ticker in a future release.
- Sign conventions follow the standard retail dealer model (dealers short
  customer options); OI updates once daily via Yahoo Finance.

## v1.1.1 — Per-Ticker Leading Indicators (2026-08-02)

### Fixed
- The **Leading Indicators — Signal Confidence** panel now follows the analyzed
  ticker for its two symbol-capable checks: **News Sentiment** and
  **Multi-Timeframe Heikin-Ashi** (both cards are labeled with the symbol they
  analyzed). Previously all six checks ran on SPX regardless of the input.

### Notes
- Crude Oil Correlation is market-level by design. Dealer Positioning,
  Net Premium Flow, and GEX Regime remain SPX market context until per-symbol
  options-chain data lands in v1.2.0.

## v1.1.0 — Any-Ticker Confluence (2026-08-02)

### Added
- **Any-ticker analysis**: a ticker input in the dashboard tab bar — enter any
  symbol (e.g. `NVDA`) and run the full 12-indicator Trend Confluence system,
  the 6-indicator Reversal Entry panel, confidence grading, and live price
  strip against it. `SPX`/`NDX` still work exactly as before.
- `/api/confluence` and `/api/confidence` accept an optional `?symbol=` query
  parameter (`/api/exit` already supported one). Omitting it preserves the
  existing index-based behavior byte-for-byte.
- Ticker bar is now labeled with whichever symbol's data it is displaying.

### Notes
- For non-index symbols, the GEX regime and net-premium checks use SPX
  market-wide dealer positioning as regime context (per-symbol dealer exposure
  arrives with the GEX/VEX/DEX ladder in v1.2.0).

## v1.0.0 — Clean Baseline (2026-08-02)

### Changed
- **Portfolio position tracker removed.** The Portfolio tab is now a lean
  **Risk Calc** tab containing the position-size calculator. The four
  `/api/portfolio*` routes were removed; `portfolio.py` now provides only
  `calculate_position_size()` (used by `/api/risk-calc`).
- **Privacy**: personal portfolio data and local editor settings removed from
  the repository and purged from git history (history was rewritten —
  re-clone if you have an old checkout). `portfolio.example.json` template
  added.

### Added
- `test_regression.py` — golden-master regression suite locking the shape of
  every API endpoint. All updates must pass it before merging.
