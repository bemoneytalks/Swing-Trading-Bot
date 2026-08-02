# Changelog

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
