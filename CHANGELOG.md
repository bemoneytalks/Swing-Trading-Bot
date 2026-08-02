# Changelog

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
