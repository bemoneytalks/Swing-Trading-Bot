"""
Position Sizing Calculator
Risk-based position sizing used by /api/risk-calc and the options grader.

(The former portfolio position tracker was removed in v1.0.0 — see CHANGELOG.)
"""


def calculate_position_size(account_size, entry_price, stop_price, risk_pct=2.0):
    """Calculate optimal position size based on risk percentage."""
    risk_amount = account_size * (risk_pct / 100)
    risk_per_share = abs(entry_price - stop_price)

    if risk_per_share <= 0:
        return {"error": "Stop price must differ from entry price"}

    shares = int(risk_amount / risk_per_share)
    position_value = shares * entry_price

    return {
        "shares": shares,
        "position_value": round(position_value, 2),
        "risk_amount": round(risk_amount, 2),
        "risk_per_share": round(risk_per_share, 2),
        "max_loss": round(shares * risk_per_share, 2),
        "position_pct": round(position_value / account_size * 100, 1),
    }
