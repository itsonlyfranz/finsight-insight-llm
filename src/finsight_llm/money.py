"""Port of FinSight's `Money` helper: pesos are always written with the peso sign and US grouping."""
from __future__ import annotations

SYMBOL = "₱"  # peso sign


def _with_symbol(number: str) -> str:
    return f"-{SYMBOL}{number[1:]}" if number.startswith("-") else f"{SYMBOL}{number}"


def fmt(amount: float) -> str:
    """1234.5 -> ₱1,234.50"""
    return _with_symbol(f"{amount:,.2f}")


def fmt_whole(amount: float) -> str:
    """1234.5 -> ₱1,234 (no decimals); mirrors the app's compact projections."""
    return _with_symbol(f"{amount:,.0f}")
