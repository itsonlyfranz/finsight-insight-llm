"""Port of FinSight's `Money` helper: pesos are always written with the peso sign and US grouping."""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal

SYMBOL = "₱"  # peso sign


def _with_symbol(number: str) -> str:
    return f"-{SYMBOL}{number[1:]}" if number.startswith("-") else f"{SYMBOL}{number}"


def _round_half_up(amount: float, places: int) -> str:
    """Java/Kotlin `String.format("%,.Nf")` rounding: HALF_UP on the shortest decimal form of the double."""
    q = Decimal(1).scaleb(-places)
    return f"{Decimal(repr(float(amount))).quantize(q, rounding=ROUND_HALF_UP):,}"


def fmt(amount: float) -> str:
    """1234.5 -> ₱1,234.50"""
    return _with_symbol(f"{amount:,.2f}")


def fmt_whole(amount: float) -> str:
    """1234.5 -> ₱1,235 (no decimals, ties round up like the app); mirrors the app's compact projections."""
    return _with_symbol(_round_half_up(amount, 0))
