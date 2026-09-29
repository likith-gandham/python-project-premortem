"""Human-friendly number formatting for reports."""

from __future__ import annotations

import math


def money(x: float) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    sign = "-" if x < 0 else ""
    a = abs(x)
    if a >= 1e6:
        return f"{sign}${a / 1e6:.1f}M"
    if a >= 1e4:
        return f"{sign}${a / 1e3:.0f}K"
    return f"{sign}${a:,.0f}"


def pct(x: float, digits: int = 0) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x * 100:.{digits}f}%"


def value(x: float, unit: str) -> str:
    """Format a value according to the unit of its assumption."""
    if unit == "$":
        return money(x)
    if unit == "%":
        return pct(x, 1)
    if unit == "bool":
        return "closes" if x >= 0.5 else "fails"
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    return f"{x:,.0f}"
