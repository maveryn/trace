"""Formula helpers for equivalent circuit diagrams."""

from __future__ import annotations

from fractions import Fraction
from typing import Sequence


def parallel_resistance(values: Sequence[int]) -> Fraction:
    """Return exact equivalent resistance for positive parallel resistors."""

    reciprocal_sum = sum(Fraction(1, int(value)) for value in values)
    if reciprocal_sum <= 0:
        raise ValueError("parallel resistance requires positive values")
    return Fraction(1, 1) / reciprocal_sum


def series_capacitance(values: Sequence[int]) -> Fraction:
    """Return exact equivalent capacitance for positive series capacitors."""

    reciprocal_sum = sum(Fraction(1, int(value)) for value in values)
    if reciprocal_sum <= 0:
        raise ValueError("series capacitance requires positive values")
    return Fraction(1, 1) / reciprocal_sum


__all__ = ["parallel_resistance", "series_capacitance"]
