"""Formula helpers for thermal-mixing tasks."""

from __future__ import annotations

from typing import Sequence


def integer_average(values: Sequence[int]) -> int:
    """Return an integer average, rejecting non-integral constructions."""

    if not values:
        raise ValueError("cannot average an empty temperature set")
    total = int(sum(int(value) for value in values))
    count = int(len(values))
    if total % count != 0:
        raise ValueError("temperature average is not an integer")
    return int(total // count)


__all__ = ["integer_average"]
