"""Shared centered multi-row option layout helpers for puzzle scenes."""

from __future__ import annotations

import math
from typing import List, Tuple


def centered_option_grid_shape(option_count: int) -> Tuple[int, int]:
    """Return `(cols, rows)` for the canonical centered option layout."""

    if int(option_count) <= 4:
        cols = int(option_count)
    elif int(option_count) <= 6:
        cols = 3
    else:
        cols = 4
    rows = int(math.ceil(float(option_count) / float(cols)))
    return int(cols), int(rows)


def centered_option_row_counts(option_count: int, cols: int) -> Tuple[int, ...]:
    """Return the number of rendered options in each centered row."""

    remaining = int(option_count)
    counts: List[int] = []
    while remaining > 0:
        row_count = int(min(int(cols), int(remaining)))
        counts.append(int(row_count))
        remaining -= int(row_count)
    return tuple(counts)


__all__ = [
    "centered_option_grid_shape",
    "centered_option_row_counts",
]
