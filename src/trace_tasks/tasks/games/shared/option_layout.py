"""Shared option-card layout helpers for games renderers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class OptionGridSpec:
    """Resolved grid shape for a small visual answer-option set."""

    option_count: int
    columns: int
    rows: int


def balanced_option_grid_spec(option_count: int) -> OptionGridSpec:
    """Return a balanced grid for visual option panels.

    The common games convention is four options as 2 x 2 and six options
    as 3 x 2. Five-option grids use the same three-column width with the
    final row centered by ``option_grid_position``.
    """

    count = int(option_count)
    if count <= 0:
        raise ValueError("option_count must be positive")
    if count <= 3:
        columns = count
    elif count == 4:
        columns = 2
    elif count <= 6:
        columns = 3
    elif count <= 8:
        columns = 4
    else:
        columns = int(math.ceil(math.sqrt(float(count))))
    rows = int(math.ceil(float(count) / float(columns)))
    return OptionGridSpec(option_count=count, columns=columns, rows=rows)


def option_grid_size(
    option_count: int,
    *,
    item_width: float,
    item_height: float,
    gap_x: float,
    gap_y: float,
    columns: int | None = None,
) -> Tuple[float, float]:
    """Return the full grid width/height for option items."""

    spec = balanced_option_grid_spec(option_count)
    resolved_columns = int(columns if columns is not None else spec.columns)
    if resolved_columns <= 0:
        raise ValueError("columns must be positive")
    rows = int(math.ceil(float(int(option_count)) / float(resolved_columns)))
    width = (float(resolved_columns) * float(item_width)) + (float(resolved_columns - 1) * float(gap_x))
    height = (float(rows) * float(item_height)) + (float(rows - 1) * float(gap_y))
    return float(width), float(height)


def option_grid_position(
    index: int,
    option_count: int,
    *,
    left: float,
    top: float,
    item_width: float,
    item_height: float,
    gap_x: float,
    gap_y: float,
    columns: int | None = None,
) -> Tuple[int, int, float, float]:
    """Return row, column, and top-left point for one option item.

    Short final rows are centered inside the full grid width so five-option
    layouts do not read as a lopsided three-plus-two panel.
    """

    count = int(option_count)
    item_index = int(index)
    if item_index < 0 or item_index >= count:
        raise ValueError("option index is outside option_count")
    spec = balanced_option_grid_spec(count)
    resolved_columns = int(columns if columns is not None else spec.columns)
    if resolved_columns <= 0:
        raise ValueError("columns must be positive")

    row = int(item_index // resolved_columns)
    col = int(item_index % resolved_columns)
    row_start = int(row * resolved_columns)
    row_count = int(min(resolved_columns, count - row_start))
    full_row_width = (float(resolved_columns) * float(item_width)) + (float(resolved_columns - 1) * float(gap_x))
    actual_row_width = (float(row_count) * float(item_width)) + (float(row_count - 1) * float(gap_x))
    row_left = float(left) + ((full_row_width - actual_row_width) / 2.0)
    item_left = row_left + (float(col) * (float(item_width) + float(gap_x)))
    item_top = float(top) + (float(row) * (float(item_height) + float(gap_y)))
    return row, col, float(item_left), float(item_top)

