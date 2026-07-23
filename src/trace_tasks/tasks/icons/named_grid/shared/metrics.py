"""Scene-local grid metric helpers for named-grid icon tasks."""

from __future__ import annotations

from collections import Counter
from typing import Dict, Sequence, Tuple


GridCell = Tuple[int, int]


def line_cells(*, axis: str, line_index: int, rows: int, cols: int) -> Tuple[GridCell, ...]:
    """Return the cell coordinates in one row or column."""

    if str(axis) == "row":
        return tuple((int(line_index), col) for col in range(int(cols)))
    if str(axis) == "column":
        return tuple((row, int(line_index)) for row in range(int(rows)))
    raise ValueError(f"unsupported named-grid axis: {axis}")


def grid_cells(*, rows: int, cols: int) -> Tuple[GridCell, ...]:
    """Return all cell coordinates in row-major order."""

    return tuple((row, col) for row in range(int(rows)) for col in range(int(cols)))


def axis_line_count(*, axis: str, rows: int, cols: int) -> int:
    """Return how many rows or columns exist for an axis."""

    if str(axis) == "row":
        return int(rows)
    if str(axis) == "column":
        return int(cols)
    raise ValueError(f"unsupported named-grid axis: {axis}")


def axis_line_capacity(*, axis: str, rows: int, cols: int) -> int:
    """Return how many cells are in each line for an axis."""

    if str(axis) == "row":
        return int(cols)
    if str(axis) == "column":
        return int(rows)
    raise ValueError(f"unsupported named-grid axis: {axis}")


def row_target_counts(shape_ids_by_cell: Sequence[Sequence[str]], *, target_shape_id: str) -> Tuple[int, ...]:
    """Count target-shape cells in each grid row."""

    return tuple(
        sum(1 for shape_id in row if str(shape_id) == str(target_shape_id))
        for row in shape_ids_by_cell
    )


def column_target_counts(shape_ids_by_cell: Sequence[Sequence[str]], *, target_shape_id: str) -> Tuple[int, ...]:
    """Count target-shape cells in each grid column."""

    if not shape_ids_by_cell:
        return tuple()
    row_count = len(shape_ids_by_cell)
    col_count = len(shape_ids_by_cell[0])
    return tuple(
        sum(1 for row in range(int(row_count)) if str(shape_ids_by_cell[int(row)][int(col)]) == str(target_shape_id))
        for col in range(int(col_count))
    )


def active_line_counts(*, axis: str, row_counts: Sequence[int], column_counts: Sequence[int]) -> Tuple[int, ...]:
    """Return the row or column count vector selected by an axis."""

    if str(axis) == "row":
        return tuple(int(value) for value in row_counts)
    if str(axis) == "column":
        return tuple(int(value) for value in column_counts)
    raise ValueError(f"unsupported named-grid axis: {axis}")


def line_shape_counts(shape_ids_by_cell: Sequence[Sequence[str]], *, axis: str, line_index: int) -> Dict[str, int]:
    """Count every shape id in one grid line."""

    rows = len(shape_ids_by_cell)
    cols = len(shape_ids_by_cell[0]) if shape_ids_by_cell else 0
    counts = Counter(
        str(shape_ids_by_cell[int(row)][int(col)])
        for row, col in line_cells(axis=str(axis), line_index=int(line_index), rows=int(rows), cols=int(cols))
    )
    return {str(key): int(value) for key, value in counts.items()}


def line_condition_matches(count: int, *, condition: str, threshold: int) -> bool:
    """Evaluate a named-grid line-count predicate."""

    if str(condition) == "at_least":
        return int(count) >= int(threshold)
    if str(condition) == "exactly":
        return int(count) == int(threshold)
    if str(condition) == "none":
        return int(count) == 0
    raise ValueError(f"unsupported named-grid line condition: {condition}")


def qualifying_line_indices(counts: Sequence[int], *, condition: str, threshold: int) -> Tuple[int, ...]:
    """Return indices of lines satisfying a line-count predicate."""

    return tuple(
        int(index)
        for index, count in enumerate(counts)
        if line_condition_matches(int(count), condition=str(condition), threshold=int(threshold))
    )


def unique_extreme_index(counts: Sequence[int], *, extremum: str) -> int:
    """Return the unique index of the requested extreme count."""

    values = tuple(int(value) for value in counts)
    if not values:
        raise ValueError("cannot find an extreme in an empty count vector")
    if str(extremum) == "most":
        target = max(values)
    elif str(extremum) == "fewest":
        target = min(values)
    else:
        raise ValueError(f"unsupported named-grid extremum: {extremum}")
    matching = tuple(index for index, value in enumerate(values) if int(value) == int(target))
    if len(matching) != 1:
        raise ValueError("requested extreme is not unique")
    return int(matching[0])


__all__ = [
    "GridCell",
    "active_line_counts",
    "axis_line_capacity",
    "axis_line_count",
    "column_target_counts",
    "grid_cells",
    "line_cells",
    "line_condition_matches",
    "line_shape_counts",
    "qualifying_line_indices",
    "row_target_counts",
    "unique_extreme_index",
]
