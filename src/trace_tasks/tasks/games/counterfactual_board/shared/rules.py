"""Board counting rules for counterfactual-board game tasks."""

from __future__ import annotations

from .state import (
    COLUMN_AXIS,
    HORIZONTAL_LINE_AXIS,
    ROW_AXIS,
    STYLE_SPECS,
    VERTICAL_LINE_AXIS,
)


def target_answer_for_axis(*, counted_axis: str, rows: int, cols: int) -> int:
    """Return the integer count requested by one resolved count axis."""

    axis = str(counted_axis)
    if axis in {ROW_AXIS, HORIZONTAL_LINE_AXIS}:
        return int(rows)
    if axis in {COLUMN_AXIS, VERTICAL_LINE_AXIS}:
        return int(cols)
    raise ValueError(f"unsupported counted axis: {counted_axis!r}")


def canonical_answer_for_style(*, counted_axis: str, style: str) -> int:
    """Return the familiar board-size answer for trace-only bias metadata."""

    spec = STYLE_SPECS[str(style)]
    return target_answer_for_axis(
        counted_axis=str(counted_axis),
        rows=int(spec.canonical_rows),
        cols=int(spec.canonical_cols),
    )


def element_kind_for_axis(counted_axis: str) -> str:
    """Return the visual entity kind produced for one resolved axis."""

    axis = str(counted_axis)
    if axis == ROW_AXIS:
        return "board_row"
    if axis == COLUMN_AXIS:
        return "board_column"
    if axis == HORIZONTAL_LINE_AXIS:
        return "horizontal_grid_line"
    if axis == VERTICAL_LINE_AXIS:
        return "vertical_grid_line"
    raise ValueError(f"unsupported counted axis: {counted_axis!r}")


__all__ = [
    "canonical_answer_for_style",
    "element_kind_for_axis",
    "target_answer_for_axis",
]
