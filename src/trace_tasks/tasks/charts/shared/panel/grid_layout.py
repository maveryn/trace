"""Shared panel-grid layout helpers for chart renderers."""

from __future__ import annotations

from typing import Sequence, Tuple

BBox = Tuple[float, float, float, float]

_EXACT_ROW_LENGTHS: dict[int, Tuple[int, ...]] = {
    1: (1,),
    2: (2,),
    3: (3,),
    4: (2, 2),
    5: (3, 2),
    6: (3, 3),
    7: (4, 3),
    8: (3, 3, 2),
    9: (3, 3, 3),
}


def panel_row_lengths(panel_count: int, *, fallback_max_columns: int = 5) -> Tuple[int, ...]:
    """Return stable row lengths for a multi-panel chart layout."""

    count = int(panel_count)
    if count <= 0:
        raise ValueError("panel_count must be positive")
    if count in _EXACT_ROW_LENGTHS:
        return _EXACT_ROW_LENGTHS[count]
    max_columns = max(1, int(fallback_max_columns))
    rows: list[int] = []
    remaining = count
    while remaining > 0:
        row_len = min(max_columns, remaining)
        rows.append(int(row_len))
        remaining -= int(row_len)
    return tuple(rows)


def layout_panel_grid(
    container_bbox: Sequence[float],
    *,
    panel_count: int,
    gap_x: float,
    gap_y: float | None = None,
    row_lengths: Sequence[int] | None = None,
) -> Tuple[BBox, ...]:
    """Place panel boxes in a centered row grid inside ``container_bbox``.

    Rows use the shared Trace chart row policy by default. When a row is not
    full, it is centered relative to the widest row.
    """

    x1, y1, x2, y2 = (float(value) for value in container_bbox[:4])
    count = int(panel_count)
    rows = tuple(int(value) for value in (row_lengths or panel_row_lengths(count)))
    if sum(rows) != count:
        raise ValueError("row_lengths must sum to panel_count")
    if not rows:
        raise ValueError("row_lengths must not be empty")
    max_columns = max(rows)
    horizontal_gap = float(gap_x)
    vertical_gap = float(horizontal_gap if gap_y is None else gap_y)
    cell_width = (float(x2 - x1) - (float(max_columns - 1) * horizontal_gap)) / float(max_columns)
    cell_height = (float(y2 - y1) - (float(len(rows) - 1) * vertical_gap)) / float(len(rows))
    if cell_width <= 0 or cell_height <= 0:
        raise ValueError("container is too small for panel grid")

    boxes: list[BBox] = []
    index = 0
    for row_index, row_len in enumerate(rows):
        row_width = (float(row_len) * cell_width) + (float(row_len - 1) * horizontal_gap)
        start_x = float(x1) + ((float(x2 - x1) - row_width) / 2.0)
        top = float(y1) + (float(row_index) * (cell_height + vertical_gap))
        for col_index in range(int(row_len)):
            left = start_x + (float(col_index) * (cell_width + horizontal_gap))
            boxes.append((left, top, left + cell_width, top + cell_height))
            index += 1
            if index >= count:
                break
    return tuple(boxes)


def layout_panel_grid_list(
    container_bbox: Sequence[float],
    *,
    panel_count: int,
    gap_x: float,
    gap_y: float | None = None,
    row_lengths: Sequence[int] | None = None,
) -> list[BBox]:
    """Return ``layout_panel_grid`` as a list for mutable scene render loops."""

    return list(
        layout_panel_grid(
            container_bbox,
            panel_count=int(panel_count),
            gap_x=float(gap_x),
            gap_y=gap_y,
            row_lengths=row_lengths,
        )
    )


def layout_panel_grid_int(
    container_bbox: Sequence[float],
    *,
    panel_count: int,
    gap_x: float,
    gap_y: float | None = None,
    row_lengths: Sequence[int] | None = None,
) -> Tuple[Tuple[int, int, int, int], ...]:
    """Return panel boxes rounded to integer pixel coordinates."""

    return tuple(
        (
            int(round(float(box[0]))),
            int(round(float(box[1]))),
            int(round(float(box[2]))),
            int(round(float(box[3]))),
        )
        for box in layout_panel_grid(
            container_bbox,
            panel_count=int(panel_count),
            gap_x=float(gap_x),
            gap_y=gap_y,
            row_lengths=row_lengths,
        )
    )


__all__ = [
    "layout_panel_grid",
    "layout_panel_grid_int",
    "layout_panel_grid_list",
    "panel_row_lengths",
]
