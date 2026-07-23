"""Shared cell-layout helpers for icon grid and row scenes."""

from __future__ import annotations

from typing import List, Tuple


BBox = Tuple[int, int, int, int]


def centered_square_bbox(bbox: BBox) -> BBox:
    """Return the largest centered square fully contained in `bbox`."""

    x0, y0, x1, y1 = [int(value) for value in bbox]
    width = max(1, int(x1 - x0))
    height = max(1, int(y1 - y0))
    side = int(max(1, min(width, height)))
    dx = int((width - side) // 2)
    dy = int((height - side) // 2)
    return (
        int(x0 + dx),
        int(y0 + dy),
        int(x0 + dx + side),
        int(y0 + dy + side),
    )


def resolve_compact_grid_shape(cell_count: int) -> Tuple[int, int]:
    """Return a compact `(rows, cols)` grid for a requested labeled cell count."""

    n = int(cell_count)
    if n <= 4:
        return 2, 2
    if n <= 6:
        return 2, 3
    if n <= 8:
        return 2, 4
    return 3, 4


def resolve_grid_cell_slots(content_bbox: BBox, *, cell_count: int, cell_padding_px: int) -> List[BBox]:
    """Return row-major grid slots inside one content rectangle."""

    rows, cols = resolve_compact_grid_shape(int(cell_count))
    return resolve_fixed_grid_cell_slots(
        content_bbox,
        rows=int(rows),
        cols=int(cols),
        cell_padding_px=int(cell_padding_px),
    )[: int(cell_count)]


def resolve_fixed_grid_cell_slots(
    content_bbox: BBox,
    *,
    rows: int,
    cols: int,
    cell_padding_px: int,
) -> List[BBox]:
    """Return row-major grid slots for one explicit `(rows, cols)` layout."""

    rows_i = max(1, int(rows))
    cols_i = max(1, int(cols))
    x0, y0, x1, y1 = content_bbox
    width = max(1, int(x1 - x0))
    height = max(1, int(y1 - y0))
    cell_w = width / float(cols_i)
    cell_h = height / float(rows_i)
    pad = max(0, int(cell_padding_px))
    slots: List[BBox] = []
    for row in range(rows_i):
        for col in range(cols_i):
            slot_x0 = int(round(float(x0) + (float(col) * cell_w))) + pad
            slot_y0 = int(round(float(y0) + (float(row) * cell_h))) + pad
            slot_x1 = int(round(float(x0) + (float(col + 1) * cell_w))) - pad
            slot_y1 = int(round(float(y0) + (float(row + 1) * cell_h))) - pad
            slots.append((slot_x0, slot_y0, slot_x1, slot_y1))
    return slots


def resolve_horizontal_row_slots(
    content_bbox: BBox,
    *,
    cell_count: int,
    cell_padding_px: int,
    target_aspect_ratio: float | None = None,
) -> List[BBox]:
    """Return one horizontal row of evenly sized cell slots.

    When `target_aspect_ratio` is provided, the helper keeps the slots in one
    horizontal row while vertically centering a shared cell height that is as
    close as possible to the requested width/height ratio.
    """

    count = max(1, int(cell_count))
    x0, y0, x1, y1 = content_bbox
    width = max(1, int(x1 - x0))
    height = max(1, int(y1 - y0))
    cell_w = width / float(count)
    pad = max(0, int(cell_padding_px))
    x_slots: List[tuple[int, int]] = []
    min_inner_width = None
    for index in range(count):
        slot_x0 = int(round(float(x0) + (float(index) * cell_w))) + pad
        slot_x1 = int(round(float(x0) + (float(index + 1) * cell_w))) - pad
        x_slots.append((slot_x0, slot_x1))
        inner_width = max(1, int(slot_x1 - slot_x0))
        min_inner_width = inner_width if min_inner_width is None else min(min_inner_width, inner_width)

    slot_y0 = int(y0) + pad
    slot_y1 = int(y1) - pad
    if target_aspect_ratio is not None:
        ratio = max(0.1, float(target_aspect_ratio))
        max_inner_height = max(1, int(height - (2 * pad)))
        target_inner_height = max(1, int(round(float(min_inner_width or 1) / ratio)))
        resolved_inner_height = min(max_inner_height, target_inner_height)
        top_slack = max(0, int(max_inner_height - resolved_inner_height))
        slot_y0 = int(y0) + pad + int(top_slack // 2)
        slot_y1 = int(slot_y0 + resolved_inner_height)

    slots: List[BBox] = []
    for slot_x0, slot_x1 in x_slots:
        slots.append((slot_x0, slot_y0, slot_x1, slot_y1))
    return slots


__all__ = [
    "centered_square_bbox",
    "resolve_compact_grid_shape",
    "resolve_fixed_grid_cell_slots",
    "resolve_grid_cell_slots",
    "resolve_horizontal_row_slots",
]
