"""Panel layout helpers for electromagnetic induction diagrams."""

from __future__ import annotations

from collections.abc import Sequence


def bbox(values: Sequence[float]) -> list[float]:
    """Return a JSON-stable pixel bounding box."""

    return [round(float(value), 3) for value in values]


def panel_layout(*, canvas_width: int, canvas_height: int) -> list[list[float]]:
    """Place six mini-panels in a fixed two-by-three induction grid."""

    margin_x = 54.0
    top = 68.0
    gap_x = 26.0
    gap_y = 28.0
    panel_w = (float(canvas_width) - (2.0 * margin_x) - (2.0 * gap_x)) / 3.0
    panel_h = (float(canvas_height) - top - 52.0 - gap_y) / 2.0
    bboxes: list[list[float]] = []
    for row in range(2):
        for col in range(3):
            left = margin_x + (float(col) * (panel_w + gap_x))
            y = top + (float(row) * (panel_h + gap_y))
            bboxes.append(bbox((left, y, left + panel_w, y + panel_h)))
    return bboxes
