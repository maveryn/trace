"""Layout helpers for symbolic abacus rendering."""

from __future__ import annotations

from typing import Sequence

from .state import AbacusOptionPanelRenderParams, AbacusReadoutRenderParams


def rounded_bbox(values: Sequence[float]) -> list[float]:
    """Round a bbox or point for stable trace serialization."""

    return [round(float(value), 3) for value in values]


def bead_bbox(cx: float, cy: float, *, width: int, height: int) -> list[float]:
    """Return the bbox for a bead centered at the given point."""

    return rounded_bbox(
        (
            float(cx - (0.5 * int(width))),
            float(cy - (0.5 * int(height))),
            float(cx + (0.5 * int(width))),
            float(cy + (0.5 * int(height))),
        )
    )


def bbox_center(bbox: Sequence[float]) -> list[float]:
    """Return the center point of a bbox."""

    x0, y0, x1, y1 = (float(value) for value in bbox)
    return rounded_bbox(((x0 + x1) * 0.5, (y0 + y1) * 0.5))


def option_card_bboxes(
    *,
    option_labels: Sequence[str],
    params: AbacusOptionPanelRenderParams,
) -> dict[str, list[float]]:
    """Return fixed 3x2 option-card boxes for visible option labels."""

    labels = tuple(str(label) for label in option_labels)
    if len(labels) != 6:
        raise ValueError("abacus option panel requires exactly six option labels")
    card_w = float(params.option_card_width_px)
    card_h = float(params.option_card_height_px)
    gap_x = float(params.option_card_gap_x_px)
    gap_y = float(params.option_card_gap_y_px)
    total_w = (3.0 * card_w) + (2.0 * gap_x)
    total_h = (2.0 * card_h) + gap_y
    start_x = float(round(0.5 * (float(params.canvas_width) - total_w)))
    start_y = float(round(0.5 * (float(params.canvas_height) - total_h)))
    bboxes: dict[str, list[float]] = {}
    for index, label in enumerate(labels):
        row = int(index // 3)
        col = int(index % 3)
        x0 = float(start_x + (col * (card_w + gap_x)))
        y0 = float(start_y + (row * (card_h + gap_y)))
        bboxes[str(label)] = rounded_bbox((x0, y0, x0 + card_w, y0 + card_h))
    return bboxes


def readout_option_bboxes(
    *,
    option_labels: Sequence[str],
    params: AbacusReadoutRenderParams,
    panel_bbox: Sequence[float],
) -> dict[str, list[float]]:
    """Return one horizontal row of six text option cards below the abacus panel."""

    labels = tuple(str(label) for label in option_labels)
    if len(labels) != 6:
        raise ValueError("abacus readout option row requires exactly six option labels")
    card_w = float(params.readout_option_card_width_px)
    card_h = float(params.readout_option_card_height_px)
    gap = float(params.readout_option_card_gap_px)
    total_w = (6.0 * card_w) + (5.0 * gap)
    start_x = float(round(0.5 * (float(params.canvas_width) - total_w)))
    y0 = float(panel_bbox[3]) + float(params.readout_option_card_margin_top_px)
    max_y0 = float(params.canvas_height) - card_h - 18.0
    y0 = min(float(y0), float(max_y0))
    bboxes: dict[str, list[float]] = {}
    for index, label in enumerate(labels):
        x0 = float(start_x + (int(index) * (card_w + gap)))
        bboxes[str(label)] = rounded_bbox((x0, y0, x0 + card_w, y0 + card_h))
    return bboxes


__all__ = [
    "bbox_center",
    "bead_bbox",
    "option_card_bboxes",
    "readout_option_bboxes",
    "rounded_bbox",
]
