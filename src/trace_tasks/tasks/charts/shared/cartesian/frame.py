"""Plot-frame helpers for cartesian chart renderers."""

from __future__ import annotations

from .geometry import round_bbox


def plot_bbox_from_margins(
    *,
    canvas_width: float,
    canvas_height: float,
    margin_left_px: float,
    margin_right_px: float,
    margin_top_px: float,
    margin_bottom_px: float,
) -> list[float]:
    """Return a plot bbox from canvas size and margins."""

    return round_bbox(
        [
            float(margin_left_px),
            float(margin_top_px),
            float(canvas_width) - float(margin_right_px),
            float(canvas_height) - float(margin_bottom_px),
        ]
    )


__all__ = ["plot_bbox_from_margins"]
