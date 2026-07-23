"""Fallback defaults for Sixteen Soldiers scene tasks.

YAML remains the source of truth. These constants are last-resort fallbacks
used by config resolvers when optional keys are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SixteenSoldiersDefaults:
    """Stable fallback defaults for shared scene generation and rendering."""

    piece_count_per_side_support: Tuple[int, ...] = (6, 7, 8, 9, 10)
    canvas_width: int = 760
    canvas_height: int = 900
    panel_margin_px: int = 52
    max_board_width_px: int = 500
    max_board_height_px: int = 680
    edge_width_px: int = 5
    point_radius_px: int = 8
    piece_radius_px: int = 21
    marker_width_px: int = 5
    dynamic_canvas_size_enabled: bool = True
    canvas_min_width_px: int = 560
    canvas_min_height_px: int = 760
    canvas_side_padding_px: int = 168
    canvas_vertical_padding_px: int = 136


DEFAULTS = SixteenSoldiersDefaults()


__all__ = ["DEFAULTS", "SixteenSoldiersDefaults"]
