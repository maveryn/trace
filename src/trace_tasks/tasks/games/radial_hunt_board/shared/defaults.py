"""Fallback defaults for radial hunt board scene tasks.

YAML remains the source of truth. These constants are last-resort fallbacks
used by config resolvers when optional keys are missing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RadialHuntBoardDefaults:
    """Stable fallback defaults for radial hunt board generation and rendering."""

    target_answer_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    min_total_piece_count: int = 5
    max_total_piece_count: int = 14
    canvas_width: int = 780
    canvas_height: int = 760
    panel_margin_px: int = 52
    max_board_size_px: int = 560
    edge_width_px: int = 5
    point_radius_px: int = 13
    piece_radius_px: int = 23
    marker_width_px: int = 5
    dynamic_canvas_size_enabled: bool = True
    canvas_min_width_px: int = 560
    canvas_min_height_px: int = 540
    canvas_side_padding_px: int = 150
    canvas_vertical_padding_px: int = 150


DEFAULTS = RadialHuntBoardDefaults()


__all__ = ["DEFAULTS", "RadialHuntBoardDefaults"]
