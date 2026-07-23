"""Fallback defaults for irregular-link-board scene tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class IrregularLinkBoardDefaults:
    """Stable code fallbacks when scene config omits optional knobs."""

    target_answer_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    board_size_support: Tuple[int, ...] = (4, 5, 6)
    capture_board_size_support: Tuple[int, ...] = (5, 6)
    min_total_piece_count: int = 4
    max_total_piece_count: int = 12
    canvas_width: int = 780
    canvas_height: int = 760
    panel_margin_px: int = 52
    max_board_size_px: int = 560
    edge_width_px: int = 5
    point_radius_px: int = 14
    piece_radius_px: int = 23
    marker_width_px: int = 5
    dynamic_canvas_size_enabled: bool = True
    canvas_min_width_px: int = 560
    canvas_min_height_px: int = 540
    canvas_side_padding_px: int = 150
    canvas_vertical_padding_px: int = 150


DEFAULTS = IrregularLinkBoardDefaults()


__all__ = ["DEFAULTS", "IrregularLinkBoardDefaults"]
