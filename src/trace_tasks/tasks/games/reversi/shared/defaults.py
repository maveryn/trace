"""Fallback defaults for Reversi scene-package tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ReversiSceneDefaults:
    """Stable fallback defaults when scene config omits optional knobs."""

    legal_move_count_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    flip_count_support: Tuple[int, ...] = (2, 3, 4, 5, 6)
    frontier_disc_count_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5)
    canvas_width: int = 900
    canvas_height: int = 900
    panel_margin_px: int = 48
    player_badge_height_px: int = 52
    player_badge_width_px: int = 190
    header_gap_px: int = 18
    max_board_size_px: int = 720
    board_corner_radius_px: int = 24
    board_frame_width_px: int = 14
    cell_line_width_px: int = 3
    marked_square_outline_width_px: int = 6
    disc_inset_fraction: float = 0.14
    player_badge_font_size_px: int = 22


DEFAULTS = ReversiSceneDefaults()


__all__ = ["DEFAULTS", "ReversiSceneDefaults"]
