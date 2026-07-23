"""Fallback defaults for Rhythm scene-package tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class RhythmSceneDefaults:
    """Stable fallback defaults when scene config omits optional knobs."""

    lane_count_support: Tuple[int, ...] = (5, 6, 7, 8)
    row_count_support: Tuple[int, ...] = (10, 11, 12, 13, 14)
    beat_window_support: Tuple[int, ...] = (5, 6, 7)
    note_count_support: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    score_total_support: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    canvas_width: int = 900
    canvas_height: int = 900
    panel_margin_px: int = 42
    grid_width_px: int = 650
    grid_height_px: int = 790
    grid_border_width_px: int = 5
    row_gap_px: int = 4
    lane_gap_px: int = 9
    note_radius_px: int = 13
    label_font_size_px: int = 25


DEFAULTS = RhythmSceneDefaults()


__all__ = ["DEFAULTS", "RhythmSceneDefaults"]
