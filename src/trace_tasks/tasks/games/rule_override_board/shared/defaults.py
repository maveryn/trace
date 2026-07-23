"""Fallback defaults for rule-override board scene-package tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults

from .state import SCENE_ID


@dataclass(frozen=True)
class RuleOverrideBoardDefaults:
    """Stable fallback defaults when scene config omits optional knobs."""

    board_count_support: Tuple[int, ...] = (4, 5, 6)
    target_answer_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    line_board_size_support: Tuple[int, ...] = (3, 4)
    piece_board_size_support: Tuple[int, ...] = (4, 5)
    min_canvas_width_px: int = 520
    min_canvas_height_px: int = 420
    cell_size_px: int = 56
    board_gap_px: int = 22
    board_padding_px: int = 14
    board_label_height_px: int = 38
    content_margin_px: int = 56
    panel_radius_px: int = 18
    panel_border_width_px: int = 2
    board_border_width_px: int = 3
    grid_width_px: int = 3
    board_label_font_size_px: int = 20
    mark_font_size_px: int = 34


DEFAULTS = RuleOverrideBoardDefaults()
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


__all__ = ["DEFAULTS", "POST_IMAGE_NOISE_DEFAULTS", "RuleOverrideBoardDefaults"]
