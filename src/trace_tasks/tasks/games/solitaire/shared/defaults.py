"""Scene defaults for solitaire tableau game tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.shared.config_defaults import group_default, split_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults

from .state import SCENE_ID


@dataclass(frozen=True)
class SolitaireDefaults:
    """Stable fallback defaults for solitaire tableau scenes."""

    canvas_width: int = 1060
    canvas_height: int = 820
    card_width_px: int = 74
    card_height_px: int = 104
    card_gap_px: int = 16
    column_gap_px: int = 18
    column_step_y_px: int = 32
    panel_margin_px: int = 44
    foundation_gap_px: int = 14
    card_corner_radius_px: int = 9
    rank_font_size_px: int = 18
    card_center_font_size_px: int = 31
    badge_font_size_px: int = 14
    label_font_size_px: int = 15
    option_font_size_px: int = 22
    option_height_px: int = 46
    option_gap_px: int = 10
    move_option_count_support: Tuple[int, ...] = (4,)
    card_option_count_support: Tuple[int, ...] = (4, 6)
    cascade_depth_support: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    foundation_ready_target_answer_support: Tuple[int, ...] = (0, 1, 2, 3, 4)
    column_card_count_target_answer_support: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    tableau_movable_card_count_target_answer_support: Tuple[int, ...] = (0, 1, 2, 3, 4)
    tableau_column_count_support: Tuple[int, ...] = (7, 8)


DEFAULTS = SolitaireDefaults()
_SCENE_DEFAULTS = get_scene_defaults("games", SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    _SCENE_DEFAULTS if isinstance(_SCENE_DEFAULTS, Mapping) else {},
)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def int_default(params: Mapping[str, Any], key: str, fallback: int) -> int:
    """Resolve an integer rendering default from params, scene config, or fallback."""

    if str(key) in params:
        return int(params[str(key)])
    return int(group_default(RENDER_DEFAULTS, str(key), int(fallback)))
