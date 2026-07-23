"""Scene-level defaults and config access for Snake tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import split_scene_generation_rendering_prompt_defaults

from .state import PLANNED_MOVE_OUTCOMES, SCENE_ID


@dataclass(frozen=True)
class SnakeDefaults:
    """Stable fallback defaults for visible Snake scenes."""

    board_size_support: Tuple[int, ...] = (7, 8, 9, 10)
    body_length_support: Tuple[int, ...] = (5, 6, 7, 8, 9, 10, 11)
    snake_length_count_support: Tuple[int, ...] = (6, 7, 8, 9, 10, 11, 12)
    safe_direction_count_support: Tuple[int, ...] = (0, 1, 2, 3)
    planned_move_count_support: Tuple[int, ...] = (3, 4, 5)
    obstacle_count_support: Tuple[int, ...] = (2, 3, 4, 5, 6)
    planned_move_outcome_support: Tuple[str, ...] = PLANNED_MOVE_OUTCOMES
    canvas_width: int = 900
    canvas_height: int = 900
    panel_margin_px: int = 54
    max_board_size_px: int = 720
    board_border_width_px: int = 6
    grid_line_width_px: int = 2
    cell_padding_px: int = 8
    food_radius_px: int = 28
    eye_radius_px: int = 4


DEFAULTS = SnakeDefaults()
SCENE_DEFAULTS = get_scene_defaults("games", SCENE_ID)
GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = split_scene_generation_rendering_prompt_defaults(
    SCENE_DEFAULTS if isinstance(SCENE_DEFAULTS, dict) else {},
    task_id="games_snake_scene",
)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)
PROMPT_WIRING_KEYS = (
    "bundle_id",
    "scene_key",
    "task_key",
    "json_output_contract",
    "json_output_contract_answer_only",
    "object_description_square_grid",
    "snake_rule_text",
    "planned_move_wall_annotation_rule_text",
    "answer_hint_snake_length_count",
    "annotation_hint_snake_length_count",
    "answer_hint_safe_direction_count",
    "annotation_hint_safe_direction_count",
    "answer_hint_path_result_option_label",
    "annotation_hint_path_result_option_label",
)


__all__ = [
    "DEFAULTS",
    "GEN_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_DEFAULTS",
    "PROMPT_WIRING_KEYS",
    "RENDER_DEFAULTS",
    "SCENE_DEFAULTS",
    "SnakeDefaults",
]
