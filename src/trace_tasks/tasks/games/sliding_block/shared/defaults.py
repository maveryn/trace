"""Scene-default access for sliding-block games tasks."""

from __future__ import annotations

from trace_tasks.tasks.games.shared.visual_defaults import load_games_scene_noise_defaults
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from .state import DOMAIN, SCENE_ID


GEN_DEFAULTS, RENDER_DEFAULTS, PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    DOMAIN,
    SCENE_ID,
)
POST_IMAGE_NOISE_DEFAULTS = load_games_scene_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)

PROMPT_WIRING_KEYS = (
    "bundle_id",
    "scene_key",
    "task_key",
    "object_description",
    "json_output_contract",
    "json_output_contract_answer_only",
    "answer_hint_integer",
    "answer_hint_option_letter",
    "annotation_hint_blocker_count",
    "annotation_hint_movable_block_count",
    "annotation_hint_block_orientation_count",
    "annotation_hint_move_result_label",
    "json_example_blocker_count",
    "json_example_movable_block_count",
    "json_example_block_orientation_count",
    "json_example_move_result_label",
    "json_example_answer_only_blocker_count",
    "json_example_answer_only_movable_block_count",
    "json_example_answer_only_block_orientation_count",
    "json_example_answer_only_move_result_label",
    "object_description_blocker_count",
    "object_description_movable_block_count",
    "object_description_block_orientation_count",
    "object_description_move_result_label",
)


__all__ = [
    "GEN_DEFAULTS",
    "POST_IMAGE_NOISE_DEFAULTS",
    "PROMPT_DEFAULTS",
    "PROMPT_WIRING_KEYS",
    "RENDER_DEFAULTS",
]
