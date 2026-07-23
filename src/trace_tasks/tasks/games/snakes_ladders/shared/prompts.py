"""Prompt assembly helpers for Snakes and Ladders tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_scene_prompt_variants

from .defaults import PROMPT_DEFAULTS
from .state import DOMAIN, SCENE_ID, SnakesLaddersAxes


PROMPT_REQUIRED_KEYS = (
    "bundle_id",
    "scene_key",
    "task_key",
    "json_output_contract",
    "json_output_contract_answer_only",
    "object_description_standard_board",
    "movement_rule_text",
    "overshoot_rule_text",
    "answer_hint_move_outcome_value",
    "annotation_hint_move_outcome_value",
    "answer_hint_ladder_count",
    "annotation_hint_ladder_count",
    "answer_hint_snake_count",
    "annotation_hint_snake_count",
    "answer_hint_remaining_to_finish_value",
    "annotation_hint_remaining_to_finish_value",
)


def build_snakes_ladders_prompt_artifacts(
    *,
    axes: SnakesLaddersAxes,
    prompt_query_key: str,
    die_value: int | None,
    horizon_roll_count: int | None,
    json_example: str,
    json_example_answer_only: str,
    instance_seed: int,
):
    """Render prompt variants for one task-owned prompt query key."""

    prompt_defaults = required_group_defaults(
        PROMPT_DEFAULTS,
        PROMPT_REQUIRED_KEYS,
        context=f"prompt defaults for games/{SCENE_ID}/{prompt_query_key}",
    )
    selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(prompt_defaults[f"object_description_{str(axes.scene_variant)}"]),
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "movement_rule_text": str(prompt_defaults["movement_rule_text"]),
            "overshoot_rule_text": str(prompt_defaults["overshoot_rule_text"]),
            "die_value": 0 if die_value is None else int(die_value),
            "answer_hint": str(prompt_defaults[f"answer_hint_{str(prompt_query_key)}"]),
            "annotation_hint": str(prompt_defaults[f"annotation_hint_{str(prompt_query_key)}"]),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(selection)


__all__ = ["PROMPT_REQUIRED_KEYS", "build_snakes_ladders_prompt_artifacts"]
