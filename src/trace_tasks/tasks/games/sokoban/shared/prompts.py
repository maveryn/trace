"""Prompt assembly helpers for Sokoban tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import PROMPT_DEFAULTS
from .state import DOMAIN, SCENE_ID


PROMPT_REQUIRED_KEYS = (
    "bundle_id",
    "scene_key",
    "task_key",
    "object_description_box_goal_status_count",
    "object_description_closest_box_goal_label",
    "object_description_push_stand_cell_label",
    "json_output_contract",
    "json_output_contract_answer_only",
    "answer_hint_box_goal_count",
    "answer_hint_option_letter",
    "answer_hint_option_letter_stand_cell",
    "annotation_hint_counted_box_bbox_set",
    "annotation_hint_selected_box_bbox",
    "annotation_hint_selected_stand_cell_bbox",
    "json_example_counted_box_bbox_set",
    "json_example_selected_box_bbox",
    "json_example_selected_stand_cell_bbox",
    "json_example_answer_only_integer",
    "json_example_answer_only_option_label",
)


def build_sokoban_prompt_artifacts(
    *,
    prompt_query_key: str,
    object_description_key: str,
    annotation_hint_key: str,
    json_example_key: str,
    answer_hint_key: str = "answer_hint_box_goal_count",
    json_example_answer_only_key: str = "json_example_answer_only_integer",
    dynamic_values: Mapping[str, Any],
    instance_seed: int,
):
    """Render prompt variants for a task-owned Sokoban prompt query key."""

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
            "object_description": str(prompt_defaults[str(object_description_key)]),
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "annotation_hint": str(prompt_defaults[str(annotation_hint_key)]),
            "answer_hint": str(prompt_defaults[str(answer_hint_key)]),
            "json_example": str(prompt_defaults[str(json_example_key)]),
            "json_example_answer_only": str(prompt_defaults[str(json_example_answer_only_key)]),
            **dict(dynamic_values),
        },
        instance_seed=int(instance_seed),
    )
    return prompt_defaults, build_prompt_trace_artifacts(selection)


__all__ = ["PROMPT_REQUIRED_KEYS", "build_sokoban_prompt_artifacts"]
