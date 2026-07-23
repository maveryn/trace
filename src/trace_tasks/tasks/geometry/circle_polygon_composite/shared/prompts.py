"""Prompt assembly for circle-polygon-composite tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_json_example import dump_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import SCENE_ID


def tangential_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_query_key: str,
    target_side: str,
    visible_sides: str,
    answer_value: int,
    annotation_keys: Sequence[str],
    instance_seed: int,
) -> tuple[dict[str, Any], Any]:
    """Render prompt variants for a task-owned tangential-quadrilateral query."""

    defaults = required_group_defaults(
        prompt_defaults,
        ("bundle_id", "scene_key", "task_key"),
        context="prompt defaults for circle_polygon_composite tangential task",
    )
    annotation_example = {
        str(key): [180 + index * 34, 220 + (index % 3) * 38]
        for index, key in enumerate(annotation_keys)
    }
    json_example, json_example_answer_only = dump_prompt_json_examples(
        annotation=annotation_example,
        answer=int(answer_value),
        ensure_ascii=False,
    )
    prompt_selection = render_scene_prompt_variants(
        domain="geometry",
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "target_side": str(target_side),
            "visible_sides": str(visible_sides),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


def tangent_angle_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_query_key: str,
    angle_object_description: str,
    round_shape: str,
    answer_value: int,
    annotation_keys: Sequence[str],
    instance_seed: int,
) -> tuple[dict[str, Any], Any]:
    """Render prompt variants for a task-owned tangent-angle query."""

    defaults = required_group_defaults(
        prompt_defaults,
        ("bundle_id", "scene_key", "task_key"),
        context="prompt defaults for circle_polygon_composite tangent-angle task",
    )
    annotation_example = {
        str(key): [170 + index * 42, 260 + (index % 2) * 46]
        for index, key in enumerate(annotation_keys)
    }
    json_example, json_example_answer_only = dump_prompt_json_examples(
        annotation=annotation_example,
        answer=int(answer_value),
        ensure_ascii=False,
    )
    prompt_selection = render_scene_prompt_variants(
        domain="geometry",
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "angle_object_description": str(angle_object_description),
            "round_shape": str(round_shape),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["tangential_prompt_artifacts", "tangent_angle_prompt_artifacts"]
