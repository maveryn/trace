"""Prompt assembly for circle-centerline-overlap tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_json_example import dump_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import SCENE_ID


def circle_centerline_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_query_key: str,
    target_name: str,
    label_mode: str,
    circle_count: int,
    answer_value: int,
    instance_seed: int,
) -> tuple[dict[str, Any], Any]:
    """Render prompt variants for a task-owned circle-centerline query."""

    defaults = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
        ),
        context="prompt defaults for circle_centerline_overlap",
    )
    annotation_example = [[120, 220], [260, 220]]
    json_example, json_example_answer_only = dump_prompt_json_examples(
        annotation=annotation_example,
        answer=int(answer_value),
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
            "circle_count_description": "two" if int(circle_count) == 2 else "three",
            "measure_label_kind": "diameter" if str(label_mode) == "diameter" else "radius",
            "target_name": str(target_name),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["circle_centerline_prompt_artifacts"]
