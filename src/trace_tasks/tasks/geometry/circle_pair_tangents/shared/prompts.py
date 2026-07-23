"""Prompt assembly for circle-pair tangent tasks."""

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


def pair_tangent_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_query_key: str,
    answer_value: int,
    annotation_keys: Sequence[str],
    instance_seed: int,
) -> tuple[dict[str, Any], Any]:
    """Render prompt variants for one task-owned tangent query."""

    defaults = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
        ),
        context="prompt defaults for circle_pair_tangents",
    )
    annotation_key_text = ", ".join(f'"{key}"' for key in annotation_keys)
    annotation_example = {
        str(key): [220 + index * 70, 300 - (index % 2) * 95]
        for index, key in enumerate(annotation_keys)
    }
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
            "annotation_keys": str(annotation_key_text),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["pair_tangent_prompt_artifacts"]
