"""Prompt rendering for polygon equation diagram tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_json_example import build_keyed_point_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import DOMAIN, SCENE_ID


def polygon_equation_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_task_key: str,
    annotation_keys: Sequence[str],
    target_name: str,
    variable_name: str,
    shape_name: str,
    answer: int,
    instance_seed: int,
) -> tuple[dict[str, Any], PromptTraceArtifacts]:
    """Render v1 prompt variants for one polygon equation task."""

    defaults = required_group_defaults(
        prompt_defaults,
        ("bundle_id", "scene_key", "task_key"),
        context="prompt defaults for polygon_equation_diagram",
    )
    json_example, json_example_answer_only = build_keyed_point_prompt_json_examples(
        annotation_keys=tuple(str(key) for key in annotation_keys),
        answer=int(answer),
    )
    annotation_key_list = ", ".join(f'"{key}"' for key in annotation_keys)
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(prompt_task_key),
        query_key=str(prompt_task_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "annotation_hint": (
                "set \"annotation\" to a JSON object with exactly these visible vertex-label keys: "
                f"{annotation_key_list}; each value must be that labeled vertex's pixel point [x,y]"
            ),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
            "shape_name": str(shape_name),
            "target_name": str(target_name),
            "variable_name": str(variable_name),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["polygon_equation_prompt_artifacts"]
