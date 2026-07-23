"""Prompt assembly for tangent-packing tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.prompt_json_example import dump_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import DOMAIN, PROMPT_BUNDLE_ID, SCENE_ID, SCENE_PROMPT_KEY


def _example_answer_value(answer_value: Any, answer_type: str) -> Any:
    if str(answer_type) == "integer":
        return int(round(float(answer_value)))
    return round(float(answer_value), 1)


def _bbox_prompt_json_examples(answer: Any) -> tuple[str, str]:
    return dump_prompt_json_examples(annotation=[88, 96, 430, 360], answer=answer)


def tangent_packing_object_description(construction_kind: str) -> str:
    """Return prompt-facing scene wording for the selected construction."""

    descriptions = {
        "circle_in_square": "a circle tangent inside a square with visible measurements and one marked target value",
        "square_in_circle": "a square tangent inside a circle with visible measurements and one marked target value",
        "two_circles_in_rectangle": "two equal circles tangent inside a rectangle with visible measurements and one marked target value",
    }
    return descriptions.get(
        str(construction_kind),
        "a tangent-packing diagram with visible measurements and one marked target value",
    )


def build_tangent_packing_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    task_prompt_key: str,
    prompt_query_key: str,
    annotation_roles: Sequence[str],
    answer_value: float | int,
    answer_type: str,
    object_description: str,
    instance_seed: int,
):
    """Render v1 prompt variants for one tangent-packing public objective."""

    _ = tuple(str(role) for role in annotation_roles)
    json_example, json_example_answer_only = _bbox_prompt_json_examples(
        _example_answer_value(answer_value, answer_type)
    )
    annotation_instruction = (
        "set \"annotation\" to the pixel bounding box [x0,y0,x1,y1] around the marked target geometric "
        "region or shape, excluding numeric labels and measurement text"
    )
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key=str(prompt_defaults.get("scene_key", SCENE_PROMPT_KEY)),
        task_key=str(task_prompt_key),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(object_description),
            "annotation_instruction": str(annotation_instruction),
            "annotation_key_list": "diagram",
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_tangent_packing_prompt_artifacts", "tangent_packing_object_description"]
