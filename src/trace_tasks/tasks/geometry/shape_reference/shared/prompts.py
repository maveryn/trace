"""Prompt helpers for the shape-reference scene package."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_json_example import build_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .relations import SCENE_ID


def relation_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_branch_key: str,
    annotation_value: Sequence[Sequence[float]],
    vertex_count: int,
    instance_seed: int,
) -> tuple[Mapping[str, Any], PromptTraceArtifacts]:
    """Render the relation-match prompt after the public task selects the relation."""

    defaults = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "object_description",
            "json_output_contract",
            "json_output_contract_answer_only",
            "annotation_hint_template",
            "answer_hint",
            "json_example",
            "json_example_answer_only",
        ),
        context="shape-reference relation prompt defaults",
    )
    json_example, json_example_answer_only = build_prompt_json_examples(
        annotation_value=[list(point) for point in annotation_value],
        answer_type="option_letter",
    )
    annotation_hint = str(defaults["annotation_hint_template"]).format(vertex_count=int(vertex_count))
    selection = render_scene_prompt_variants(
        domain="geometry",
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(defaults["task_key"]),
        query_key=str(prompt_branch_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        slots={
            "object_description": str(defaults["object_description"]),
            "json_output_contract": str(defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(defaults["json_output_contract_answer_only"]),
            "annotation_hint": str(annotation_hint),
            "answer_hint": str(defaults["answer_hint"]),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return defaults, build_prompt_trace_artifacts(selection)


def transformation_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_branch_key: str,
    annotation_value: Sequence[Sequence[float]],
    vertex_count: int,
    rotation_instruction: str | None,
    instance_seed: int,
) -> tuple[Mapping[str, Any], PromptTraceArtifacts]:
    """Render the transform-task prompt after the public task selects the rule."""

    defaults = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "object_description",
            "json_output_contract",
            "json_output_contract_answer_only",
            "annotation_hint_template",
            "answer_hint",
        ),
        context="shape-reference transformation prompt defaults",
    )
    json_example, json_example_answer_only = build_prompt_json_examples(
        annotation_value=[list(point) for point in annotation_value],
        answer_type="option_letter",
    )
    annotation_hint = str(defaults["annotation_hint_template"]).format(vertex_count=int(vertex_count))
    slots = {
        "object_description": str(defaults["object_description"]),
        "json_output_contract": str(defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(defaults["json_output_contract_answer_only"]),
        "annotation_hint": str(annotation_hint),
        "answer_hint": str(defaults["answer_hint"]),
        "json_example": str(json_example),
        "json_example_answer_only": str(json_example_answer_only),
    }
    if rotation_instruction is not None:
        slots["rotation_instruction"] = str(rotation_instruction)
    selection = render_scene_prompt_variants(
        domain="geometry",
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(defaults["task_key"]),
        query_key=str(prompt_branch_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        slots=slots,
        instance_seed=int(instance_seed),
    )
    return defaults, build_prompt_trace_artifacts(selection)


__all__ = ["relation_prompt_artifacts", "transformation_prompt_artifacts"]
