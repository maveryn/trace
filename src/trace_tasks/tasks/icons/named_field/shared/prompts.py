"""Prompt artifact helpers for named-field icon tasks."""

from __future__ import annotations

from typing import Any, Mapping

from ....shared.config_defaults import required_group_defaults
from ....shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)
from ...shared.procedural_named_icon_field_scene import SCENE_ID

from .metrics import boolean_attribute_phrase


def build_shape_count_prompt_artifacts(
    *,
    domain: str,
    run_namespace: str,
    prompt_defaults_map: Mapping[str, Any],
    sample: Any,
    instance_seed: int,
) -> tuple[PromptTraceArtifacts, Mapping[str, Any]]:
    """Build prompt artifacts for a direct named-shape count sample."""

    prompt_defaults = required_group_defaults(
        dict(prompt_defaults_map),
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "object_description",
            "question_text",
            "annotation_hint",
            "answer_hint",
            "json_example",
            "json_example_answer_only",
        ),
        context=f"prompt defaults for {run_namespace}",
    )
    prompt_selection = render_task_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        slots={
            "object_description": str(prompt_defaults["object_description"]),
            "question_text": str(prompt_defaults["question_text"]).format(shape_name=str(sample.target_shape_name)),
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "annotation_hint": str(prompt_defaults["annotation_hint"]).format(shape_name=str(sample.target_shape_name)),
            "answer_hint": str(prompt_defaults["answer_hint"]),
            "json_example": str(prompt_defaults["json_example"]),
            "json_example_answer_only": str(prompt_defaults["json_example_answer_only"]),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection), prompt_defaults


def build_boolean_prompt_artifacts(
    *,
    domain: str,
    run_namespace: str,
    prompt_defaults_map: Mapping[str, Any],
    sample: Any,
    instance_seed: int,
) -> tuple[PromptTraceArtifacts, Mapping[str, Any]]:
    """Build prompt artifacts for a Boolean named-field count sample."""

    question_key = f"question_text_{sample.prompt_query_key}"
    prompt_defaults = required_group_defaults(
        dict(prompt_defaults_map),
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "object_description",
            question_key,
            "annotation_hint",
            "answer_hint",
            "json_example",
            "json_example_answer_only",
        ),
        context=f"prompt defaults for {run_namespace}",
    )
    attribute_phrase = boolean_attribute_phrase(sample)
    prompt_selection = render_task_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        slots={
            "object_description": str(prompt_defaults["object_description"]),
            "question_text": str(prompt_defaults[question_key]).format(
                shape_name=str(sample.target_shape_name),
                color_label=str(sample.target_attribute_label),
                attribute_phrase=str(attribute_phrase),
            ),
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "annotation_hint": str(prompt_defaults["annotation_hint"]).format(
                shape_name=str(sample.target_shape_name),
                color_label=str(sample.target_attribute_label),
                attribute_phrase=str(attribute_phrase),
            ),
            "answer_hint": str(prompt_defaults["answer_hint"]),
            "json_example": str(prompt_defaults["json_example"]),
            "json_example_answer_only": str(prompt_defaults["json_example_answer_only"]),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection), prompt_defaults


def build_counterfactual_prompt_artifacts(
    *,
    domain: str,
    run_namespace: str,
    prompt_defaults_map: Mapping[str, Any],
    sample: Any,
    instance_seed: int,
) -> tuple[PromptTraceArtifacts, Mapping[str, Any]]:
    """Build prompt artifacts for a counterfactual named-field count sample."""

    question_key = f"question_text_{sample.prompt_query_key}"
    prompt_defaults = required_group_defaults(
        dict(prompt_defaults_map),
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "object_description",
            question_key,
            "annotation_hint",
            "answer_hint",
            "json_example",
            "json_example_answer_only",
        ),
        context=f"prompt defaults for {run_namespace}",
    )
    prompt_selection = render_task_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        slots={
            "object_description": str(prompt_defaults["object_description"]),
            "question_text": str(prompt_defaults[question_key]).format(
                source_shape_name=str(sample.source_shape_name),
                target_shape_name=str(sample.target_shape_name),
                remove_shape_name=str(sample.remove_shape_name),
            ),
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "annotation_hint": str(prompt_defaults["annotation_hint"]),
            "answer_hint": str(prompt_defaults["answer_hint"]),
            "json_example": str(prompt_defaults["json_example"]),
            "json_example_answer_only": str(prompt_defaults["json_example_answer_only"]),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection), prompt_defaults


__all__ = [
    "build_boolean_prompt_artifacts",
    "build_counterfactual_prompt_artifacts",
    "build_shape_count_prompt_artifacts",
]
