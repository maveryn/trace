"""Prompt helpers for symbolic truth-table tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from ....shared.config_defaults import required_group_defaults
from ....shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


@dataclass(frozen=True)
class TruthPromptRuntime:
    prompt: str
    prompt_variants: dict[str, str]
    metadata: dict[str, Any]
    artifacts: Any


def render_truth_prompt(
    prompt_defaults: Mapping[str, Any],
    *,
    domain: str,
    scene_id: str,
    scene_variant: str,
    task_key: str,
    object_description_key: str,
    annotation_hint_key: str,
    answer_hint_key: str,
    json_example_key: str,
    json_example_answer_only_key: str,
    instance_seed: int,
    context: str,
) -> TruthPromptRuntime:
    """Render one truth-table prompt from external templates."""

    prompt_values = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            object_description_key,
            annotation_hint_key,
            answer_hint_key,
            json_example_key,
            json_example_answer_only_key,
        ),
        context=str(context),
    )
    selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=str(scene_id),
        bundle_id=str(prompt_values["bundle_id"]),
        scene_key=str(prompt_values["scene_key"]),
        task_key=str(task_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(prompt_values[object_description_key]),
            "json_output_contract": str(prompt_values["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_values["json_output_contract_answer_only"]),
            "annotation_hint": str(prompt_values[annotation_hint_key]),
            "answer_hint": str(prompt_values[answer_hint_key]),
            "json_example": str(prompt_values[json_example_key]),
            "json_example_answer_only": str(prompt_values[json_example_answer_only_key]),
        },
        instance_seed=int(instance_seed),
    )
    artifacts = build_prompt_trace_artifacts(selection)
    return TruthPromptRuntime(
        prompt=str(artifacts.prompt),
        prompt_variants=dict(artifacts.prompt_variants),
        metadata={
            "prompt_variant": dict(artifacts.prompt_variant),
            "prompt_variant_active_key": str(artifacts.prompt_variant_active_key),
            "prompt_variants_for_trace": dict(artifacts.prompt_variants_for_trace),
            "bundle_id": str(prompt_values["bundle_id"]),
        },
        artifacts=artifacts,
    )


__all__ = ["TruthPromptRuntime", "render_truth_prompt"]
