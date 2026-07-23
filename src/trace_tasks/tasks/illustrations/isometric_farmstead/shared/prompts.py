"""Prompt rendering helpers for isometric farmstead tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)
from trace_tasks.tasks.shared.config_defaults import required_group_defaults


def build_isometric_farmstead_prompt_artifacts(
    *,
    domain: str,
    scene_id: str,
    prompt_defaults: Mapping[str, Any],
    prompt_query_key: str,
    slots: Mapping[str, Any],
    instance_seed: int,
) -> Any:
    """Render external prompt templates for one isometric farmstead task."""

    selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=str(scene_id),
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_query_key),
        slots=dict(slots),
        instance_seed=int(instance_seed),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        preferred_mode="answer_and_annotation",
    )
    return build_prompt_trace_artifacts(selection)


def build_isometric_farmstead_task_prompt_with_default_slots(
    *,
    domain: str,
    scene_id: str,
    prompt_defaults_source: Mapping[str, Any],
    prompt_query_key: str,
    answer_hint_key: str,
    annotation_hint_key: str,
    json_example_key: str,
    json_example_answer_only_key: str,
    context_label: str,
    instance_seed: int,
) -> tuple[Mapping[str, Any], Any]:
    """Resolve prompt defaults and render common JSON-output prompt slots."""

    prompt_defaults = required_group_defaults(
        prompt_defaults_source,
        [
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            str(answer_hint_key),
            str(annotation_hint_key),
            str(json_example_key),
            str(json_example_answer_only_key),
        ],
        context=f"prompt defaults for {context_label}",
    )
    artifacts = build_isometric_farmstead_prompt_artifacts(
        domain=str(domain),
        scene_id=str(scene_id),
        prompt_defaults=prompt_defaults,
        prompt_query_key=str(prompt_query_key),
        slots={
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "answer_hint": str(prompt_defaults[str(answer_hint_key)]),
            "annotation_hint": str(prompt_defaults[str(annotation_hint_key)]),
            "json_example": str(prompt_defaults[str(json_example_key)]),
            "json_example_answer_only": str(prompt_defaults[str(json_example_answer_only_key)]),
        },
        instance_seed=int(instance_seed),
    )
    return prompt_defaults, artifacts


__all__ = [
    "build_isometric_farmstead_prompt_artifacts",
    "build_isometric_farmstead_task_prompt_with_default_slots",
]
