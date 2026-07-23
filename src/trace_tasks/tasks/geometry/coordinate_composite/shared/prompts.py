"""Prompt assembly helpers for coordinate-composite scene tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import group_default, required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_scene_prompt_variants


def build_coordinate_composite_prompt_artifacts(
    *,
    domain: str,
    scene_id: str,
    prompt_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    query_key: str,
    prompt_bundle_id: str,
    object_description_key: str,
    annotation_hint_key: str,
    answer_hint_key: str,
    json_example_key: str,
    json_example_answer_only_key: str,
    context: str,
) -> Any:
    """Build prompt variants from one coordinate-composite bundle key set."""

    prompt_params = required_group_defaults(
        {
            **dict(prompt_defaults),
            "bundle_id": str(group_default(prompt_defaults, "bundle_id", prompt_bundle_id)),
            "scene_key": str(group_default(prompt_defaults, "scene_key", "coordinate_composite_scene")),
            "task_key": str(group_default(prompt_defaults, "task_key", "coordinate_composite_query")),
        },
        (
            "bundle_id",
            "scene_key",
            "task_key",
            object_description_key,
            "json_output_contract",
            "json_output_contract_answer_only",
            annotation_hint_key,
            answer_hint_key,
            json_example_key,
            json_example_answer_only_key,
        ),
        context=str(context),
    )
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=str(scene_id),
        bundle_id=str(prompt_params["bundle_id"]),
        scene_key=str(prompt_params["scene_key"]),
        task_key=str(prompt_params["task_key"]),
        query_key=str(query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        slots={
            "object_description": str(prompt_params[object_description_key]),
            "json_output_contract": str(prompt_params["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_params["json_output_contract_answer_only"]),
            "annotation_hint": str(prompt_params[annotation_hint_key]),
            "answer_hint": str(prompt_params[answer_hint_key]),
            "json_example": str(prompt_params[json_example_key]),
            "json_example_answer_only": str(prompt_params[json_example_answer_only_key]),
        },
        instance_seed=int(instance_seed),
        preferred_mode=str(params.get("prompt_mode", "answer_and_annotation")),
    )
    return build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_coordinate_composite_prompt_artifacts"]
