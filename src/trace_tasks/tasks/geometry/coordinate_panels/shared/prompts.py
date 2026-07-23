"""Prompt assembly helpers for coordinate-panel geometry scenes."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_json_example import resolve_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_scene_prompt_variants


def build_coordinate_panel_prompt_artifacts(
    *,
    domain: str,
    scene_id: str,
    bundle_id: str,
    prompt_defaults_all: Mapping[str, Any],
    prompt_query_key: str,
    object_description_key: str,
    annotation_hint_key: str,
    annotation_value: Any,
    answer_type: str,
    params: Mapping[str, Any],
    instance_seed: int,
    context: str,
) -> Any:
    """Render prompt variants from caller-selected scene/task/query semantics."""

    prompt_defaults = required_group_defaults(
        prompt_defaults_all,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            str(object_description_key),
            str(annotation_hint_key),
            "answer_hint_option_letter",
        ),
        context=str(context),
    )
    json_example, json_example_answer_only = resolve_prompt_json_examples(
        prompt_defaults_all,
        annotation_value=annotation_value,
        answer_type=str(answer_type),
    )
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=str(scene_id),
        bundle_id=str(prompt_defaults.get("bundle_id", bundle_id)),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        slots={
            "object_description": str(prompt_defaults[str(object_description_key)]),
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "annotation_hint": str(prompt_defaults[str(annotation_hint_key)]),
            "answer_hint": str(prompt_defaults["answer_hint_option_letter"]),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
        preferred_mode=str(params.get("prompt_mode", "answer_and_annotation")),
    )
    return build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_coordinate_panel_prompt_artifacts"]
