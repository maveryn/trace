"""Prompt loading for polar graph paper tasks."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_json_example import resolve_prompt_json_examples
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


def polar_graph_prompt_artifacts(
    *,
    scene_id: str,
    prompt_defaults_all: Mapping[str, Any],
    prompt_query_key: str,
    annotation_value: Any,
    instance_seed: int,
    answer_hint_key: str = "answer_hint_integer",
    annotation_hint_key: str = "annotation_hint_point_p",
    answer_type: str = "integer",
    extra_dynamic_slots: Mapping[str, Any] | None = None,
    json_examples: tuple[str, str] | None = None,
) -> PromptTraceArtifacts:
    """Render v1 prompt variants from task-selected readout semantics."""

    prompt_defaults = required_group_defaults(
        prompt_defaults_all,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "object_description",
            str(annotation_hint_key),
            str(answer_hint_key),
        ),
        context=f"prompt defaults for {scene_id}",
    )
    if json_examples is None:
        json_example, json_example_answer_only = resolve_prompt_json_examples(
            prompt_defaults_all,
            annotation_value=annotation_value,
            answer_type=str(answer_type),
        )
    else:
        json_example, json_example_answer_only = json_examples
    dynamic_slots = {
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "annotation_hint": str(prompt_defaults[str(annotation_hint_key)]),
        "answer_hint": str(prompt_defaults[str(answer_hint_key)]),
        "json_example": str(json_example),
        "json_example_answer_only": str(json_example_answer_only),
    }
    dynamic_slots.update({str(key): value for key, value in dict(extra_dynamic_slots or {}).items()})

    prompt_selection = render_scene_prompt_variants(
        domain="geometry",
        scene_id=str(scene_id),
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)
