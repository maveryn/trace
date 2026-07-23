"""Prompt rendering helpers for similar-figure tasks."""

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

from .state import SCENE_ID


def similar_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_branch_key: str,
    target_name: str,
    annotation_keys: Sequence[str],
    answer_value: int | float,
    answer_hint_key: str,
    instance_seed: int,
) -> tuple[Mapping[str, Any], PromptTraceArtifacts]:
    """Render prompt variants after the public task binds the target slots."""

    defaults = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "object_description",
            "json_output_contract",
            "json_output_contract_answer_only",
            "annotation_hint",
            answer_hint_key,
        ),
        context="similar-figure prompt defaults",
    )
    json_example, json_example_answer_only = build_keyed_point_prompt_json_examples(
        annotation_keys=tuple(str(key) for key in annotation_keys),
        answer=answer_value,
    )
    annotation_key_list = ", ".join(f'"{key}"' for key in annotation_keys)
    annotation_hint = str(defaults["annotation_hint"]).format(annotation_keys=annotation_key_list)
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
            "target_name": str(target_name),
            "variable_name": str(target_name),
            "json_output_contract": str(defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(defaults["json_output_contract_answer_only"]),
            "annotation_hint": str(annotation_hint),
            "answer_hint": str(defaults[answer_hint_key]),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return defaults, build_prompt_trace_artifacts(selection)
