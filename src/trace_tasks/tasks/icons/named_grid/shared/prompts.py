"""Prompt artifact helpers for named-grid icon tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ....shared.config_defaults import required_group_defaults
from ....shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)

from .defaults import SCENE_ID


_COMMON_PROMPT_KEYS = (
    "bundle_id",
    "scene_key",
    "task_key",
    "json_output_contract",
    "json_output_contract_answer_only",
    "object_description",
)


def build_named_grid_prompt_artifacts(
    *,
    domain: str,
    run_namespace: str,
    prompt_defaults_map: Mapping[str, Any],
    question_key: str,
    question_slots: Mapping[str, Any],
    annotation_slots: Mapping[str, Any],
    answer_slots: Mapping[str, Any] | None = None,
    instance_seed: int,
    extra_required_keys: Sequence[str] = (),
    annotation_hint_key: str = "annotation_hint",
    answer_hint_key: str = "answer_hint",
    json_example_key: str = "json_example",
    json_example_answer_only_key: str = "json_example_answer_only",
) -> tuple[PromptTraceArtifacts, Mapping[str, Any]]:
    """Build prompt artifacts from task-owned prompt keys and slots."""

    required_keys = tuple(
        dict.fromkeys(
            (
                *_COMMON_PROMPT_KEYS,
                str(question_key),
                str(annotation_hint_key),
                str(answer_hint_key),
                str(json_example_key),
                str(json_example_answer_only_key),
                *tuple(str(key) for key in extra_required_keys),
            )
        )
    )
    prompt_defaults = required_group_defaults(
        dict(prompt_defaults_map),
        required_keys,
        context=f"prompt defaults for {run_namespace}",
    )
    prompt_selection = render_task_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(prompt_defaults["object_description"]),
            "question_text": str(prompt_defaults[str(question_key)]).format(**dict(question_slots)),
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "annotation_hint": str(prompt_defaults[str(annotation_hint_key)]).format(**dict(annotation_slots)),
            "answer_hint": str(prompt_defaults[str(answer_hint_key)]).format(**dict(answer_slots or {})),
            "json_example": str(prompt_defaults[str(json_example_key)]),
            "json_example_answer_only": str(prompt_defaults[str(json_example_answer_only_key)]),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection), prompt_defaults


__all__ = ["build_named_grid_prompt_artifacts"]
