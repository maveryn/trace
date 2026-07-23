"""Prompt helpers for paired-canvas icon tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

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
    "annotation_hint",
    "answer_hint",
    "json_example",
    "json_example_answer_only",
)


def required_paired_prompt_defaults(
    prompt_defaults: Mapping[str, Any],
    *,
    run_namespace: str,
    extra_required_keys: Sequence[str] = (),
) -> Dict[str, Any]:
    """Return prompt defaults needed by a paired-canvas task."""

    required = required_group_defaults(
        prompt_defaults,
        tuple(dict.fromkeys((*_COMMON_PROMPT_KEYS, *tuple(str(key) for key in extra_required_keys)))),
        context=f"prompt defaults for {run_namespace}",
    )
    merged = dict(prompt_defaults)
    merged.update(required)
    return merged


def build_paired_prompt(
    *,
    domain: str,
    prompt_defaults: Mapping[str, Any],
    question_text: str,
    instance_seed: int,
) -> PromptTraceArtifacts:
    """Render prompt variants for one paired-canvas task."""

    prompt_selection = render_task_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        slots={
            "object_description": str(prompt_defaults["object_description"]),
            "question_text": str(question_text),
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "annotation_hint": str(prompt_defaults["annotation_hint"]),
            "answer_hint": str(prompt_defaults["answer_hint"]),
            "json_example": str(prompt_defaults["json_example"]),
            "json_example_answer_only": str(prompt_defaults["json_example_answer_only"]),
        },
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_paired_prompt", "required_paired_prompt_defaults"]
