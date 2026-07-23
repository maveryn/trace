"""Prompt asset helpers for Mini-golf games tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import SCENE_ID


@dataclass(frozen=True)
class MinigolfPromptSlots:
    """Task-owned prompt keys, dynamic values, and output examples."""

    prompt_query_key: str
    object_description_key: str
    answer_hint_key: str
    annotation_hint_key: str
    example_annotation: Any
    example_answer: Any


def format_minigolf_json_examples(*, annotation: Any, answer: Any) -> tuple[str, str]:
    """Format task-owned examples for both output modes."""

    return (
        json.dumps({"annotation": annotation, "answer": answer}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": answer}, separators=(",", ":"), ensure_ascii=False),
    )


def build_minigolf_prompt_artifacts(
    *,
    domain: str,
    prompt_defaults: Mapping[str, Any],
    slots: MinigolfPromptSlots,
    instance_seed: int,
) -> tuple[Mapping[str, Any], PromptTraceArtifacts]:
    """Render prompt variants from the Mini-golf scene prompt bundle."""

    resolved_defaults = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "minigolf_cue_rule_text",
            "minigolf_bank_rule_text",
            str(slots.object_description_key),
            str(slots.answer_hint_key),
            str(slots.annotation_hint_key),
        ),
        context=f"prompt defaults for {SCENE_ID}",
    )
    json_example, json_example_answer_only = format_minigolf_json_examples(
        annotation=slots.example_annotation,
        answer=slots.example_answer,
    )
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(resolved_defaults["bundle_id"]),
        scene_key=str(resolved_defaults["scene_key"]),
        task_key=str(resolved_defaults["task_key"]),
        query_key=str(slots.prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(resolved_defaults[str(slots.object_description_key)]),
            "minigolf_cue_rule_text": str(resolved_defaults["minigolf_cue_rule_text"]),
            "minigolf_bank_rule_text": str(resolved_defaults["minigolf_bank_rule_text"]),
            "json_output_contract": str(resolved_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(resolved_defaults["json_output_contract_answer_only"]),
            "answer_hint": str(resolved_defaults[str(slots.answer_hint_key)]),
            "annotation_hint": str(resolved_defaults[str(slots.annotation_hint_key)]),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return dict(resolved_defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = [
    "MinigolfPromptSlots",
    "build_minigolf_prompt_artifacts",
    "format_minigolf_json_examples",
]
