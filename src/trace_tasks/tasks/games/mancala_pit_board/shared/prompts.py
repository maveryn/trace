"""Prompt asset helpers for Mancala pit-board tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import PROMPT_OUTPUT_MODES, PromptTraceArtifacts, build_prompt_trace_artifacts, render_scene_prompt_variants

from .state import SCENE_ID


@dataclass(frozen=True)
class MancalaPromptSlots:
    """Task-owned prompt keys and examples for one Mancala objective."""

    prompt_query_key: str
    answer_hint_key: str
    annotation_hint_key: str
    example_annotation: Mapping[str, Any] | list[Any]
    example_answer: Any


@dataclass(frozen=True)
class MancalaPromptContext:
    """Dynamic prompt context for one rendered Mancala instance."""

    prompt_query_key: str
    answer_hint_key: str
    annotation_hint_key: str
    scene_variant: str
    json_example: str
    json_example_answer_only: str


def format_mancala_json_examples(*, annotation: Mapping[str, Any] | list[Any], answer: Any) -> tuple[str, str]:
    """Format task-owned annotation and answer examples for prompt slots."""

    return (
        json.dumps({"annotation": annotation, "answer": answer}, separators=(",", ":"), ensure_ascii=True),
        json.dumps({"answer": answer}, separators=(",", ":"), ensure_ascii=True),
    )


def make_mancala_prompt_slots(
    *,
    prompt_query_key: str,
    answer_hint_key: str,
    annotation_hint_key: str,
    example_annotation: Mapping[str, Any] | list[Any],
    example_answer: Any,
) -> MancalaPromptSlots:
    """Create immutable prompt-slot metadata from task-owned keys."""

    return MancalaPromptSlots(
        prompt_query_key=str(prompt_query_key),
        answer_hint_key=str(answer_hint_key),
        annotation_hint_key=str(annotation_hint_key),
        example_annotation=example_annotation,
        example_answer=example_answer,
    )


def build_mancala_prompt_artifacts(
    *,
    domain: str,
    prompt_defaults: Mapping[str, Any],
    context: MancalaPromptContext,
    instance_seed: int,
) -> tuple[Mapping[str, Any], PromptTraceArtifacts]:
    """Render prompt variants from the Mancala scene bundle and dynamic slots."""

    required_keys = (
        "bundle_id",
        "scene_key",
        "task_key",
        "json_output_contract",
        "json_output_contract_answer_only",
        f"object_description_{str(context.scene_variant)}",
        "sowing_rule_text",
        str(context.answer_hint_key),
        str(context.annotation_hint_key),
    )
    resolved_defaults = required_group_defaults(
        prompt_defaults,
        tuple(required_keys),
        context=f"prompt defaults for {SCENE_ID}",
    )
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(resolved_defaults["bundle_id"]),
        scene_key=str(resolved_defaults["scene_key"]),
        task_key=str(resolved_defaults["task_key"]),
        query_key=str(context.prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(resolved_defaults[f"object_description_{str(context.scene_variant)}"]),
            "sowing_rule_text": str(resolved_defaults["sowing_rule_text"]),
            "json_output_contract": str(resolved_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(resolved_defaults["json_output_contract_answer_only"]),
            "answer_hint": str(resolved_defaults[str(context.answer_hint_key)]),
            "annotation_hint": str(resolved_defaults[str(context.annotation_hint_key)]),
            "json_example": str(context.json_example),
            "json_example_answer_only": str(context.json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return dict(resolved_defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = [
    "MancalaPromptContext",
    "MancalaPromptSlots",
    "build_mancala_prompt_artifacts",
    "format_mancala_json_examples",
    "make_mancala_prompt_slots",
]
