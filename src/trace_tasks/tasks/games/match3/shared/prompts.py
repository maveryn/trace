"""Prompt asset helpers for match-3 game tasks."""

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
class Match3PromptSlots:
    """Task-owned prompt keys and output examples."""

    prompt_query_key: str
    object_description_key: str
    answer_hint_key: str
    annotation_hint_key: str
    example_annotation: list[Any]
    example_answer: Any


@dataclass(frozen=True)
class Match3PromptContext:
    """Dynamic prompt context for one rendered match-3 instance."""

    prompt_query_key: str
    object_description_key: str
    answer_hint_key: str
    annotation_hint_key: str
    target_color_label: str
    row_index: int | None
    col_index: int | None
    json_example: str
    json_example_answer_only: str


def format_match3_json_examples(*, annotation: list[Any], answer: Any) -> tuple[str, str]:
    """Format task-owned examples for both prompt output modes."""

    return (
        json.dumps({"annotation": annotation, "answer": answer}, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
        json.dumps({"answer": answer}, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
    )


def make_match3_prompt_slots(
    *,
    prompt_query_key: str,
    object_description_key: str,
    answer_hint_key: str,
    annotation_hint_key: str,
    example_annotation: list[Any],
    example_answer: Any,
) -> Match3PromptSlots:
    """Create immutable prompt-slot metadata from task-owned keys."""

    return Match3PromptSlots(
        prompt_query_key=str(prompt_query_key),
        object_description_key=str(object_description_key),
        answer_hint_key=str(answer_hint_key),
        annotation_hint_key=str(annotation_hint_key),
        example_annotation=list(example_annotation),
        example_answer=example_answer,
    )


def build_match3_prompt_artifacts(
    *,
    domain: str,
    prompt_defaults: Mapping[str, Any],
    context: Match3PromptContext,
    instance_seed: int,
) -> tuple[Mapping[str, Any], PromptTraceArtifacts]:
    """Render prompt variants from the match-3 scene prompt bundle."""

    resolved_defaults = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            str(context.object_description_key),
            "match3_rule_text",
            str(context.answer_hint_key),
            str(context.annotation_hint_key),
        ),
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
            "object_description": str(resolved_defaults[str(context.object_description_key)]),
            "json_output_contract": str(resolved_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(resolved_defaults["json_output_contract_answer_only"]),
            "answer_hint": str(resolved_defaults[str(context.answer_hint_key)]),
            "annotation_hint": str(resolved_defaults[str(context.annotation_hint_key)]),
            "json_example": str(context.json_example),
            "json_example_answer_only": str(context.json_example_answer_only),
            "match3_rule_text": str(resolved_defaults["match3_rule_text"]),
            "target_color_label": str(context.target_color_label),
            "row_index": "" if context.row_index is None else str(int(context.row_index)),
            "col_index": "" if context.col_index is None else str(int(context.col_index)),
        },
        instance_seed=int(instance_seed),
    )
    return dict(resolved_defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = [
    "Match3PromptContext",
    "Match3PromptSlots",
    "build_match3_prompt_artifacts",
    "format_match3_json_examples",
    "make_match3_prompt_slots",
]
