"""Prompt rendering helpers for Ludo board scene tasks."""

from __future__ import annotations

import json
from collections.abc import Mapping as MappingABC
from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import PROMPT_OUTPUT_MODES, PromptTraceArtifacts, build_prompt_trace_artifacts, render_scene_prompt_variants

from .state import SCENE_ID


@dataclass(frozen=True)
class LudoPromptContext:
    """Task-owned prompt slots for one Ludo objective."""

    prompt_query_key: str
    rule_slot_name: str
    answer_hint_key: str
    annotation_hint_key: str
    query_color: str
    target_color: str | None
    json_example: str
    json_example_answer_only: str


@dataclass(frozen=True)
class LudoPromptSlots:
    """Prompt asset keys and examples owned by one public Ludo task."""

    prompt_query_key: str
    rule_slot_name: str
    answer_hint_key: str
    annotation_hint_key: str
    json_example: str
    json_example_answer_only: str


def format_ludo_json_examples(*, annotation: Any, answer: Any) -> tuple[str, str]:
    """Format task-owned annotation and answer examples for prompt slots."""

    annotation_value = dict(annotation) if isinstance(annotation, MappingABC) else annotation
    return (
        json.dumps({"annotation": annotation_value, "answer": answer}, separators=(",", ":"), ensure_ascii=True),
        json.dumps({"answer": answer}, separators=(",", ":"), ensure_ascii=True),
    )


def make_ludo_prompt_slots(
    *,
    prompt_query_key: str,
    rule_slot_name: str,
    answer_hint_key: str,
    annotation_hint_key: str,
    example_annotation: Any,
    example_answer: Any,
) -> LudoPromptSlots:
    """Build prompt slots from task-owned keys and a format-only JSON example."""

    json_example, json_example_answer_only = format_ludo_json_examples(
        annotation=example_annotation,
        answer=example_answer,
    )
    return LudoPromptSlots(
        prompt_query_key=str(prompt_query_key),
        rule_slot_name=str(rule_slot_name),
        answer_hint_key=str(answer_hint_key),
        annotation_hint_key=str(annotation_hint_key),
        json_example=str(json_example),
        json_example_answer_only=str(json_example_answer_only),
    )


def make_ludo_prompt_slots_from_keys(
    *,
    keys: tuple[str, str, str, str],
    example_annotation: Any,
    example_answer: Any,
) -> LudoPromptSlots:
    """Build prompt slots from ordered task keys plus a format-only JSON example."""

    prompt_query_key, rule_slot_name, answer_hint_key, annotation_hint_key = keys
    return make_ludo_prompt_slots(
        prompt_query_key=str(prompt_query_key),
        rule_slot_name=str(rule_slot_name),
        answer_hint_key=str(answer_hint_key),
        annotation_hint_key=str(annotation_hint_key),
        example_annotation=example_annotation,
        example_answer=example_answer,
    )


def build_ludo_prompt_artifacts(
    *,
    domain: str,
    prompt_defaults: Mapping[str, Any],
    context: LudoPromptContext,
    instance_seed: int,
) -> tuple[Mapping[str, Any], PromptTraceArtifacts]:
    """Render v1 prompt assets for one task-owned Ludo prompt context."""

    required_keys = (
        "bundle_id",
        "scene_key",
        "task_key",
        "json_output_contract",
        "json_output_contract_answer_only",
        "object_description",
        str(context.rule_slot_name),
        str(context.answer_hint_key),
        str(context.annotation_hint_key),
    )
    resolved_defaults = required_group_defaults(
        prompt_defaults,
        required_keys,
        context=f"prompt defaults for {SCENE_ID}",
    )
    dynamic_slots = {
        "object_description": str(resolved_defaults["object_description"]),
        "query_color": str(context.query_color),
        "target_color": "" if context.target_color is None else str(context.target_color),
        "exact_finish_rule_text": str(resolved_defaults.get("exact_finish_rule_text", "")),
        "capture_option_rule_text": str(resolved_defaults.get("capture_option_rule_text", "")),
        "move_sequence_rule_text": str(resolved_defaults.get("move_sequence_rule_text", "")),
        "json_output_contract": str(resolved_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(resolved_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(resolved_defaults[str(context.answer_hint_key)]),
        "annotation_hint": str(resolved_defaults[str(context.annotation_hint_key)]),
        "json_example": str(context.json_example),
        "json_example_answer_only": str(context.json_example_answer_only),
    }
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(resolved_defaults["bundle_id"]),
        scene_key=str(resolved_defaults["scene_key"]),
        task_key=str(resolved_defaults["task_key"]),
        query_key=str(context.prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
    )
    return resolved_defaults, build_prompt_trace_artifacts(prompt_selection)


__all__ = [
    "LudoPromptContext",
    "LudoPromptSlots",
    "build_ludo_prompt_artifacts",
    "format_ludo_json_examples",
    "make_ludo_prompt_slots_from_keys",
    "make_ludo_prompt_slots",
]
