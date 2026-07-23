"""Prompt assembly helpers for rule-override board scene-package tasks."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import SCENE_ID


_PROMPT_WIRING_KEYS = (
    "bundle_id",
    "scene_key",
    "task_key",
    "json_output_contract",
    "json_output_contract_answer_only",
)


def rule_override_json_examples() -> Tuple[str, str]:
    """Return format-only JSON examples aligned to bbox-set count tasks."""

    annotation = [[80, 170, 258, 378], [286, 170, 464, 378], [492, 170, 670, 378]]
    answer = 3
    return (
        json.dumps({"annotation": annotation, "answer": answer}, separators=(",", ":"), ensure_ascii=True),
        json.dumps({"answer": answer}, separators=(",", ":"), ensure_ascii=True),
    )


def rule_text_from_prompt_defaults(*, prompt_defaults: Mapping[str, Any], rule_text_key: str) -> str:
    """Resolve one task-owned rule text slot from prompt assets."""

    defaults = required_group_defaults(
        prompt_defaults,
        (str(rule_text_key),),
        context="rule-override prompt rule text defaults",
    )
    return str(defaults[str(rule_text_key)])


def build_rule_override_prompt_artifacts(
    *,
    domain: str,
    prompt_query_key: str,
    prompt_defaults: Mapping[str, Any],
    target_player: str,
    rule_text: str,
    instance_seed: int,
) -> tuple[Dict[str, Any], Any]:
    """Build prompt artifacts from external prompt templates."""

    answer_hint_key = f"answer_hint_{str(prompt_query_key)}"
    annotation_hint_key = f"annotation_hint_{str(prompt_query_key)}"
    defaults = required_group_defaults(
        prompt_defaults,
        (
            *_PROMPT_WIRING_KEYS,
            "object_description_rule_override_board",
            answer_hint_key,
            annotation_hint_key,
        ),
        context="rule-override prompt wiring defaults",
    )
    json_example, json_example_answer_only = rule_override_json_examples()
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(defaults["object_description_rule_override_board"]),
            "target_player": str(target_player),
            "rule_text": str(rule_text),
            "json_output_contract": str(defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(defaults["json_output_contract_answer_only"]),
            "answer_hint": str(defaults[answer_hint_key]),
            "annotation_hint": str(defaults[annotation_hint_key]),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = [
    "build_rule_override_prompt_artifacts",
    "rule_override_json_examples",
    "rule_text_from_prompt_defaults",
]
