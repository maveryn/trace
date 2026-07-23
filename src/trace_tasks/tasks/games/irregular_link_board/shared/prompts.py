"""Prompt assembly helpers for irregular-link-board scene tasks."""

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


def _json_examples() -> Tuple[str, str]:
    annotation = [[180.0, 220.0], [268.0, 220.0]]
    return (
        json.dumps({"annotation": annotation, "answer": 2}, separators=(",", ":"), ensure_ascii=True),
        json.dumps({"answer": 2}, separators=(",", ":"), ensure_ascii=True),
    )


def build_irregular_link_prompt_artifacts(
    *,
    domain: str,
    scene_variant: str,
    prompt_query_key: str,
    rule_slot_name: str,
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
) -> tuple[Dict[str, Any], Any]:
    """Build prompt artifacts for one task-owned query key."""

    answer_hint_key = f"answer_hint_{str(prompt_query_key)}"
    annotation_hint_key = f"annotation_hint_{str(prompt_query_key)}"
    required_keys = [
        "bundle_id",
        "scene_key",
        "task_key",
        "json_output_contract",
        "json_output_contract_answer_only",
        f"object_description_{str(scene_variant)}",
        answer_hint_key,
        annotation_hint_key,
        str(rule_slot_name),
    ]
    prompt_defaults = required_group_defaults(
        prompt_defaults,
        tuple(required_keys),
        context=f"prompt defaults for {SCENE_ID}",
    )
    json_example, json_example_answer_only = _json_examples()
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(prompt_defaults[f"object_description_{str(scene_variant)}"]),
            "movement_rule_text": str(prompt_defaults.get("movement_rule_text", "")),
            "capture_rule_text": str(prompt_defaults.get("capture_rule_text", "")),
            "json_output_contract": str(prompt_defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
            "answer_hint": str(prompt_defaults[answer_hint_key]),
            "annotation_hint": str(prompt_defaults[annotation_hint_key]),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return dict(prompt_defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_irregular_link_prompt_artifacts"]
