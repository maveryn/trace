"""Prompt assembly helpers for Reversi scene-package tasks."""

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


def reversi_json_examples(*, annotation_type: str) -> Tuple[str, str]:
    """Return format-only examples with answer and annotation cardinality aligned."""

    if str(annotation_type) == "bbox_set":
        annotation = [[112, 184, 176, 248], [184, 184, 248, 248]]
        answer = 2
    else:
        annotation = [[144, 216], [216, 216], [288, 216]]
        answer = 3
    return (
        json.dumps(
            {"annotation": annotation, "answer": answer},
            separators=(",", ":"),
            ensure_ascii=True,
        ),
        json.dumps({"answer": answer}, separators=(",", ":"), ensure_ascii=True),
    )


def build_reversi_prompt_artifacts(
    *,
    domain: str,
    scene_variant: str,
    prompt_query_key: str,
    current_player_name: str,
    query_player_name: str,
    annotation_type: str,
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
) -> tuple[Dict[str, Any], Any]:
    """Build Reversi prompt artifacts from external prompt templates."""

    object_description_key = f"object_description_{str(scene_variant)}"
    answer_hint_key = f"answer_hint_{str(prompt_query_key)}"
    annotation_hint_key = f"annotation_hint_{str(prompt_query_key)}"
    defaults = required_group_defaults(
        prompt_defaults,
        (
            *_PROMPT_WIRING_KEYS,
            object_description_key,
            "legal_move_rule_text",
            "marked_move_rule_text",
            "flip_rule_text",
            "frontier_rule_text",
            answer_hint_key,
            annotation_hint_key,
        ),
        context="reversi prompt wiring defaults",
    )
    json_example, json_example_answer_only = reversi_json_examples(
        annotation_type=str(annotation_type)
    )
    format_slots = {
        "query_player": str(query_player_name).strip() or str(current_player_name)
    }
    answer_hint = str(defaults[answer_hint_key]).format_map(format_slots)
    annotation_hint = str(defaults[annotation_hint_key]).format_map(format_slots)
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(defaults[object_description_key]),
            "json_output_contract": str(defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(
                defaults["json_output_contract_answer_only"]
            ),
            "answer_hint": str(answer_hint),
            "annotation_hint": str(annotation_hint),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
            "current_player_name": str(current_player_name),
            "query_player": str(query_player_name).strip() or str(current_player_name),
            "legal_move_rule_text": str(defaults["legal_move_rule_text"]),
            "marked_move_rule_text": str(defaults["marked_move_rule_text"]),
            "flip_rule_text": str(defaults["flip_rule_text"]),
            "frontier_rule_text": str(defaults["frontier_rule_text"]),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_reversi_prompt_artifacts", "reversi_json_examples"]
