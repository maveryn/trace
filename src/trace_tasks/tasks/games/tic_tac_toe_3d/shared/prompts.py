"""Prompt assembly helpers for 3D Tic-Tac-Toe scene tasks."""

from __future__ import annotations

import json
from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import DOMAIN, SCENE_ID

_PROMPT_WIRING_KEYS = (
    "bundle_id",
    "scene_key",
    "task_key",
    "json_output_contract",
    "json_output_contract_answer_only",
    "object_description_tic_tac_toe_3d",
    "tic_tac_toe_3d_winning_rule_text",
)


def format_json_examples(*, annotation: Any, answer: Any) -> tuple[str, str]:
    """Return format-only JSON examples aligned to the annotation shape."""

    return (
        json.dumps(
            {"annotation": annotation, "answer": answer},
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ),
        json.dumps(
            {"answer": answer},
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
        ),
    )


def build_tic_tac_toe_3d_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_query_key: str,
    answer_hint_key: str,
    annotation_hint_key: str,
    json_example: str,
    json_example_answer_only: str,
    target_layer_label: str,
    target_player: str,
    instance_seed: int,
) -> tuple[dict[str, Any], Any]:
    """Build prompt artifacts from external template assets."""

    defaults = required_group_defaults(
        prompt_defaults,
        (
            *_PROMPT_WIRING_KEYS,
            str(answer_hint_key),
            str(annotation_hint_key),
        ),
        context="3D Tic-Tac-Toe prompt wiring defaults",
    )
    format_slots = {"target_player": str(target_player)}
    answer_hint = str(defaults[str(answer_hint_key)]).format_map(format_slots)
    annotation_hint = str(defaults[str(annotation_hint_key)]).format_map(format_slots)
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(defaults["object_description_tic_tac_toe_3d"]),
            "json_output_contract": str(defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(
                defaults["json_output_contract_answer_only"]
            ),
            "answer_hint": str(answer_hint),
            "annotation_hint": str(annotation_hint),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
            "winning_rule_text": str(defaults["tic_tac_toe_3d_winning_rule_text"]),
            "target_layer_label": str(target_layer_label),
            "target_player": str(target_player),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_tic_tac_toe_3d_prompt_artifacts", "format_json_examples"]
