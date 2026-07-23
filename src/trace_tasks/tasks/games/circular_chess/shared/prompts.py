"""Prompt assembly helpers for circular-chess games tasks."""

from __future__ import annotations

import json
from typing import Any

from trace_tasks.tasks.games.shared.piece_board_rules import color_name
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults, required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_scene_prompt_variants

from .defaults import PROMPT_WIRING_KEYS, SCENE_ID


_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)

PROMPT_DYNAMIC_DEFAULT_KEYS: tuple[str, ...] = (
    "json_output_contract",
    "json_output_contract_answer_only",
    "object_description_sparse_board",
    "object_description_crowded_board",
    "circular_board_rule_text",
    "piece_rule_text_marked",
    "piece_rule_text_mixed",
    "landing_rule_text",
    "answer_hint_marked_piece_move_count",
    "answer_hint_marked_piece_capture_count",
    "answer_hint_white_piece_reaches_target_count",
    "answer_hint_black_piece_reaches_target_count",
    "annotation_hint_marked_piece_move_count",
    "annotation_hint_marked_piece_capture_count",
    "annotation_hint_white_piece_reaches_target_count",
    "annotation_hint_black_piece_reaches_target_count",
)


def prompt_defaults() -> dict[str, Any]:
    """Return required prompt wiring and code-facing dynamic slot defaults."""

    return required_group_defaults(
        _PROMPT_DEFAULTS,
        (*PROMPT_WIRING_KEYS, *PROMPT_DYNAMIC_DEFAULT_KEYS),
        context="circular-chess prompt defaults",
    )


def json_examples(*, answer_value: int) -> tuple[str, str]:
    """Return valid output-format examples for the point-set annotation contract."""

    annotation_value = [[420, 210], [502, 244], [536, 328]][: max(0, min(3, int(answer_value)))]
    return (
        json.dumps({"annotation": annotation_value, "answer": int(answer_value)}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": int(answer_value)}, separators=(",", ":"), ensure_ascii=False),
    )


def build_circular_chess_prompt_artifacts(
    *,
    domain: str,
    prompt_query_key: str,
    scene_variant: str,
    target_color: str,
    marked_piece_present: bool,
    example_answer: int,
    instance_seed: int,
) -> tuple[dict[str, Any], Any]:
    """Build prompt artifacts from the external circular-chess prompt bundle."""

    defaults = prompt_defaults()
    example, example_answer_only = json_examples(answer_value=int(example_answer))
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(defaults[f"object_description_{str(scene_variant)}"]),
            "json_output_contract": str(defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(defaults["json_output_contract_answer_only"]),
            "answer_hint": str(defaults[f"answer_hint_{str(prompt_query_key)}"]),
            "annotation_hint": str(defaults[f"annotation_hint_{str(prompt_query_key)}"]),
            "json_example": str(example),
            "json_example_answer_only": str(example_answer_only),
            "circular_board_rule_text": str(defaults["circular_board_rule_text"]),
            "piece_rule_text": str(defaults["piece_rule_text_marked"] if bool(marked_piece_present) else defaults["piece_rule_text_mixed"]),
            "landing_rule_text": str(defaults["landing_rule_text"]),
            "target_color_name": str(color_name(target_color)) if str(target_color) else "",
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = [
    "build_circular_chess_prompt_artifacts",
    "json_examples",
    "prompt_defaults",
]
