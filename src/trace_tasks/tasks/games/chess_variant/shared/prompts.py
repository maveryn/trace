"""Prompt assembly helpers for chess-variant games tasks."""

from __future__ import annotations

import json
from typing import Any, Mapping

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
    "marked_piece_rule_text",
    "landing_rule_text",
    "destination_landing_rule_text",
    "capture_landing_rule_text",
    "slider_block_rule_text",
    "leaper_block_rule_text",
    "answer_hint_marked_piece_destination_count",
    "answer_hint_marked_piece_capture_count",
    "annotation_hint_marked_piece_destination_count",
    "annotation_hint_marked_piece_capture_count",
    "rule_text_straight_range",
    "rule_text_diagonal_range",
    "rule_text_straight_or_diagonal_range",
    "rule_text_leaper_2_1",
    "rule_text_leaper_3_1",
    "rule_badge_straight_range",
    "rule_badge_diagonal_range",
    "rule_badge_straight_or_diagonal_range",
    "rule_badge_leaper_2_1",
    "rule_badge_leaper_3_1",
)


def prompt_defaults() -> dict[str, Any]:
    """Return required prompt wiring and code-facing dynamic slot defaults."""

    return required_group_defaults(
        _PROMPT_DEFAULTS,
        (*PROMPT_WIRING_KEYS, *PROMPT_DYNAMIC_DEFAULT_KEYS),
        context="chess-variant prompt defaults",
    )


def rule_text(rule_family: str, range_k: int, defaults: Mapping[str, Any]) -> str:
    """Return movement-rule text for the active prompt query."""

    return str(defaults[f"rule_text_{str(rule_family)}"]).format(range_k=int(range_k))


def block_rule_text(rule_family: str, defaults: Mapping[str, Any]) -> str:
    """Return blocking-rule text for the active prompt query."""

    key = "leaper_block_rule_text" if str(rule_family).startswith("leaper") else "slider_block_rule_text"
    return str(defaults[key])


def json_examples(*, point_annotation: bool, answer_value: int) -> tuple[str, str]:
    """Return valid output-format examples for one annotation contract."""

    if bool(point_annotation):
        annotation_value = [[180 + (70 * index), 245 + (55 * index)] for index in range(max(0, int(answer_value)))]
    else:
        annotation_value = [
            [140 + (70 * index), 220, 210 + (70 * index), 290]
            for index in range(max(0, int(answer_value)))
        ]
    return (
        json.dumps({"annotation": annotation_value, "answer": int(answer_value)}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": int(answer_value)}, separators=(",", ":"), ensure_ascii=False),
    )


def build_chess_variant_prompt_artifacts(
    *,
    domain: str,
    prompt_query_key: str,
    scene_variant: str,
    rule_family: str,
    range_k: int,
    landing_rule_text: str,
    point_annotation: bool,
    example_answer: int,
    instance_seed: int,
) -> tuple[dict[str, Any], Any]:
    """Build prompt artifacts from the external chess-variant prompt bundle."""

    defaults = prompt_defaults()
    example, example_answer_only = json_examples(point_annotation=bool(point_annotation), answer_value=int(example_answer))
    landing_text = str(landing_rule_text) if str(landing_rule_text) else str(defaults["landing_rule_text"])
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
            "rule_text": rule_text(str(rule_family), int(range_k), defaults),
            "marked_piece_rule_text": str(defaults["marked_piece_rule_text"]),
            "landing_rule_text": str(landing_text),
            "block_rule_text": block_rule_text(str(rule_family), defaults),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = [
    "build_chess_variant_prompt_artifacts",
    "block_rule_text",
    "json_examples",
    "prompt_defaults",
    "rule_text",
]
