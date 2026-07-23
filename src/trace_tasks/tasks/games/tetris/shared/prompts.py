"""Prompt assembly helpers for Tetris board scenes."""

from __future__ import annotations

import json
from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import PROMPT_DEFAULTS
from .state import DOMAIN, SCENE_ID


def format_json_examples(*, annotation: Any, answer: Any) -> tuple[str, str]:
    """Serialize schema-valid examples for both output modes."""

    return (
        json.dumps({"annotation": annotation, "answer": answer}, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
        json.dumps({"answer": answer}, ensure_ascii=True, allow_nan=False, separators=(",", ":")),
    )


def build_tetris_prompt_artifacts(
    *,
    prompt_query_key: str,
    answer_hint_key: str,
    annotation_hint_key: str,
    json_example: str,
    json_example_answer_only: str,
    dynamic_slots: Mapping[str, Any] | None,
    instance_seed: int,
) -> tuple[Mapping[str, Any], PromptTraceArtifacts]:
    """Render v1 Tetris prompts from scene assets and task-owned dynamic slots."""

    key = str(prompt_query_key)
    prompt_defaults = required_group_defaults(
        PROMPT_DEFAULTS,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "object_description_tetris_board",
            "next_piece_rule_text",
            "fixed_drop_rule_text",
            "collision_time_rule_text",
            "line_clear_rule_text",
            str(answer_hint_key),
            str(annotation_hint_key),
        ),
        context=f"prompt defaults for {SCENE_ID}/{key}",
    )
    slots = {
        "object_description": str(prompt_defaults["object_description_tetris_board"]),
        "next_piece_rule_text": str(prompt_defaults["next_piece_rule_text"]),
        "fixed_drop_rule_text": str(prompt_defaults["fixed_drop_rule_text"]),
        "collision_time_rule_text": str(prompt_defaults["collision_time_rule_text"]),
        "line_clear_rule_text": str(prompt_defaults["line_clear_rule_text"]),
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(prompt_defaults[str(answer_hint_key)]),
        "annotation_hint": str(prompt_defaults[str(annotation_hint_key)]),
        "json_example": str(json_example),
        "json_example_answer_only": str(json_example_answer_only),
    }
    slots.update(dict(dynamic_slots or {}))
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=key,
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=slots,
        instance_seed=int(instance_seed),
    )
    return dict(prompt_defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_tetris_prompt_artifacts", "format_json_examples"]
