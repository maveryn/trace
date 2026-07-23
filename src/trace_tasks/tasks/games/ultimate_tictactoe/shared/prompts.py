"""Prompt rendering helpers for Ultimate Tic-Tac-Toe."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import PROMPT_DEFAULTS
from .state import DOMAIN, SCENE_ID


def build_ultimate_prompt_artifacts(
    *,
    prompt_key: str,
    json_example: str,
    json_example_answer_only: str,
    instance_seed: int,
    dynamic_values: Mapping[str, Any] | None = None,
) -> PromptTraceArtifacts:
    """Render v1 prompt variants for a task-owned semantic prompt key."""

    prompt_defaults = required_group_defaults(
        PROMPT_DEFAULTS,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "object_description_ultimate_tictactoe_board",
            f"answer_hint_{str(prompt_key)}",
            f"annotation_hint_{str(prompt_key)}",
            "ultimate_tictactoe_status_rule_text",
            "ultimate_tictactoe_tactic_rule_text",
            "ultimate_tictactoe_macro_threat_rule_text",
        ),
        context=f"prompt defaults for {str(prompt_key)}",
    )
    dynamic_slots = {
        "object_description": str(prompt_defaults["object_description_ultimate_tictactoe_board"]),
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(prompt_defaults[f"answer_hint_{str(prompt_key)}"]),
        "annotation_hint": str(prompt_defaults[f"annotation_hint_{str(prompt_key)}"]),
        "json_example": str(json_example),
        "json_example_answer_only": str(json_example_answer_only),
        "status_rule_text": str(prompt_defaults["ultimate_tictactoe_status_rule_text"]),
        "tactic_rule_text": str(prompt_defaults["ultimate_tictactoe_tactic_rule_text"]),
        "macro_threat_rule_text": str(prompt_defaults["ultimate_tictactoe_macro_threat_rule_text"]),
        **dict(dynamic_values or {}),
    }
    prompt_selection = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)
