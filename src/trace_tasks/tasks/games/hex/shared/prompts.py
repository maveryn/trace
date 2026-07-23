"""Prompt asset rendering for Hex tasks."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import SCENE_ID


@dataclass(frozen=True)
class HexPromptContext:
    """Dynamic slots needed by the Hex prompt asset."""

    scene_variant: str
    prompt_query_key: str
    query_player: str
    neighbor_state: str = ""


def build_hex_prompt_json_examples(*, answer_type: str) -> Tuple[str, str]:
    """Return deterministic format examples for Hex JSON output."""

    if str(answer_type) == "string":
        answer_value: str | int = "C"
        annotation_value = [210, 250]
    else:
        answer_value = 3
        annotation_value = [[150, 220], [210, 250], [270, 280]]
    return (
        json.dumps({"annotation": annotation_value, "answer": answer_value}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": answer_value}, separators=(",", ":"), ensure_ascii=False),
    )


def build_hex_prompt_artifacts(
    *,
    domain: str,
    prompt_defaults: Mapping[str, Any],
    context: HexPromptContext,
    answer_type: str,
    instance_seed: int,
) -> tuple[dict[str, Any], PromptTraceArtifacts]:
    """Render one Hex prompt from external prompt assets."""

    defaults = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "object_description_open_board",
            "object_description_crowded_board",
            "hex_rule_text",
            "red_goal_text",
            "blue_goal_text",
            f"answer_hint_{str(context.prompt_query_key)}",
            f"annotation_hint_{str(context.prompt_query_key)}",
        ),
        context=f"prompt defaults for {SCENE_ID}",
    )
    json_example, json_example_answer_only = build_hex_prompt_json_examples(answer_type=str(answer_type))
    answer_hint = str(defaults[f"answer_hint_{str(context.prompt_query_key)}"]).format(
        query_player=str(context.query_player),
        neighbor_state=str(context.neighbor_state),
    )
    annotation_hint = str(defaults[f"annotation_hint_{str(context.prompt_query_key)}"]).format(
        query_player=str(context.query_player),
        neighbor_state=str(context.neighbor_state),
    )
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(defaults["task_key"]),
        query_key=str(context.prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(defaults[f"object_description_{str(context.scene_variant)}"]),
            "json_output_contract": str(defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(defaults["json_output_contract_answer_only"]),
            "answer_hint": str(answer_hint),
            "annotation_hint": str(annotation_hint),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
            "hex_rule_text": str(defaults["hex_rule_text"]),
            "red_goal_text": str(defaults["red_goal_text"]),
            "blue_goal_text": str(defaults["blue_goal_text"]),
            "query_player": str(context.query_player),
            "query_player_lower": str(context.query_player).lower(),
            "neighbor_state": str(context.neighbor_state),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = [
    "HexPromptContext",
    "build_hex_prompt_artifacts",
    "build_hex_prompt_json_examples",
]
