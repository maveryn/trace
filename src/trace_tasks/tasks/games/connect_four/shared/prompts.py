"""Prompt assembly helpers for Connect Four games scene tasks."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
    required_group_default,
)
from trace_tasks.tasks.games.shared.prompts import build_games_prompt_artifacts, games_prompt_output_slots

from .rules import player_name
from .defaults import PROMPT_WIRING_KEYS, SCENE_ID


_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)


def json_examples_for_integer_answer() -> tuple[str, str]:
    """Return generic integer-answer JSON examples for this scene."""

    return (
        json.dumps(
            {"annotation": [[140, 220, 190, 270], [210, 220, 260, 270], [280, 220, 330, 270]], "answer": 3},
            separators=(",", ":"),
        ),
        json.dumps({"answer": 3}, separators=(",", ":")),
    )


def json_examples_for_label_answer(*, scalar_annotation: bool = False) -> tuple[str, str]:
    """Return generic label-answer JSON examples for this scene."""

    annotation = [155, 215] if bool(scalar_annotation) else [[140, 220, 190, 270], [140, 290, 190, 340]]
    return (
        json.dumps({"annotation": annotation, "answer": "C"}, separators=(",", ":")),
        json.dumps({"answer": "C"}, separators=(",", ":")),
    )


def connect_four_object_description(scene_variant: str) -> str:
    """Return prompt object description for one scene variant."""

    key = f"object_description_{str(scene_variant)}"
    return str(required_group_default(_PROMPT_DEFAULTS, key, context="Connect Four prompt defaults"))


def connect_four_rule_slots(*, current_player: int, include_safety_rule: bool = False) -> Dict[str, Any]:
    """Return common Connect Four prompt slots."""

    slots: Dict[str, Any] = {
        "current_player_name": player_name(int(current_player)),
        "legal_drop_rule_text": required_group_default(
            _PROMPT_DEFAULTS,
            "legal_drop_rule_text",
            context="Connect Four prompt defaults",
        ),
        "winning_rule_text": required_group_default(
            _PROMPT_DEFAULTS,
            "winning_rule_text",
            context="Connect Four prompt defaults",
        ),
    }
    if include_safety_rule:
        slots["opponent_player_name"] = player_name(-int(current_player))
        slots["safety_rule_text"] = required_group_default(
            _PROMPT_DEFAULTS,
            "safety_rule_text",
            context="Connect Four prompt defaults",
        )
    return slots


def connect_four_output_slots(
    *,
    prompt_query_key: str,
    json_example: str,
    json_example_answer_only: str,
) -> Dict[str, Any]:
    """Return answer/annotation prompt slots for one Connect Four query key."""

    return games_prompt_output_slots(
        prompt_defaults=_PROMPT_DEFAULTS,
        prompt_query_key=str(prompt_query_key),
        context="Connect Four prompt defaults",
        json_example=str(json_example),
        json_example_answer_only=str(json_example_answer_only),
    )


def build_connect_four_prompt_artifacts(
    *,
    domain: str,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> tuple[Dict[str, Any], Any]:
    """Build prompt artifacts from the external Connect Four prompt bundle."""

    return build_games_prompt_artifacts(
        domain=str(domain),
        scene_id=SCENE_ID,
        prompt_defaults=_PROMPT_DEFAULTS,
        prompt_wiring_keys=PROMPT_WIRING_KEYS,
        context="Connect Four prompt wiring defaults",
        prompt_query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )


__all__ = [
    "build_connect_four_prompt_artifacts",
    "connect_four_object_description",
    "connect_four_output_slots",
    "connect_four_rule_slots",
    "json_examples_for_integer_answer",
    "json_examples_for_label_answer",
]
