"""Prompt assembly helpers for Crossing games scene tasks."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
    required_group_default,
)
from trace_tasks.tasks.games.shared.prompts import build_games_prompt_artifacts, games_prompt_output_slots

from .defaults import PROMPT_WIRING_KEYS, SCENE_ID


_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)


def json_examples_for_integer_answer() -> tuple[str, str]:
    """Return generic integer-answer JSON examples for this scene."""

    return (
        json.dumps(
            {"annotation": [[190, 285, 258, 330], [520, 406, 588, 451]], "answer": 2},
            separators=(",", ":"),
        ),
        json.dumps({"answer": 2}, separators=(",", ":")),
    )


def json_examples_for_label_answer() -> tuple[str, str]:
    """Return generic label-answer JSON examples for Crossing option tasks."""

    return (
        json.dumps({"annotation": [463, 363], "answer": "C"}, separators=(",", ":")),
        json.dumps({"answer": "C"}, separators=(",", ":")),
    )


def crossing_object_description(*, include_route: bool) -> str:
    """Return prompt object description for one crossing scene type."""

    key = "object_description_traffic_crossing" if bool(include_route) else "object_description_traffic_crossing_no_route"
    return str(required_group_default(_PROMPT_DEFAULTS, key, context="Crossing prompt defaults"))


def crossing_motion_rule_text() -> str:
    """Return the shared route-motion rule text."""

    return str(required_group_default(_PROMPT_DEFAULTS, "crossing_motion_rule_text", context="Crossing prompt defaults"))


def crossing_exit_motion_rule_text() -> str:
    """Return the no-route vehicle exit rule text."""

    return str(required_group_default(_PROMPT_DEFAULTS, "crossing_exit_motion_rule_text", context="Crossing prompt defaults"))


def crossing_output_slots(
    *,
    prompt_query_key: str,
    json_example: str,
    json_example_answer_only: str,
) -> Dict[str, Any]:
    """Return answer/annotation prompt slots for one Crossing query key."""

    return games_prompt_output_slots(
        prompt_defaults=_PROMPT_DEFAULTS,
        prompt_query_key=str(prompt_query_key),
        context="Crossing prompt defaults",
        json_example=str(json_example),
        json_example_answer_only=str(json_example_answer_only),
    )


def build_crossing_prompt_artifacts(
    *,
    domain: str,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> tuple[Dict[str, Any], Any]:
    """Build prompt artifacts from the external Crossing prompt bundle."""

    return build_games_prompt_artifacts(
        domain=str(domain),
        scene_id=SCENE_ID,
        prompt_defaults=_PROMPT_DEFAULTS,
        prompt_wiring_keys=PROMPT_WIRING_KEYS,
        context="Crossing prompt wiring defaults",
        prompt_query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )


__all__ = [
    "build_crossing_prompt_artifacts",
    "crossing_exit_motion_rule_text",
    "crossing_motion_rule_text",
    "crossing_object_description",
    "crossing_output_slots",
    "json_examples_for_integer_answer",
    "json_examples_for_label_answer",
]
