"""Prompt assembly helpers for simplified darts scene tasks."""

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


def darts_integer_json_examples() -> tuple[str, str]:
    """Return generic JSON examples for integer-answer darts tasks."""

    return (
        json.dumps(
            {"annotation": [[397, 205, 425, 233], [511, 350, 539, 378]], "answer": 2},
            separators=(",", ":"),
        ),
        json.dumps({"answer": 2}, separators=(",", ":")),
    )


def darts_single_point_json_examples() -> tuple[str, str]:
    """Return generic JSON examples for single-dart score tasks."""

    return (
        json.dumps({"annotation": [411, 219], "answer": 17}, separators=(",", ":")),
        json.dumps({"answer": 17}, separators=(",", ":")),
    )


def darts_option_letter_json_examples() -> tuple[str, str]:
    """Return generic JSON examples for option-letter darts tasks."""

    return (
        json.dumps({"annotation": [411, 219], "answer": "B"}, separators=(",", ":")),
        json.dumps({"answer": "B"}, separators=(",", ":")),
    )


def darts_output_slots(
    *,
    prompt_query_key: str,
    json_example: str,
    json_example_answer_only: str,
    extra_slots: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Return answer and annotation prompt slots for one darts query key."""

    slots = games_prompt_output_slots(
        prompt_defaults=_PROMPT_DEFAULTS,
        prompt_query_key=str(prompt_query_key),
        context="darts prompt defaults",
        json_example=str(json_example),
        json_example_answer_only=str(json_example_answer_only),
    )
    slots.update(dict(extra_slots or {}))
    return slots


def build_darts_prompt_artifacts(
    *,
    domain: str,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> tuple[Dict[str, Any], Any]:
    """Build prompt artifacts from the external darts prompt bundle."""

    return build_games_prompt_artifacts(
        domain=str(domain),
        scene_id=SCENE_ID,
        prompt_defaults=_PROMPT_DEFAULTS,
        prompt_wiring_keys=PROMPT_WIRING_KEYS,
        context="darts prompt wiring defaults",
        prompt_query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )


__all__ = [
    "build_darts_prompt_artifacts",
    "darts_integer_json_examples",
    "darts_option_letter_json_examples",
    "darts_output_slots",
    "darts_single_point_json_examples",
]
