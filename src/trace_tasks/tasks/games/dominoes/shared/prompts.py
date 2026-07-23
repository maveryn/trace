"""Prompt assembly helpers for dominoes scene tasks."""

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


def domino_integer_json_examples(answer_value: int = 2) -> tuple[str, str]:
    """Return generic integer-answer JSON examples for domino count tasks."""

    answer = int(answer_value)
    example_boxes = [
        [248, 318, 386, 394],
        [408, 318, 546, 394],
        [568, 318, 706, 394],
        [728, 318, 866, 394],
    ][: max(0, int(answer))]
    return (
        json.dumps({"annotation": example_boxes, "answer": answer}, separators=(",", ":")),
        json.dumps({"answer": answer}, separators=(",", ":")),
    )


def domino_option_label_segment_json_examples(answer_value: str = "C") -> tuple[str, str]:
    """Return generic option-label JSON examples for scalar segment annotation tasks."""

    return (
        json.dumps({"annotation": [[312, 142], [326, 142]], "answer": str(answer_value)}, separators=(",", ":")),
        json.dumps({"answer": str(answer_value)}, separators=(",", ":")),
    )


def domino_output_slots(
    *,
    prompt_query_key: str,
    json_example: str,
    json_example_answer_only: str,
) -> Dict[str, Any]:
    """Return answer and annotation prompt slots for one dominoes query key."""

    return games_prompt_output_slots(
        prompt_defaults=_PROMPT_DEFAULTS,
        prompt_query_key=str(prompt_query_key),
        context="dominoes prompt defaults",
        json_example=str(json_example),
        json_example_answer_only=str(json_example_answer_only),
    )


def domino_object_description(*, has_chain: bool, has_reference: bool, has_candidates: bool, scene_variant: str) -> str:
    """Return prompt object text for the sampled domino layout."""

    if bool(has_chain) and not bool(has_candidates):
        key = "object_description_chain_only"
    elif bool(has_chain):
        key = f"object_description_chain_{str(scene_variant)}"
    elif bool(has_reference):
        key = f"object_description_tableau_reference_{str(scene_variant)}"
    else:
        key = f"object_description_tableau_{str(scene_variant)}"
    return str(
        required_group_default(
            _PROMPT_DEFAULTS,
            key,
            context="dominoes prompt object-description defaults",
        )
    )


def build_domino_prompt_artifacts(
    *,
    domain: str,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> tuple[Dict[str, Any], Any]:
    """Build prompt artifacts from the external dominoes prompt bundle."""

    return build_games_prompt_artifacts(
        domain=str(domain),
        scene_id=SCENE_ID,
        prompt_defaults=_PROMPT_DEFAULTS,
        prompt_wiring_keys=PROMPT_WIRING_KEYS,
        context="dominoes prompt wiring defaults",
        prompt_query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )


__all__ = [
    "build_domino_prompt_artifacts",
    "domino_integer_json_examples",
    "domino_option_label_segment_json_examples",
    "domino_object_description",
    "domino_output_slots",
]
