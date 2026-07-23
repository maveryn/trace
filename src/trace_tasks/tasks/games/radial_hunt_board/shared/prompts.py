"""Prompt assembly helpers for radial hunt board tasks."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import SCENE_ID


_PROMPT_WIRING_KEYS = (
    "bundle_id",
    "scene_key",
    "task_key",
    "json_output_contract",
    "json_output_contract_answer_only",
)


def radial_hunt_json_examples() -> Tuple[str, str]:
    """Return format-only examples with answer and point-set cardinality aligned."""

    annotation = [[180.0, 220.0], [268.0, 220.0]]
    return (
        json.dumps({"annotation": annotation, "answer": 2}, separators=(",", ":"), ensure_ascii=True),
        json.dumps({"answer": 2}, separators=(",", ":"), ensure_ascii=True),
    )


def build_radial_hunt_prompt_artifacts(
    *,
    domain: str,
    scene_variant: str,
    prompt_query_key: str,
    rule_slot_name: str,
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
) -> tuple[Dict[str, Any], Any]:
    """Build prompt artifacts from external radial hunt board prompt templates."""

    object_description_key = f"object_description_{str(scene_variant)}"
    answer_hint_key = f"answer_hint_{str(prompt_query_key)}"
    annotation_hint_key = f"annotation_hint_{str(prompt_query_key)}"
    required_keys = (
        *_PROMPT_WIRING_KEYS,
        object_description_key,
        str(rule_slot_name),
        answer_hint_key,
        annotation_hint_key,
    )
    defaults = required_group_defaults(
        prompt_defaults,
        required_keys,
        context="radial hunt board prompt wiring defaults",
    )
    json_example, json_example_answer_only = radial_hunt_json_examples()
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(defaults["bundle_id"]),
        scene_key=str(defaults["scene_key"]),
        task_key=str(defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(defaults[object_description_key]),
            str(rule_slot_name): str(defaults[str(rule_slot_name)]),
            "json_output_contract": str(defaults["json_output_contract"]),
            "json_output_contract_answer_only": str(defaults["json_output_contract_answer_only"]),
            "answer_hint": str(defaults[answer_hint_key]),
            "annotation_hint": str(defaults[annotation_hint_key]),
            "json_example": str(json_example),
            "json_example_answer_only": str(json_example_answer_only),
        },
        instance_seed=int(instance_seed),
    )
    return dict(defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_radial_hunt_prompt_artifacts", "radial_hunt_json_examples"]
