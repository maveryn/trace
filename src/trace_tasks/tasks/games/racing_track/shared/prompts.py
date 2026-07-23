"""Prompt assembly helpers for racing-track tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
    required_group_defaults,
)
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import SCENE_ID


_GEN_DEFAULTS_UNUSED, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
)

_PROMPT_WIRING_KEYS = (
    "bundle_id",
    "scene_key",
    "task_key",
    "json_output_contract",
    "json_output_contract_answer_only",
)


def build_racing_track_prompt_artifacts(
    *,
    domain: str,
    prompt_query_key: str,
    object_description_key: str,
    rule_text_key: str,
    instance_seed: int,
    answer_hint_key: str,
    annotation_hint_key: str,
    json_example: str,
    json_example_answer_only: str,
    dynamic_slots: Mapping[str, Any] | None = None,
) -> tuple[Dict[str, Any], Any]:
    """Build prompt artifacts from external racing-track prompt templates."""

    required_keys = (
        *_PROMPT_WIRING_KEYS,
        str(object_description_key),
        str(rule_text_key),
        str(answer_hint_key),
        str(annotation_hint_key),
    )
    prompt_defaults = required_group_defaults(
        _PROMPT_DEFAULTS,
        required_keys,
        context="racing-track prompt wiring defaults",
    )
    slots: Dict[str, Any] = {
        "object_description": str(prompt_defaults[str(object_description_key)]),
        str(rule_text_key): str(prompt_defaults[str(rule_text_key)]),
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(prompt_defaults[str(answer_hint_key)]),
        "annotation_hint": str(prompt_defaults[str(annotation_hint_key)]),
        "json_example": str(json_example),
        "json_example_answer_only": str(json_example_answer_only),
    }
    if dynamic_slots:
        slots.update(dict(dynamic_slots))
    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=slots,
        instance_seed=int(instance_seed),
    )
    return dict(prompt_defaults), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_racing_track_prompt_artifacts"]
