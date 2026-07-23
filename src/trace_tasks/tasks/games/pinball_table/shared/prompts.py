"""Prompt assembly helpers for pinball-table tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults, required_group_defaults
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
    "object_description_schematic_table",
    "object_description_scoreable_count",
    "pinball_motion_rule_text",
    "pinball_scoreable_rule_text",
)


def build_pinball_prompt_artifacts(
    *,
    domain: str,
    prompt_query_key: str,
    scene_variant: str,
    instance_seed: int,
    object_description_key: str,
    answer_hint: str,
    annotation_hint: str,
    json_example: str,
    json_example_answer_only: str,
    dynamic_slots: Mapping[str, Any] | None = None,
) -> tuple[Dict[str, Any], Any]:
    """Build prompt artifacts from external templates and task-owned slots."""

    prompt_defaults = required_group_defaults(
        _PROMPT_DEFAULTS,
        _PROMPT_WIRING_KEYS,
        context="pinball prompt wiring defaults",
    )
    slots: Dict[str, Any] = {
        "object_description": str(prompt_defaults[str(object_description_key)]),
        "pinball_motion_rule_text": str(prompt_defaults["pinball_motion_rule_text"]),
        "pinball_scoreable_rule_text": str(prompt_defaults["pinball_scoreable_rule_text"]),
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(answer_hint),
        "annotation_hint": str(annotation_hint),
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


__all__ = ["build_pinball_prompt_artifacts"]
