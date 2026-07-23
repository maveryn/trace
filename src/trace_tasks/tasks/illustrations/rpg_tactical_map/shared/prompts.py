"""Prompt rendering helpers for RPG tactical map tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


def rpg_tactical_map_terrain_rules_text() -> str:
    """Return the public terrain-cost rule sentence shared by tactical map tasks."""

    return "Grass, roads, and bridges cost 1 movement point; forests cost 2; mountains cost 3; water cannot be entered. Moves are only up, down, left, or right."


def build_rpg_tactical_map_prompt_artifacts(
    *,
    domain: str,
    scene_id: str,
    prompt_defaults: Mapping[str, Any],
    prompt_query_key: str,
    slots: Mapping[str, Any],
    instance_seed: int,
) -> Any:
    """Render external prompt templates for one tactical map task."""

    selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=str(scene_id),
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_query_key),
        slots=dict(slots),
        instance_seed=int(instance_seed),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        preferred_mode="answer_and_annotation",
    )
    return build_prompt_trace_artifacts(selection)


def build_rpg_tactical_map_task_prompt_with_default_slots(
    *,
    domain: str,
    scene_id: str,
    prompt_defaults_source: Mapping[str, Any],
    prompt_query_key: str,
    answer_hint_key: str,
    annotation_hint_key: str,
    json_example_key: str,
    json_example_answer_only_key: str,
    context_label: str,
    slots: Mapping[str, Any],
    instance_seed: int,
) -> tuple[Mapping[str, Any], Any]:
    """Resolve prompt defaults and render common JSON-output prompt slots."""

    prompt_defaults = required_group_defaults(
        prompt_defaults_source,
        [
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            str(answer_hint_key),
            str(annotation_hint_key),
            str(json_example_key),
            str(json_example_answer_only_key),
        ],
        context=f"prompt defaults for {context_label}",
    )
    rendered_slots = {
        "json_output_contract": str(prompt_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(prompt_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(prompt_defaults[str(answer_hint_key)]),
        "annotation_hint": str(prompt_defaults[str(annotation_hint_key)]),
        "json_example": str(prompt_defaults[str(json_example_key)]),
        "json_example_answer_only": str(prompt_defaults[str(json_example_answer_only_key)]),
    }
    rendered_slots.update(dict(slots))
    artifacts = build_rpg_tactical_map_prompt_artifacts(
        domain=str(domain),
        scene_id=str(scene_id),
        prompt_defaults=prompt_defaults,
        prompt_query_key=str(prompt_query_key),
        slots=rendered_slots,
        instance_seed=int(instance_seed),
    )
    return prompt_defaults, artifacts


__all__ = [
    "build_rpg_tactical_map_prompt_artifacts",
    "build_rpg_tactical_map_task_prompt_with_default_slots",
    "rpg_tactical_map_terrain_rules_text",
]
