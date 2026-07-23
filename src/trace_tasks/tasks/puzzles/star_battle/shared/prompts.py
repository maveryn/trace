"""Prompt asset helpers for Star Battle puzzle tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)

from .state import DOMAIN, SCENE_ID


def build_star_battle_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_task_key: str,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any] | None = None,
    instance_seed: int,
) -> tuple[Dict[str, Any], Any]:
    """Render Star Battle prompt variants from the scene prompt bundle."""

    prompt_values = required_group_defaults(
        prompt_defaults,
        ("bundle_id", "scene_key"),
        context=f"prompt defaults for {prompt_task_key}",
    )
    prompt_selection = render_task_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_values["bundle_id"]),
        scene_key=str(prompt_values["scene_key"]),
        task_key=str(prompt_task_key),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dict(dynamic_slots or {}),
        instance_seed=int(instance_seed),
    )
    return dict(prompt_values), build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_star_battle_prompt_artifacts"]
