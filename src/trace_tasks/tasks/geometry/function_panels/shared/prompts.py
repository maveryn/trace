"""Prompt helpers for function-panel scene packages."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import SCENE_ID


def render_panel_prompt_artifacts(
    *,
    prompt_defaults: Mapping[str, Any],
    prompt_key: str,
    target_range: str = "",
    target_interval: str = "",
    instance_seed: int,
):
    """Render v1 prompt variants for one selected-panel objective."""

    dynamic_slots = {}
    if str(target_range):
        dynamic_slots["target_range"] = str(target_range)
    if str(target_interval):
        dynamic_slots["target_interval"] = str(target_interval)
    selection = render_scene_prompt_variants(
        domain="geometry",
        scene_id=SCENE_ID,
        bundle_id=str(prompt_defaults["bundle_id"]),
        scene_key=str(prompt_defaults["scene_key"]),
        task_key=str(prompt_defaults["task_key"]),
        query_key=str(prompt_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(selection)
