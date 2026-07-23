"""Prompt rendering for spinner tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ....shared.config_defaults import required_group_defaults
from ....shared.prompt_variants import PROMPT_OUTPUT_MODES, build_prompt_trace_artifacts, render_scene_prompt_variants

from .defaults import SCENE_ID


def render_spinner_prompt(
    *,
    prompt_query_key: str,
    scene_variant: str,
    prompt_defaults: Mapping[str, Any],
    instance_seed: int,
    event_description: str,
) -> Tuple[str, Dict[str, str], Dict[str, Any]]:
    """Render the externalized spinner prompt bundle."""

    raw_task_key = str(prompt_defaults.get("task_key", ""))
    object_description_key = f"object_description_{raw_task_key}_{scene_variant}"
    prompt_values = required_group_defaults(
        prompt_defaults,
        (
            "bundle_id",
            "scene_key",
            "task_key",
            object_description_key,
        ),
        context="prompt defaults for spinner probability",
    )
    prompt_selection = render_scene_prompt_variants(
        domain="symbolic",
        scene_id=SCENE_ID,
        bundle_id=str(prompt_values["bundle_id"]),
        scene_key=str(prompt_values["scene_key"]),
        task_key=str(prompt_values["task_key"]),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={
            "object_description": str(prompt_values[object_description_key]),
            "event_description": str(event_description),
        },
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
    return str(prompt_artifacts.prompt), dict(prompt_artifacts.prompt_variants), {
        "prompt_variant": dict(prompt_artifacts.prompt_variant),
        "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
        "prompt_variants_for_trace": dict(prompt_artifacts.prompt_variants_for_trace),
        "bundle_id": str(prompt_values["bundle_id"]),
    }


__all__ = ["render_spinner_prompt"]
