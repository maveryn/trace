"""Prompt assembly for pipe-flow repair puzzles."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)

from .state import (
    PROMPT_BUNDLE_ID,
    PROMPT_SCENE_KEY,
    PROMPT_TASK_KEY,
    SCENE_ID,
)


def build_prompt(
    prompt_defaults: Mapping[str, Any],
    *,
    scene_variant: str,
    prompt_query_key: str,
    instance_seed: int,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Build prompt variants from the pipe-flow prompt asset."""

    required_keys = (
        "bundle_id",
        "scene_key",
        "task_key",
        f"object_description_{scene_variant}",
    )
    prompt_values = required_group_defaults(
        prompt_defaults,
        required_keys,
        context="prompt defaults for pipe_flow",
    )
    dynamic_slots = {
        "object_description": str(prompt_values[f"object_description_{scene_variant}"]),
    }
    prompt_selection = render_task_prompt_variants(
        domain="puzzles",
        scene_id=SCENE_ID,
        bundle_id=str(prompt_values.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key=str(prompt_values.get("scene_key", PROMPT_SCENE_KEY)),
        task_key=str(prompt_values.get("task_key", PROMPT_TASK_KEY)),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots=dynamic_slots,
        instance_seed=int(instance_seed),
    )
    prompt_artifacts = build_prompt_trace_artifacts(prompt_selection)
    return (
        str(prompt_artifacts.prompt),
        dict(prompt_artifacts.prompt_variants),
        {
            "bundle_id": str(prompt_values.get("bundle_id", PROMPT_BUNDLE_ID)),
            "prompt_variant": dict(prompt_artifacts.prompt_variant),
            "prompt_variant_active_key": str(prompt_artifacts.prompt_variant_active_key),
            "prompt_variants_for_trace": dict(prompt_artifacts.prompt_variants_for_trace),
        },
    )
