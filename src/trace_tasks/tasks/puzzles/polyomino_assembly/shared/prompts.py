"""Prompt assembly for polyomino assembly puzzles."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import required_group_defaults
from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_task_prompt_variants,
)

from .state import DOMAIN, PROMPT_BUNDLE_ID, PROMPT_SCENE_KEY, SCENE_ID


def build_prompt(
    prompt_defaults: Mapping[str, Any],
    *,
    prompt_task_key: str,
    prompt_query_key: str,
    instance_seed: int,
) -> tuple[str, dict[str, str], dict[str, Any], PromptTraceArtifacts]:
    """Build prompt text and prompt-trace metadata from the scene asset."""

    prompt_values = required_group_defaults(
        prompt_defaults,
        ("bundle_id", "scene_key", "task_key"),
        context="prompt defaults for polyomino_assembly",
    )
    prompt_selection = render_task_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_values.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key=str(prompt_values.get("scene_key", PROMPT_SCENE_KEY)),
        task_key=str(prompt_values.get("task_key", prompt_task_key)),
        query_key=str(prompt_query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={},
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
            "prompt_variants_for_trace": dict(
                prompt_artifacts.prompt_variants_for_trace
            ),
        },
        prompt_artifacts,
    )


__all__ = ["build_prompt"]
