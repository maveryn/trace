"""Prompt helpers for thermal-mixing tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import SCENE_ID, SCENE_PROMPT_KEY


PROMPT_BUNDLE_ID = "physics_thermal_mixing_v1"


def build_thermal_mixing_prompt_artifacts(
    *,
    domain: str,
    bundle_id: str,
    task_key: str,
    query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    """Build prompt artifacts from the thermal-mixing prompt bundle."""

    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(bundle_id),
        scene_key=SCENE_PROMPT_KEY,
        task_key=str(task_key),
        query_key=str(query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={str(key): value for key, value in dynamic_slots.items()},
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


__all__ = ["PROMPT_BUNDLE_ID", "build_thermal_mixing_prompt_artifacts"]
