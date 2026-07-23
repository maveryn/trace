"""Prompt assembly helpers for refraction-layers tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import SCENE_ID


PROMPT_BUNDLE_ID = "physics_refraction_layers_v1"
SCENE_PROMPT_KEY = "refraction_layers_diagram"


def build_refraction_prompt_artifacts(
    *,
    domain: str,
    bundle_id: str,
    task_key: str,
    query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    """Build prompt artifacts from v1 refraction prompt assets."""

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
