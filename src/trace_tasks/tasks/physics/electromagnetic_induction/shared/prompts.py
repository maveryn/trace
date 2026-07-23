"""Prompt helpers for the electromagnetic induction scene."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .state import SCENE_ID


PROMPT_BUNDLE_ID = "physics_electromagnetic_induction_v1"
SCENE_PROMPT_KEY = "electromagnetic_induction_panel_grid"


def build_induction_prompt_artifacts(
    *,
    domain: str,
    bundle_id: str,
    task_key: str,
    prompt_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    """Render v1 prompt assets for one induction count request."""

    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(bundle_id),
        scene_key=SCENE_PROMPT_KEY,
        task_key=str(task_key),
        query_key=str(prompt_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={str(key): value for key, value in dynamic_slots.items()},
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)
