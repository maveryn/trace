"""Shared prompt assembly helpers for graph scene packages."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PROMPT_OUTPUT_MODES,
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


def build_graph_scene_prompt_artifacts(
    *,
    domain: str,
    scene_id: str,
    bundle_id: str,
    scene_key: str,
    task_key: str,
    query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    """Render prompt variants for one graph scene/query."""

    prompt_selection = render_scene_prompt_variants(
        domain=str(domain),
        scene_id=str(scene_id),
        bundle_id=str(bundle_id),
        scene_key=str(scene_key),
        task_key=str(task_key),
        query_key=str(query_key),
        answer_or_annotation_keys=PROMPT_OUTPUT_MODES,
        dynamic_slots={str(key): value for key, value in dynamic_slots.items()},
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(prompt_selection)


__all__ = ["build_graph_scene_prompt_artifacts"]
