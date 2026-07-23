"""Prompt helpers for phylogeny-tree graph scenes."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.graph.shared.prompting import build_graph_scene_prompt_artifacts
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts


PROMPT_BUNDLE_ID = "graph_phylogeny_tree_v1"
TREE_SCENE_PROMPT_KEY = "phylogeny_tree"
OPTIONS_SCENE_PROMPT_KEY = "phylogeny_tree_options"
TASK_PROMPT_KEY = "phylogeny_query"


def build_phylogeny_prompt_artifacts(
    *,
    domain: str,
    scene_key: str,
    prompt_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
    bundle_id: str = PROMPT_BUNDLE_ID,
) -> PromptTraceArtifacts:
    """Render prompt variants with task-owned dynamic slots."""

    return build_graph_scene_prompt_artifacts(
        domain=str(domain),
        scene_id="phylogeny_tree",
        bundle_id=str(bundle_id),
        scene_key=str(scene_key),
        task_key=TASK_PROMPT_KEY,
        query_key=str(prompt_key),
        dynamic_slots={str(key): value for key, value in dynamic_slots.items()},
        instance_seed=int(instance_seed),
    )


__all__ = [
    "OPTIONS_SCENE_PROMPT_KEY",
    "PROMPT_BUNDLE_ID",
    "TASK_PROMPT_KEY",
    "TREE_SCENE_PROMPT_KEY",
    "build_phylogeny_prompt_artifacts",
]
