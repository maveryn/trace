"""Prompt helpers for graph option-panel scenes."""

from __future__ import annotations

from trace_tasks.tasks.graph.shared.prompting import build_graph_scene_prompt_artifacts
from trace_tasks.tasks.shared.prompt_variants import PromptTraceArtifacts
from .state import SCENE_ID


PROMPT_BUNDLE_ID = "graph_options_v1"
SCENE_PROMPT_KEY = "graph_options"
TASK_PROMPT_KEY = "structure_match_label_query"


def build_graph_options_prompt_artifacts(
    *,
    domain: str,
    bundle_id: str,
    prompt_key: str,
    object_description: str,
    instance_seed: int,
) -> PromptTraceArtifacts:
    """Render prompt variants for one graph-option objective."""

    return build_graph_scene_prompt_artifacts(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(bundle_id),
        scene_key=SCENE_PROMPT_KEY,
        task_key=TASK_PROMPT_KEY,
        query_key=str(prompt_key),
        dynamic_slots={"object_description": str(object_description)},
        instance_seed=int(instance_seed),
    )


__all__ = [
    "PROMPT_BUNDLE_ID",
    "SCENE_PROMPT_KEY",
    "TASK_PROMPT_KEY",
    "build_graph_options_prompt_artifacts",
]
