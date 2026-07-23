"""Prompt assembly helpers for graph automaton scene tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.graph.shared.prompting import build_graph_scene_prompt_artifacts

from .state import SCENE_ID


PROMPT_BUNDLE_ID = "automaton_v1"
SCENE_PROMPT_KEY = "automaton"


def build_automaton_prompt_artifacts(
    *,
    domain: str,
    bundle_id: str,
    task_key: str,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> Any:
    """Build prompt artifacts from task-owned query key and dynamic slots."""

    return build_graph_scene_prompt_artifacts(
        domain=str(domain),
        scene_id=SCENE_ID,
        bundle_id=str(bundle_id),
        scene_key=SCENE_PROMPT_KEY,
        task_key=str(task_key),
        query_key=str(prompt_query_key),
        dynamic_slots={str(key): value for key, value in dynamic_slots.items()},
        instance_seed=int(instance_seed),
    )


__all__ = [
    "PROMPT_BUNDLE_ID",
    "build_automaton_prompt_artifacts",
]
