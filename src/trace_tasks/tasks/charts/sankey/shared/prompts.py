"""Prompt assembly for standard Sankey chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import PROMPT_DEFAULTS
from .state import PROMPT_BUNDLE_ID, SCENE_ID, SankeyDataset


DOMAIN = "charts"


def dynamic_slots(*, dataset: SankeyDataset) -> dict[str, Any]:
    question = dataset.question
    return {
        "object_description": (
            "a three-column Sankey-style flow diagram. Each node has a visible label, each directed band "
            "has a printed integer value, and bands run from a source node through one middle node to a target node"
        ),
        "source_label": str(question.params.get("source_label", "")),
        "middle_label": str(question.params.get("middle_label", "")),
        "target_label": str(question.params.get("target_label", "")),
    }


def build_prompt_artifacts(
    *,
    prompt_query_key: str,
    dynamic_slot_values: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    rendered_prompt = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key="sankey",
        task_key="sankey_path_query",
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts", "dynamic_slots"]
