"""Prompt assembly for radial Sankey chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import prompt_bundle_id
from .state import SCENE_ID, RadialSankeyDataset


DOMAIN = "charts"
PROMPT_BUNDLE_ID = "charts_radial_sankey_v1"
SCENE_PROMPT_KEY = "radial_sankey"
TASK_PROMPT_KEY = "radial_sankey_query"


def dynamic_slots(*, dataset: RadialSankeyDataset) -> dict[str, Any]:
    params = dict(dataset.question.params)
    return {
        "object_description": (
            "a radial Sankey-style flow diagram. Source nodes and target nodes are labeled around a ring, "
            "and each curved band has a printed integer flow value"
        ),
        "source_label": str(params.get("source_label", "")),
        "target_label": str(params.get("target_label", "")),
        "source_labels": str(params.get("source_labels_joined", "")),
        "target_labels": str(params.get("target_labels_joined", "")),
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
        bundle_id=str(prompt_bundle_id() or PROMPT_BUNDLE_ID),
        scene_key=SCENE_PROMPT_KEY,
        task_key=TASK_PROMPT_KEY,
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts", "dynamic_slots"]
