"""Prompt assembly for pictogram tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import prompt_bundle_id
from .state import DOMAIN, PROMPT_BUNDLE_ID, SCENE_ID, PictogramDataset


_OBJECT_DESCRIPTIONS = {
    "waffle_grid_blocks": "a repeated-mark quantity chart where each colored block represents the unit scale shown in the legend",
    "pictogram_rows": "a repeated-mark pictogram where each icon represents the unit scale shown in the legend",
}


def dynamic_slots(*, dataset: PictogramDataset, scene_variant: str) -> dict[str, Any]:
    qparams = dict(dataset.query.params)
    return {
        "object_description": str(_OBJECT_DESCRIPTIONS[str(scene_variant)]),
        "category_label": str(qparams.get("target_category_label", "")),
        "category_label_a": str(qparams.get("category_label_a", "")),
        "category_label_b": str(qparams.get("category_label_b", "")),
        "target_value": int(qparams.get("target_value", 0) or 0),
        "threshold_value": int(qparams.get("threshold_value", 0) or 0),
    }


def build_prompt_artifacts(
    *,
    prompt_task_key: str,
    prompt_query_key: str | None,
    dynamic_slot_values: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    rendered_prompt = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_bundle_id() or PROMPT_BUNDLE_ID),
        scene_key="pictogram_scene",
        task_key=str(prompt_task_key),
        query_key=(str(prompt_query_key) if prompt_query_key is not None else None),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts", "dynamic_slots"]
