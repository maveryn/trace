"""Prompt assembly for radial-progress tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import prompt_bundle_id
from .state import SCENE_ID, ProgressDataset


DOMAIN = "charts"
PROMPT_BUNDLE_ID = "charts_radial_progress_v1"
SCENE_PROMPT_KEY = "radial_progress_scene"
COUNT_TASK_PROMPT_KEY = "radial_progress_condition_count_query"
EXTREMUM_TASK_PROMPT_KEY = "radial_progress_remaining_extremum_query"

OBJECT_DESCRIPTION_BY_VARIANT = {
    "full_progress_rings": "a grid of labeled circular progress rings. Each ring shows completion from 0 to 100 percent",
    "semicircle_gauges": "a grid of labeled semicircle progress gauges. Each gauge shows completion from 0 to 100 percent",
    "segmented_radial_bars": "a grid of labeled segmented radial progress bars. Filled segments show completion from 0 to 100 percent",
}


def dynamic_slots(*, dataset: ProgressDataset) -> dict[str, Any]:
    return {
        "object_description": str(OBJECT_DESCRIPTION_BY_VARIANT[str(dataset.scene_variant)]),
        "threshold_phrase": str(dataset.question.params.get("threshold_phrase", "")),
        "range_phrase": str(dataset.question.params.get("range_phrase", "")),
        "extremum_phrase": str(dataset.question.params.get("extremum_phrase", "")),
    }


def build_prompt_artifacts(
    *,
    prompt_query_key: str,
    is_label_answer: bool,
    dynamic_slot_values: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    rendered_prompt = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_bundle_id() or PROMPT_BUNDLE_ID),
        scene_key=SCENE_PROMPT_KEY,
        task_key=EXTREMUM_TASK_PROMPT_KEY if bool(is_label_answer) else COUNT_TASK_PROMPT_KEY,
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts", "dynamic_slots"]
