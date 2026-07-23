"""Prompt assembly for part-whole chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)

from .defaults import DOMAIN, PROMPT_DEFAULTS, SCENE_ID, chart_order_phrase


PROMPT_BUNDLE_ID = "charts_part_whole_v1"

_OBJECT_DESCRIPTIONS = {
    "pie": "one pie chart with a table of exact integer category shares",
    "donut": "one donut chart with a table of exact integer category shares",
}


def object_description(scene_variant: str) -> str:
    return str(_OBJECT_DESCRIPTIONS.get(str(scene_variant), "one part-whole chart with exact category shares"))


def dynamic_slots(dataset_extras: Mapping[str, Any], *, scene_variant: str) -> dict[str, Any]:
    return {
        "object_description": object_description(str(scene_variant)),
        "chart_order_phrase": str(dataset_extras.get("chart_order_direction", chart_order_phrase(str(scene_variant)))),
        "start_category": str(dataset_extras.get("start_category", "")),
        "end_category": str(dataset_extras.get("end_category", "")),
        "subset_category_list_text": str(dataset_extras.get("subset_category_list_text", "")),
        "target_category": str(dataset_extras.get("target_category", "")),
        "source_category": str(dataset_extras.get("source_category", "")),
        "source_order_direction": str(dataset_extras.get("source_order_direction", "")),
        "transfer_delta": int(dataset_extras.get("transfer_delta", 0) or 0),
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
        scene_key="part_whole_share_chart",
        task_key="part_whole_query",
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slot_values),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts", "dynamic_slots", "object_description"]
