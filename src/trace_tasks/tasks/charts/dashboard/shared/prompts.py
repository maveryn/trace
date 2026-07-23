"""Prompt assembly for dashboard chart tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.charts.dashboard.shared.defaults import prompt_default
from trace_tasks.tasks.charts.dashboard.shared.metrics import join_quoted_labels
from trace_tasks.tasks.charts.dashboard.shared.state import SCENE_ID, DashboardDataset
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


DOMAIN = "charts"
PROMPT_BUNDLE_ID = "charts_dashboard_v1"
_DYNAMIC_SLOT_NAMES = {
    "condition_category_label",
    "object_description",
    "panel_condition_phrase",
    "rank_phrase",
    "requested_truth_phrase",
    "selected_panel_name",
    "source_panel_name",
    "target_category_label",
    "target_panel_name",
    "unanswerable_instruction",
}


def _object_description(dataset: DashboardDataset) -> str:
    return (
        f"a dashboard with {int(len(dataset.panels))} titled panels named "
        f"{join_quoted_labels([str(panel.name) for panel in dataset.panels])}. "
        f"It uses a shared category/color key with {int(len(dataset.categories))} possible category labels, "
        "and exact integer values are shown for each plotted category mark"
    )


def build_prompt_slots(
    *,
    dataset: DashboardDataset,
    extra_slots: Mapping[str, Any] | None = None,
) -> Dict[str, str]:
    slots: Dict[str, str] = {
        "object_description": _object_description(dataset),
    }
    for key, value in dataset.query.params.items():
        if str(key) in _DYNAMIC_SLOT_NAMES and isinstance(value, (str, int, float)):
            slots[str(key)] = str(value)
    if extra_slots:
        for key, value in extra_slots.items():
            if str(key) in _DYNAMIC_SLOT_NAMES:
                slots[str(key)] = str(value)
    return slots


def build_prompt_artifacts(
    *,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    rendered_prompt = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(prompt_default("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key="dashboard_mixed_chart",
        task_key="dashboard_cross_panel_query",
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts", "build_prompt_slots"]
