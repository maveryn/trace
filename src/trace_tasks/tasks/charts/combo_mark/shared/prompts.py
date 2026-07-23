"""Prompt assembly for combo-mark chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.combo_mark.shared.defaults import (
    DOMAIN,
    PROMPT_BUNDLE_ID,
    PROMPT_DEFAULTS,
    SCENE_ID,
)
from trace_tasks.tasks.charts.combo_mark.shared.state import ComboDataset
from trace_tasks.tasks.shared.prompt_variants import (
    PromptTraceArtifacts,
    build_prompt_trace_artifacts,
    render_scene_prompt_variants,
)


SCENE_PROMPT_KEY_BY_VARIANT = {
    "bar_line_shared_axis": "bar_line_shared_axis",
    "bar_line_dual_axis": "bar_line_dual_axis",
    "stacked_bar_line": "stacked_bar_line",
    "area_line_overlay": "area_line_overlay",
}
TASK_PROMPT_KEY = "combo_mark_query"


def combo_series_slots(dataset: ComboDataset, **extra_slots: Any) -> dict[str, Any]:
    """Return quoted combo-series names plus any task-owned dynamic slots."""

    return {
        "primary_name": f'"{dataset.primary_name}"',
        "line_name": f'"{dataset.line_name}"',
        **dict(extra_slots),
    }


def build_prompt_artifacts(
    *,
    scene_variant: str,
    prompt_query_key: str,
    dynamic_slots: Mapping[str, Any],
    instance_seed: int,
) -> PromptTraceArtifacts:
    rendered_prompt = render_scene_prompt_variants(
        domain=DOMAIN,
        scene_id=SCENE_ID,
        bundle_id=str(PROMPT_DEFAULTS.get("bundle_id", PROMPT_BUNDLE_ID)),
        scene_key=SCENE_PROMPT_KEY_BY_VARIANT.get(str(scene_variant), "bar_line_shared_axis"),
        task_key=TASK_PROMPT_KEY,
        query_key=str(prompt_query_key),
        dynamic_slots=dict(dynamic_slots),
        instance_seed=int(instance_seed),
    )
    return build_prompt_trace_artifacts(rendered_prompt)


__all__ = ["build_prompt_artifacts", "combo_series_slots"]
