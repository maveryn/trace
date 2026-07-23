"""Measure the area of a graph-paper ellipse."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    GraphPaperTaskPlan,
    _build_ellipse_area_value,
    run_graph_paper_entry,
)

TASK_ID = "task_geometry__graph_paper__ellipse_area_value"
QUERY_ID = "single"
SUPPORTED_QUERY_IDS = (QUERY_ID,)


def _build_ellipse_area_value_plan() -> GraphPaperTaskPlan:
    """Bind the exact ellipse-area objective."""

    return GraphPaperTaskPlan(
        builder=_build_ellipse_area_value,
        prompt_key="ellipse_area_value",
        salt="ellipse_area_value_seed",
    )


@register_task
class GeometryGraphPaperEllipseAreaValueTask:
    """Compute exact ellipse area from graph-paper radii."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    domain = "geometry"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self, instance_seed: int, *, params: dict[str, Any], max_attempts: int
    ) -> TaskOutput:
        return run_graph_paper_entry(
            self,
            instance_seed,
            params=params,
            max_attempts=max_attempts,
            plan=_build_ellipse_area_value_plan(),
        )


__all__ = ["GeometryGraphPaperEllipseAreaValueTask"]
