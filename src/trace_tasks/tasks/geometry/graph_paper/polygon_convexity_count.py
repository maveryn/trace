"""Count convex or concave polygons on graph paper."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    GraphPaperTaskPlan,
    _build_polygon_convexity_count,
    run_graph_paper_entry,
)

TASK_ID = "task_geometry__graph_paper__polygon_convexity_count"
POLYGON_CONVEXITY_QUERY_TO_CLASS = {
    "convex_polygon_count": "convex",
    "concave_polygon_count": "concave",
}
POLYGON_CONVEXITY_QUERY_TO_TEXT = {
    "convex_polygon_count": "convex polygons",
    "concave_polygon_count": "concave polygons",
}
SUPPORTED_QUERY_IDS = tuple(POLYGON_CONVEXITY_QUERY_TO_CLASS)


def _build_polygon_convexity_count_plan() -> GraphPaperTaskPlan:
    """Bind the polygon-convexity count objective."""

    return GraphPaperTaskPlan(
        builder=_build_polygon_convexity_count,
        prompt_key="polygon_convexity_count",
        salt="polygon_convexity_count_seed",
        default_branch=SUPPORTED_QUERY_IDS[0],
        target_class_by_branch=POLYGON_CONVEXITY_QUERY_TO_CLASS,
        target_text_by_branch=POLYGON_CONVEXITY_QUERY_TO_TEXT,
    )


@register_task
class GeometryGraphPaperPolygonConvexityCountTask:
    """Count polygons with the requested convexity."""

    task_id = TASK_ID
    reasoning_operations = ('counting',)
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
            plan=_build_polygon_convexity_count_plan(),
        )


__all__ = ["GeometryGraphPaperPolygonConvexityCountTask"]
