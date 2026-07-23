"""Count triangles of a requested type on graph paper."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    GraphPaperTaskPlan,
    _build_triangle_type_count,
    run_graph_paper_entry,
)

TASK_ID = "task_geometry__graph_paper__triangle_type_count"
TRIANGLE_TYPE_QUERY_TO_CLASS = {
    "equilateral_triangle_count": "equilateral",
    "right_triangle_count": "right",
    "scalene_triangle_count": "scalene",
    "non_equilateral_isosceles_triangle_count": "non_equilateral_isosceles",
}
TRIANGLE_TYPE_QUERY_TO_TEXT = {
    "equilateral_triangle_count": "equilateral triangles",
    "right_triangle_count": "right triangles",
    "scalene_triangle_count": "scalene triangles",
    "non_equilateral_isosceles_triangle_count": "non-equilateral isosceles triangles",
}
SUPPORTED_QUERY_IDS = tuple(TRIANGLE_TYPE_QUERY_TO_CLASS)


def _build_triangle_type_count_plan() -> GraphPaperTaskPlan:
    """Bind the triangle-type count objective."""

    return GraphPaperTaskPlan(
        builder=_build_triangle_type_count,
        prompt_key="triangle_type_count",
        salt="triangle_type_count_seed",
        default_branch=SUPPORTED_QUERY_IDS[0],
        target_class_by_branch=TRIANGLE_TYPE_QUERY_TO_CLASS,
        target_text_by_branch=TRIANGLE_TYPE_QUERY_TO_TEXT,
    )


@register_task
class GeometryGraphPaperTriangleTypeCountTask:
    """Count how many rendered triangles have the requested type."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
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
            plan=_build_triangle_type_count_plan(),
        )


__all__ = ["GeometryGraphPaperTriangleTypeCountTask"]
