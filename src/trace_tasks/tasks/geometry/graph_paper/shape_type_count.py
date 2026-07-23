"""Count shapes of a requested type on graph paper."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    GraphPaperTaskPlan,
    _build_shape_type_count,
    run_graph_paper_entry,
)

TASK_ID = "task_geometry__graph_paper__shape_type_count"
SHAPE_TYPE_QUERY_TO_CLASS = {
    "triangle_count": "triangle",
    "quadrilateral_count": "quadrilateral",
    "pentagon_count": "pentagon",
    "hexagon_count": "hexagon",
    "circle_count": "circle",
    "ellipse_count": "ellipse",
}
SHAPE_TYPE_QUERY_TO_TEXT = {
    "triangle_count": "triangles",
    "quadrilateral_count": "quadrilaterals",
    "pentagon_count": "pentagons",
    "hexagon_count": "hexagons",
    "circle_count": "circles",
    "ellipse_count": "non-circular ellipses",
}
SUPPORTED_QUERY_IDS = tuple(SHAPE_TYPE_QUERY_TO_CLASS)


def _build_shape_type_count_plan() -> GraphPaperTaskPlan:
    """Bind the mixed-shape count objective."""

    return GraphPaperTaskPlan(
        builder=_build_shape_type_count,
        prompt_key="shape_type_count",
        salt="shape_type_count_seed",
        default_branch=SUPPORTED_QUERY_IDS[0],
        target_class_by_branch=SHAPE_TYPE_QUERY_TO_CLASS,
        target_text_by_branch=SHAPE_TYPE_QUERY_TO_TEXT,
    )


@register_task
class GeometryGraphPaperShapeTypeCountTask:
    """Count how many rendered objects have the requested shape type."""

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
            plan=_build_shape_type_count_plan(),
        )


__all__ = ["GeometryGraphPaperShapeTypeCountTask"]
