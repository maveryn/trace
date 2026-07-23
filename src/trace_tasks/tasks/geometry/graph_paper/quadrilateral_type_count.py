"""Count quadrilaterals of a requested type on graph paper."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    GraphPaperTaskPlan,
    _build_quadrilateral_type_count,
    run_graph_paper_entry,
)

TASK_ID = "task_geometry__graph_paper__quadrilateral_type_count"
QUADRILATERAL_TYPE_QUERY_TO_CLASS = {
    "square_count": "square",
    "non_square_rectangle_count": "non_square_rectangle",
    "non_square_rhombus_count": "non_square_rhombus",
    "slanted_parallelogram_count": "slanted_parallelogram",
}
QUADRILATERAL_TYPE_QUERY_TO_TEXT = {
    "square_count": "squares",
    "non_square_rectangle_count": "non-square rectangles",
    "non_square_rhombus_count": "non-square rhombuses",
    "slanted_parallelogram_count": "slanted parallelograms",
}
SUPPORTED_QUERY_IDS = tuple(QUADRILATERAL_TYPE_QUERY_TO_CLASS)


def _build_quadrilateral_type_count_plan() -> GraphPaperTaskPlan:
    """Bind the quadrilateral-type count objective."""

    return GraphPaperTaskPlan(
        builder=_build_quadrilateral_type_count,
        prompt_key="quadrilateral_type_count",
        salt="quadrilateral_type_count_seed",
        default_branch=SUPPORTED_QUERY_IDS[0],
        target_class_by_branch=QUADRILATERAL_TYPE_QUERY_TO_CLASS,
        target_text_by_branch=QUADRILATERAL_TYPE_QUERY_TO_TEXT,
    )


@register_task
class GeometryGraphPaperQuadrilateralTypeCountTask:
    """Count how many rendered quadrilaterals have the requested type."""

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
            plan=_build_quadrilateral_type_count_plan(),
        )


__all__ = ["GeometryGraphPaperQuadrilateralTypeCountTask"]
