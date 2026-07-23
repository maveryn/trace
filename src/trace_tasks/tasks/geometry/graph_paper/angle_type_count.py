"""Count angles of a requested type on graph paper."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    GraphPaperTaskPlan,
    _build_angle_type_count,
    run_graph_paper_entry,
)

TASK_ID = "task_geometry__graph_paper__angle_type_count"
ANGLE_TYPE_QUERY_TO_CLASS = {
    "acute_angle_count": "acute",
    "right_angle_count": "right",
    "obtuse_angle_count": "obtuse",
}
ANGLE_TYPE_QUERY_TO_TEXT = {
    "acute_angle_count": "acute angles",
    "right_angle_count": "right angles",
    "obtuse_angle_count": "obtuse angles",
}
SUPPORTED_QUERY_IDS = tuple(ANGLE_TYPE_QUERY_TO_CLASS)


def _build_angle_type_count_plan() -> GraphPaperTaskPlan:
    """Bind the angle-type count objective."""

    return GraphPaperTaskPlan(
        builder=_build_angle_type_count,
        prompt_key="angle_type_count",
        salt="angle_type_count_seed",
        default_branch=SUPPORTED_QUERY_IDS[0],
        target_class_by_branch=ANGLE_TYPE_QUERY_TO_CLASS,
        target_text_by_branch=ANGLE_TYPE_QUERY_TO_TEXT,
    )


@register_task
class GeometryGraphPaperAngleTypeCountTask:
    """Count how many rendered angles have the requested type."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'formula_evaluation')
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
            plan=_build_angle_type_count_plan(),
        )


__all__ = ["GeometryGraphPaperAngleTypeCountTask"]
