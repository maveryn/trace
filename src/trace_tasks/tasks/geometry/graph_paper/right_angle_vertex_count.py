"""Count right-angle vertices in one polygon drawn on graph paper."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    GraphPaperTaskPlan,
    _build_right_angle_vertex_count,
    run_graph_paper_entry,
)

TASK_ID = "task_geometry__graph_paper__right_angle_vertex_count"
QUERY_ID = "single"
SUPPORTED_QUERY_IDS = (QUERY_ID,)


def _build_right_angle_vertex_count_plan() -> GraphPaperTaskPlan:
    """Bind the single-polygon right-angle vertex count objective."""

    return GraphPaperTaskPlan(
        builder=_build_right_angle_vertex_count,
        prompt_key="right_angle_vertex_count",
        salt="right_angle_vertex_count_seed",
    )


@register_task
class GeometryGraphPaperRightAngleVertexCountTask:
    """Count vertices where adjacent polygon sides meet at a right angle."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'formula_evaluation')
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
            plan=_build_right_angle_vertex_count_plan(),
        )


__all__ = ["GeometryGraphPaperRightAngleVertexCountTask"]
