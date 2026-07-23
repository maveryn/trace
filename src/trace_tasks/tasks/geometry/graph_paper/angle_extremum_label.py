"""Select the largest or smallest angle drawn on graph paper."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    GraphPaperTaskPlan,
    _build_angle_extremum_label,
    run_graph_paper_entry,
)

TASK_ID = "task_geometry__graph_paper__angle_extremum_label"
SUPPORTED_QUERY_IDS = ("largest", "smallest")


def _build_angle_extremum_label_plan() -> GraphPaperTaskPlan:
    """Bind the angle-extremum objective."""

    return GraphPaperTaskPlan(
        builder=_build_angle_extremum_label,
        prompt_key="",
        salt="angle_extremum_label_seed",
        default_branch="largest",
        prompt_keys_by_branch={
            "largest": "angle_extremum_largest",
            "smallest": "angle_extremum_smallest",
        },
        role_by_branch={"largest": "max", "smallest": "min"},
    )


@register_task
class GeometryGraphPaperAngleExtremumLabelTask:
    """Choose the angle label with the requested extreme measure."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation')
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
            plan=_build_angle_extremum_label_plan(),
        )


__all__ = ["GeometryGraphPaperAngleExtremumLabelTask"]
