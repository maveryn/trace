"""Count marked polar-graph points with a prompted coordinate value."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    PolarCoordinateValuePointCountPlan,
    run_polar_coordinate_value_point_count,
)
from .shared.state import ReadoutComponent

TASK_ID = "task_geometry__polar_graph_paper__coordinate_value_point_count"
SCENE_ID = "polar_graph_paper"
QUERY_IDS = ("radius_value_point_count", "angle_value_point_count")

QUERY_COMPONENTS: dict[str, ReadoutComponent] = {
    "radius_value_point_count": "radius",
    "angle_value_point_count": "angle_degrees",
}


def _prepare_coordinate_value_point_count_plan() -> PolarCoordinateValuePointCountPlan:
    """Bind radius/angle value-count semantics to the polar graph lifecycle."""

    return PolarCoordinateValuePointCountPlan(
        public_identifier=TASK_ID,
        query_ids=QUERY_IDS,
        component_by_query_id=QUERY_COMPONENTS,
    )


@register_task
class PolarGraphPaperCoordinateValuePointCountTask:
    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison', 'formula_evaluation')
    domain = "geometry"
    default_dataset_enabled = True
    supported_query_ids = QUERY_IDS

    def generate(
        self,
        instance_seed: int,
        *,
        params: dict[str, Any],
        max_attempts: int = 1,
    ) -> TaskOutput:
        """Generate one polar coordinate-value point-count sample."""

        return run_polar_coordinate_value_point_count(
            plan=_prepare_coordinate_value_point_count_plan(),
            instance_seed=instance_seed,
            params=params,
            max_attempts=max_attempts,
        )


__all__ = [
    "PolarGraphPaperCoordinateValuePointCountTask",
    "QUERY_IDS",
    "SCENE_ID",
    "TASK_ID",
]
