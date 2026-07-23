"""Compute average speed over a marked distance-time graph interval."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    AverageSpeedLifecyclePlan,
    run_average_speed_lifecycle,
)


TASK_ID = "task_physics__motion_graph__average_speed_value"
TASK_NAMESPACE = "physics_motion_graph_average_speed_value"
TASK_PROMPT_KEY = "motion_graph_average_speed_value_query"
PROMPT_BRANCH_KEY = "average_speed_value"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)


def _build_lifecycle_plan() -> AverageSpeedLifecyclePlan:
    """Bind this public task contract to the scene-private lifecycle."""

    return AverageSpeedLifecyclePlan(
        task_identifier=TASK_ID,
        namespace=TASK_NAMESPACE,
        prompt_key=TASK_PROMPT_KEY,
        prompt_branch_key=PROMPT_BRANCH_KEY,
        public_branch_ids=SUPPORTED_QUERY_IDS,
        default_branch_id=SINGLE_QUERY_ID,
    )


@register_task
class PhysicsMotionGraphAverageSpeedValueTask:
    """Compute average speed over a marked distance-time graph interval."""

    domain = "physics"
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate a distance-time graph and bind the average-speed answer."""

        return run_average_speed_lifecycle(
            domain=self.domain,
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            max_attempts=int(max_attempts),
            plan=_build_lifecycle_plan(),
        )


__all__ = ["PhysicsMotionGraphAverageSpeedValueTask", "TASK_ID"]
