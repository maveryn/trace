"""Choose the speed-change state from a marked velocity-time graph."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import StateChoiceLifecyclePlan, run_state_choice_lifecycle
from .shared.sampling import SPEED_MAGNITUDE_PROFILE
from .shared.state import SPEED_CHANGE_STATES


TASK_ID = "task_physics__motion_graph__speed_change_state_choice"
TASK_NAMESPACE = "physics_motion_graph_speed_change_state_choice"
TASK_PROMPT_KEY = "motion_graph_speed_change_state_choice_query"
SPEED_CHANGE_OPERATION = "speed_change_state_choice"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (SINGLE_QUERY_ID,)


def _build_speed_change_plan() -> StateChoiceLifecyclePlan:
    """Bind the public speed-change objective to the shared graph lifecycle."""

    return StateChoiceLifecyclePlan(
        task_identifier=TASK_ID,
        namespace=TASK_NAMESPACE,
        prompt_key=TASK_PROMPT_KEY,
        public_branch_ids=SUPPORTED_QUERY_IDS,
        default_branch_id=SINGLE_QUERY_ID,
        operation_label=SPEED_CHANGE_OPERATION,
        prompt_branch_key=SPEED_CHANGE_OPERATION,
        state_profile=SPEED_MAGNITUDE_PROFILE,
        state_support=SPEED_CHANGE_STATES,
        graph_kind="velocity_time",
        y_axis_label="v (m/s)",
        title="velocity-time graph",
        error_label="speed-change",
    )


@register_task
class PhysicsMotionGraphSpeedChangeStateChoiceTask:
    """Choose whether the marked velocity-time segment speeds up or slows down."""

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
        """Generate a velocity-time graph and bind the speed-change answer."""

        return run_state_choice_lifecycle(
            domain=self.domain,
            instance_seed=int(instance_seed),
            params=dict(params or {}),
            max_attempts=int(max_attempts),
            plan=_build_speed_change_plan(),
        )


__all__ = ["PhysicsMotionGraphSpeedChangeStateChoiceTask", "TASK_ID"]
