"""Public task for `task_charts__error_interval__reference_exclusion_side_count`."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.error_interval._lifecycle import ErrorIntervalTaskPlan, build_reference_count_plan, run_error_interval_plan
from trace_tasks.tasks.charts.error_interval.shared.defaults import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__error_interval__reference_exclusion_side_count"
OBJECTIVE_CONTRACT = "reference_exclusion_side_count"
ABOVE_QUERY_ID = "entirely_above_reference_count"
BELOW_QUERY_ID = "entirely_below_reference_count"
QUERY_TO_PREDICATE = {
    ABOVE_QUERY_ID: "above",
    BELOW_QUERY_ID: "below",
}


@register_task
class ChartsErrorIntervalReferenceExclusionSideCountTask:
    """Count intervals entirely above or below a displayed reference value."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = (ABOVE_QUERY_ID, BELOW_QUERY_ID)
    default_dataset_enabled = True

    def _build_plan(self, instance_seed: int, *, params: dict[str, Any], selected_query_id: str) -> ErrorIntervalTaskPlan:
        """Bind one reference-exclusion side count for the public task."""

        return build_reference_count_plan(
            params=params,
            instance_seed=int(instance_seed),
            selected_query_id=str(selected_query_id),
            prompt_key=str(selected_query_id),
            predicate=str(QUERY_TO_PREDICATE[str(selected_query_id)]),
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_error_interval_plan(
            task_id=self.task_id,
            supported_query_ids=self.supported_query_ids,
            default_query_id=ABOVE_QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            build_plan=self._build_plan,
        )


__all__ = ["ChartsErrorIntervalReferenceExclusionSideCountTask"]
