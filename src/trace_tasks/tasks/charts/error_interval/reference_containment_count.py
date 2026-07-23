"""Public task for `task_charts__error_interval__reference_containment_count`."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.error_interval._lifecycle import ErrorIntervalTaskPlan, build_reference_count_plan, run_error_interval_plan
from trace_tasks.tasks.charts.error_interval.shared.defaults import DOMAIN
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID


TASK_ID = "task_charts__error_interval__reference_containment_count"
OBJECTIVE_CONTRACT = "reference_containment_count"
PROMPT_KEY = "contains_reference_count"
REFERENCE_PREDICATE = "contains"


@register_task
class ChartsErrorIntervalReferenceContainmentCountTask:
    """Count intervals that contain a displayed reference value."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = (DEFAULT_QUERY_ID,)
    default_dataset_enabled = True

    def _build_plan(self, instance_seed: int, *, params: dict[str, Any], selected_query_id: str) -> ErrorIntervalTaskPlan:
        """Bind the containment-count objective for the public task."""

        return build_reference_count_plan(
            params=params,
            instance_seed=int(instance_seed),
            selected_query_id=str(selected_query_id),
            prompt_key=PROMPT_KEY,
            predicate=REFERENCE_PREDICATE,
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_error_interval_plan(
            task_id=self.task_id,
            supported_query_ids=self.supported_query_ids,
            default_query_id=DEFAULT_QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            build_plan=self._build_plan,
        )


__all__ = ["ChartsErrorIntervalReferenceContainmentCountTask"]
