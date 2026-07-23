"""Public task for `task_charts__error_interval__interval_width_rank_label`."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.charts.error_interval._lifecycle import ErrorIntervalTaskPlan, build_width_rank_plan, run_error_interval_plan
from trace_tasks.tasks.charts.error_interval.shared.defaults import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__error_interval__interval_width_rank_label"
OBJECTIVE_CONTRACT = "interval_width_rank_label"
WIDEST_QUERY_ID = "widest_interval_label"
NARROWEST_QUERY_ID = "narrowest_interval_label"
SECOND_WIDEST_QUERY_ID = "second_widest_interval_label"
SECOND_NARROWEST_QUERY_ID = "second_narrowest_interval_label"
QUERY_TO_RANK = {
    WIDEST_QUERY_ID: ("widest", "widest interval"),
    NARROWEST_QUERY_ID: ("narrowest", "narrowest interval"),
    SECOND_WIDEST_QUERY_ID: ("second_widest", "second widest interval"),
    SECOND_NARROWEST_QUERY_ID: ("second_narrowest", "second narrowest interval"),
}


@register_task
class ChartsErrorIntervalRelationLabelTask:
    """Identify a category by ranked interval width."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = OBJECTIVE_CONTRACT
    supported_query_ids = (
        WIDEST_QUERY_ID,
        NARROWEST_QUERY_ID,
        SECOND_WIDEST_QUERY_ID,
        SECOND_NARROWEST_QUERY_ID,
    )
    default_dataset_enabled = True

    def _build_plan(self, instance_seed: int, *, params: dict[str, Any], selected_query_id: str) -> ErrorIntervalTaskPlan:
        """Bind the ranked-width label objective for the public task."""

        rank_key, relation_phrase = QUERY_TO_RANK[str(selected_query_id)]
        return build_width_rank_plan(
            params=params,
            instance_seed=int(instance_seed),
            selected_query_id=str(selected_query_id),
            prompt_key=str(selected_query_id),
            rank_key=str(rank_key),
            relation_phrase=str(relation_phrase),
        )

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_error_interval_plan(
            task_id=self.task_id,
            supported_query_ids=self.supported_query_ids,
            default_query_id=WIDEST_QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            build_plan=self._build_plan,
        )


__all__ = ["ChartsErrorIntervalRelationLabelTask"]
