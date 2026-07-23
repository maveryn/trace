"""Public task for `task_charts__table__filtered_column_mean`."""

from __future__ import annotations

from trace_tasks.tasks.charts.table._lifecycle import (
    build_table_filtered_mean_plan,
    run_table_task_from_public_class,
)
from trace_tasks.tasks.charts.table.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__table__filtered_column_mean"
SUPPORTED_QUERY_IDS = (
    "above_threshold_filtered_mean",
    "below_threshold_filtered_mean",
    "interval_filtered_mean",
)
PROGRAM_CODE = "mean(value(target_column) for row where filter_column satisfies predicate); output=integer_value; annotation=bbox_set_map(filter_cells,target_cells); scene=table; scope=filtered_column_mean"
QUERY_FILTERS = {
    "above_threshold_filtered_mean": "above_threshold",
    "below_threshold_filtered_mean": "below_threshold",
    "interval_filtered_mean": "in_interval",
}
JSON_EXAMPLE = '{"annotation":{"filter_cells":[[260,180,372,236],[260,236,372,292]],"target_cells":[[374,180,486,236],[374,236,486,292]]},"answer":14}'
ANSWER_ONLY_EXAMPLE = '{"answer":14}'


def _prepare_filtered_column_mean(instance_seed, params, selected_query_id):
    """Bind the row-filter predicate for a filtered target-column mean."""

    return build_table_filtered_mean_plan(
        public_task_id=TASK_ID,
        instance_seed=int(instance_seed),
        params=params,
        filter_variant=QUERY_FILTERS[str(selected_query_id)],
        prompt_key=str(selected_query_id),
        program_code=PROGRAM_CODE,
        json_example=JSON_EXAMPLE,
        json_example_answer_only=ANSWER_ONLY_EXAMPLE,
    )


@register_task
class ChartsTableFilteredColumnMeanTask:
    """Compute the mean of one target column over rows selected by another column."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'comparison', 'aggregation')
    domain = DOMAIN
    objective_contract = "filtered_column_mean"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SUPPORTED_QUERY_IDS[0]
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_table_task_from_public_class(
            self,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
            build_plan=_prepare_filtered_column_mean,
        )


__all__ = ["ChartsTableFilteredColumnMeanTask"]
