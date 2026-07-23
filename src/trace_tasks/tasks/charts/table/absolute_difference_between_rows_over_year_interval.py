"""Public task for `task_charts__table__absolute_difference_between_rows_over_year_interval`."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.charts.table._lifecycle import (
    build_table_temporal_two_row_plan,
    run_table_task_from_public_class,
)
from trace_tasks.tasks.charts.table.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__table__absolute_difference_between_rows_over_year_interval"
PROMPT_KEY = "absolute_difference_between_rows_over_year_interval"
PROGRAM_CODE = "abs(sum(values(row_a, years)) - sum(values(row_b, years))); output=integer_value; annotation=bbox_map(row_interval_span_by_row_label); scene=table; scope=absolute_difference_between_rows_over_year_interval"
JSON_EXAMPLE = '{"annotation":{"Aster":[260,180,444,220],"Beacon":[260,236,444,276]},"answer":7}'
ANSWER_ONLY_EXAMPLE = '{"answer":7}'


def _prepare_interval_sum_difference(instance_seed, params, selected_query_id):
    """Bind the interval-sum comparison program for two selected table rows."""

    del selected_query_id
    return build_table_temporal_two_row_plan(
        public_task_id=TASK_ID,
        instance_seed=int(instance_seed),
        params=params,
        operation="row_interval_sum_difference_abs",
        prompt_key=PROMPT_KEY,
        program_code=PROGRAM_CODE,
        question_format="table_absolute_difference_between_rows_over_year_interval",
        json_example=JSON_EXAMPLE,
        json_example_answer_only=ANSWER_ONLY_EXAMPLE,
    )


@register_task
class ChartsTableAbsoluteDifferenceBetweenRowsOverYearIntervalTask:
    """Compute the absolute difference between two row interval sums."""

    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "absolute_difference_between_rows_over_year_interval"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_query_id = SINGLE_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_table_task_from_public_class(
            self,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
            build_plan=_prepare_interval_sum_difference,
        )


__all__ = ["ChartsTableAbsoluteDifferenceBetweenRowsOverYearIntervalTask"]
