"""Public task for `task_charts__table__sum_absolute_differences_between_rows_over_year_interval`."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.charts.table._lifecycle import (
    build_table_temporal_two_row_plan,
    run_table_task_from_public_class,
)
from trace_tasks.tasks.charts.table.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__table__sum_absolute_differences_between_rows_over_year_interval"
PROMPT_KEY = "sum_absolute_differences_between_rows_over_year_interval"
PROGRAM_CODE = "sum(abs(value(row_a, year) - value(row_b, year)) for year in interval); output=integer_value; annotation=bbox_set(two_row_interval_cells); scene=table; scope=sum_absolute_differences_between_rows_over_year_interval"
JSON_EXAMPLE = '{"annotation":[[260,180,320,220],[322,180,382,220],[384,180,444,220],[260,236,320,276],[322,236,382,276],[384,236,444,276]],"answer":17}'
ANSWER_ONLY_EXAMPLE = '{"answer":17}'


def _prepare_yearwise_abs_difference_sum(instance_seed, params, selected_query_id):
    """Bind the year-by-year absolute-difference sum program for two selected rows."""

    del selected_query_id
    return build_table_temporal_two_row_plan(
        public_task_id=TASK_ID,
        instance_seed=int(instance_seed),
        params=params,
        operation="yearwise_abs_difference_sum",
        prompt_key=PROMPT_KEY,
        program_code=PROGRAM_CODE,
        question_format="table_sum_absolute_differences_between_rows_over_year_interval",
        json_example=JSON_EXAMPLE,
        json_example_answer_only=ANSWER_ONLY_EXAMPLE,
    )


@register_task
class ChartsTableSumAbsoluteDifferencesBetweenRowsOverYearIntervalTask:
    """Compute the sum of year-by-year absolute differences between two rows."""

    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'formula_evaluation')
    domain = DOMAIN
    objective_contract = "sum_absolute_differences_between_rows_over_year_interval"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_query_id = SINGLE_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_table_task_from_public_class(
            self,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
            build_plan=_prepare_yearwise_abs_difference_sum,
        )


__all__ = ["ChartsTableSumAbsoluteDifferencesBetweenRowsOverYearIntervalTask"]
