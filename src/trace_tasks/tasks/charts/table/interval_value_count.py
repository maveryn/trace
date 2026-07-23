"""Public task for `task_charts__table__interval_value_count`."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.charts.table._lifecycle import (
    build_table_numeric_filter_count_plan,
    run_table_task_from_public_class,
)
from trace_tasks.tasks.charts.table.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__table__interval_value_count"
PROMPT_KEY = "interval_value_count"
PROGRAM_CODE = "count(row where lower <= value(column) <= upper); output=integer_count; annotation=bbox_set(matching_cells); scene=table; scope=interval_value_count"
JSON_EXAMPLE = '{"annotation":[[260,180,372,236],[260,236,372,292],[260,292,372,348]],"answer":3}'
ANSWER_ONLY_EXAMPLE = '{"answer":3}'


def _prepare_interval_value_count(instance_seed, params, selected_query_id):
    """Bind the inclusive-interval predicate for this single-query count objective."""

    del selected_query_id
    return build_table_numeric_filter_count_plan(
        public_task_id=TASK_ID,
        instance_seed=int(instance_seed),
        params=params,
        operation="in_interval",
        prompt_key=PROMPT_KEY,
        program_code=PROGRAM_CODE,
        question_format="table_interval_value_count",
        json_example=JSON_EXAMPLE,
        json_example_answer_only=ANSWER_ONLY_EXAMPLE,
    )


@register_task
class ChartsTableIntervalValueCountTask:
    """Count rows whose value in one table column lies inside an inclusive interval."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = "interval_value_count"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_query_id = SINGLE_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_table_task_from_public_class(
            self,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
            build_plan=_prepare_interval_value_count,
        )


__all__ = ["ChartsTableIntervalValueCountTask"]
