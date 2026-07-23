"""Public task for `task_charts__table__column_summary_value`."""

from __future__ import annotations

from trace_tasks.tasks.charts.table._lifecycle import (
    build_table_column_summary_plan,
    run_table_task_from_public_class,
)
from trace_tasks.tasks.charts.table.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__table__column_summary_value"
SUPPORTED_QUERY_IDS = ("column_sum", "column_mean", "column_median")
PROGRAM_CODE = "aggregate(values(column), operation=sum|mean|median); output=integer_value; annotation=bbox(column_values); scene=table; scope=column_summary_value"
QUERY_OPERATIONS = {"column_sum": "sum", "column_mean": "mean", "column_median": "median"}
JSON_EXAMPLES = {
    "column_sum": ('{"annotation":[260,180,372,520],"answer":84}', '{"answer":84}'),
    "column_mean": ('{"annotation":[260,180,372,520],"answer":14}', '{"answer":14}'),
    "column_median": ('{"annotation":[260,180,372,520],"answer":13}', '{"answer":13}'),
}


def _prepare_column_summary_value(instance_seed, params, selected_query_id):
    """Bind the aggregate operator for one full-column numeric summary."""

    json_example, answer_only_example = JSON_EXAMPLES[str(selected_query_id)]
    return build_table_column_summary_plan(
        public_task_id=TASK_ID,
        instance_seed=int(instance_seed),
        params=params,
        operation=QUERY_OPERATIONS[str(selected_query_id)],
        prompt_key=str(selected_query_id),
        program_code=PROGRAM_CODE,
        json_example=json_example,
        json_example_answer_only=answer_only_example,
    )


@register_task
class ChartsTableColumnSummaryValueTask:
    """Compute a sum, mean, or median over one numeric table column."""

    task_id = TASK_ID
    reasoning_operations = ('aggregation',)
    domain = DOMAIN
    objective_contract = "column_summary_value"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SUPPORTED_QUERY_IDS[0]
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_table_task_from_public_class(
            self,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
            build_plan=_prepare_column_summary_value,
        )


__all__ = ["ChartsTableColumnSummaryValueTask"]
