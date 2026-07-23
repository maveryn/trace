"""Public task for `task_charts__table__threshold_count`."""

from __future__ import annotations

from trace_tasks.tasks.charts.table._lifecycle import (
    build_table_numeric_filter_count_plan,
    run_table_task_from_public_class,
)
from trace_tasks.tasks.charts.table.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__table__threshold_count"
SUPPORTED_QUERY_IDS = ("above_threshold_count", "below_threshold_count")
PROGRAM_CODE = "count(row where value(column) threshold_compare threshold); output=integer_count; annotation=bbox_set(matching_cells); scene=table; scope=threshold_count"
QUERY_OPERATIONS = {
    "above_threshold_count": "above_threshold",
    "below_threshold_count": "below_threshold",
}
JSON_EXAMPLE = '{"annotation":[[260,180,372,236],[260,236,372,292]],"answer":2}'
ANSWER_ONLY_EXAMPLE = '{"answer":2}'


def _prepare_threshold_count(instance_seed, params, selected_query_id):
    """Bind threshold direction locally, then let neutral table lifecycle assemble the output."""

    return build_table_numeric_filter_count_plan(
        public_task_id=TASK_ID,
        instance_seed=int(instance_seed),
        params=params,
        operation=QUERY_OPERATIONS[str(selected_query_id)],
        prompt_key=str(selected_query_id),
        program_code=PROGRAM_CODE,
        question_format="table_threshold_count",
        json_example=JSON_EXAMPLE,
        json_example_answer_only=ANSWER_ONLY_EXAMPLE,
    )


@register_task
class ChartsTableThresholdCountTask:
    """Count rows whose value in one table column is above or below a threshold."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
    domain = DOMAIN
    objective_contract = "threshold_count"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SUPPORTED_QUERY_IDS[0]
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_table_task_from_public_class(
            self,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
            build_plan=_prepare_threshold_count,
        )


__all__ = ["ChartsTableThresholdCountTask"]
