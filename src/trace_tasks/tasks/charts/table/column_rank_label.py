"""Public task for `task_charts__table__column_rank_label`."""

from __future__ import annotations

from trace_tasks.tasks.charts.table._lifecycle import (
    build_table_rank_label_plan,
    run_table_task_from_public_class,
)
from trace_tasks.tasks.charts.table.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__table__column_rank_label"
SUPPORTED_QUERY_IDS = ("highest_rank_in_column", "lowest_rank_in_column")
PROGRAM_CODE = "rank(rows by value(column), direction, rank_k); output=row_label; annotation=bbox(answer_value_cell); scene=table; scope=column_rank_label"
QUERY_SPECS = {
    "highest_rank_in_column": ("descending", "highest"),
    "lowest_rank_in_column": ("ascending", "lowest"),
}
JSON_EXAMPLE = '{"annotation":[260,180,372,236],"answer":"Ava"}'
ANSWER_ONLY_EXAMPLE = '{"answer":"Ava"}'


def _prepare_column_rank_label(instance_seed, params, selected_query_id):
    """Bind rank direction locally for the table row-label ranking objective."""

    operation, rank_direction = QUERY_SPECS[str(selected_query_id)]
    return build_table_rank_label_plan(
        public_task_id=TASK_ID,
        instance_seed=int(instance_seed),
        params=params,
        operation=operation,
        prompt_key=str(selected_query_id),
        rank_direction=rank_direction,
        program_code=PROGRAM_CODE,
        json_example=JSON_EXAMPLE,
        json_example_answer_only=ANSWER_ONLY_EXAMPLE,
    )


@register_task
class ChartsTableKthRankInColumnLabelTask:
    """Return the row label at a selected rank in one numeric table column."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = DOMAIN
    objective_contract = "column_rank_label"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_query_id = SUPPORTED_QUERY_IDS[0]
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_table_task_from_public_class(
            self,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
            build_plan=_prepare_column_rank_label,
        )


__all__ = ["ChartsTableKthRankInColumnLabelTask"]
