"""Public task for `task_charts__table__categorical_value_count`."""

from __future__ import annotations

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.charts.table._lifecycle import (
    build_table_category_membership_count_plan,
    run_table_task_from_public_class,
)
from trace_tasks.tasks.charts.table.shared.state import DOMAIN
from trace_tasks.tasks.registry import register_task


TASK_ID = "task_charts__table__categorical_value_count"
PROMPT_KEY = "categorical_value_count"
PROGRAM_CODE = "count(row where category(column) == target_category); output=integer_count; annotation=bbox_set(matching_category_cells); scene=table; scope=categorical_value_count"
JSON_EXAMPLE = '{"annotation":[[260,180,372,236],[260,292,372,348]],"answer":2}'
ANSWER_ONLY_EXAMPLE = '{"answer":2}'


def _prepare_categorical_value_count(instance_seed, params, selected_query_id):
    """Bind category-membership counting over the categorical table column."""

    del selected_query_id
    return build_table_category_membership_count_plan(
        public_task_id=TASK_ID,
        instance_seed=int(instance_seed),
        params=params,
        prompt_key=PROMPT_KEY,
        program_code=PROGRAM_CODE,
        question_format="table_categorical_value_count",
        json_example=JSON_EXAMPLE,
        json_example_answer_only=ANSWER_ONLY_EXAMPLE,
    )


@register_task
class ChartsTableCategoricalValueCountTask:
    """Count rows whose categorical table-cell value matches a target category."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = DOMAIN
    objective_contract = "categorical_value_count"
    supported_query_ids = (SINGLE_QUERY_ID,)
    default_query_id = SINGLE_QUERY_ID
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_table_task_from_public_class(
            self,
            instance_seed=int(instance_seed),
            params=dict(params),
            max_attempts=int(max_attempts),
            build_plan=_prepare_categorical_value_count,
        )


__all__ = ["ChartsTableCategoricalValueCountTask"]
