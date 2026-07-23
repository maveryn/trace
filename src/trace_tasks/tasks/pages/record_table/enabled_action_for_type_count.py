"""Record-table task for counting enabled actions by row type."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.defaults import (
    DOMAIN,
    ENABLED_ACTION_ANSWER_COUNT_SUPPORT,
    ENABLED_ACTION_ROW_COUNT_SUPPORT,
    ENABLED_TYPE_ACTION_FILTER,
)


TASK_ID = "task_pages__record_table__enabled_action_for_type_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "enabled_action_for_type_count"
FILTER_KEY = ENABLED_TYPE_ACTION_FILTER
ANSWER_SUPPORT = ENABLED_ACTION_ANSWER_COUNT_SUPPORT
ROW_COUNT_SUPPORT = ENABLED_ACTION_ROW_COUNT_SUPPORT


def _bind_enabled_action(
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    case,
    rendered,
):
    """Bind the target type/action predicate to prompt slots and annotation."""

    prompt_binding = _lifecycle.RecordTablePromptBinding(
        prompt_branch_key=PROMPT_QUERY_KEY,
        dynamic_slots={
            "type_label": str(case.target_type),
            "action_label": str(case.target_action_label),
        },
    )
    answer_binding = _lifecycle.integer_binding(
        annotation_value=_lifecycle.counted_row_annotation(case, rendered),
        selected_branch=str(selected_branch),
        branch_probabilities=branch_probabilities,
        answer_value=int(case.answer_value),
        target_payload={
            "type_label": str(case.target_type),
            "action_label": str(case.target_action_label),
            "action_enabled": True,
        },
        question_format="record_table_enabled_action_for_type_count",
    )
    return prompt_binding, answer_binding


@register_task
class PagesRecordTableEnabledActionForTypeCountTask:
    """Count rows of one type whose visible action is enabled."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int):
        del max_attempts
        selected_branch, branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=SINGLE_QUERY_ID,
            public_task=TASK_ID,
        )
        case, rendered = _lifecycle.build_case_and_render(
            instance_seed=int(instance_seed),
            params=task_params,
            filter_key=FILTER_KEY,
            row_count_support=ROW_COUNT_SUPPORT,
            answer_count_support=ANSWER_SUPPORT,
        )
        prompt_binding, answer_binding = _bind_enabled_action(
            str(selected_branch),
            branch_probabilities,
            case,
            rendered,
        )
        return _lifecycle.build_record_table_response(
            instance_seed=int(instance_seed),
            case=case,
            rendered=rendered,
            prompt_binding=prompt_binding,
            answer_binding=answer_binding,
        )


__all__ = [
    "ANSWER_SUPPORT",
    "FILTER_KEY",
    "PROMPT_QUERY_KEY",
    "ROW_COUNT_SUPPORT",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesRecordTableEnabledActionForTypeCountTask",
]
