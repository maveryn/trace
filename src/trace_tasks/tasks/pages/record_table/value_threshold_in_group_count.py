"""Record-table task for counting thresholded values in one section."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from . import _lifecycle
from .shared.defaults import (
    DEFAULT_ANSWER_COUNT_SUPPORT,
    DEFAULT_ROW_COUNT_SUPPORT,
    DOMAIN,
    SECTION_SIZE_THRESHOLD_FILTER,
)


TASK_ID = "task_pages__record_table__value_threshold_in_group_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "value_threshold_in_group_count"
FILTER_KEY = SECTION_SIZE_THRESHOLD_FILTER
ANSWER_SUPPORT = DEFAULT_ANSWER_COUNT_SUPPORT
ROW_COUNT_SUPPORT = DEFAULT_ROW_COUNT_SUPPORT


def _bind_value_threshold(
    selected_branch: str,
    branch_probabilities: Mapping[str, float],
    case,
    rendered,
):
    """Bind the section and threshold predicate to prompt slots and annotation."""

    prompt_binding = _lifecycle.RecordTablePromptBinding(
        prompt_branch_key=PROMPT_QUERY_KEY,
        dynamic_slots={
            "section_name": str(case.target_section_name),
            "size_threshold_mb": str(case.size_threshold_mb),
        },
    )
    answer_binding = _lifecycle.integer_binding(
        annotation_value=_lifecycle.counted_row_annotation(case, rendered),
        selected_branch=str(selected_branch),
        branch_probabilities=branch_probabilities,
        answer_value=int(case.answer_value),
        target_payload={
            "section_name": str(case.target_section_name),
            "size_threshold_mb": int(case.size_threshold_mb),
            "relation": ">=",
        },
        question_format="record_table_value_threshold_in_group_count",
    )
    return prompt_binding, answer_binding


@register_task
class PagesRecordTableValueThresholdInGroupCountTask:
    """Count table rows in a section whose visible size crosses a threshold."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison')
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
        prompt_binding, answer_binding = _bind_value_threshold(
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
    "PagesRecordTableValueThresholdInGroupCountTask",
]
