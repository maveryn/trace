"""Step-list task for reading a step title at a relative offset from a named step."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task

from . import _lifecycle


TASK_ID = "task_pages__step_list__relative_offset_step_label"
OFFSET_AFTER_QUERY_ID = "offset_after_named_step"
OFFSET_BEFORE_QUERY_ID = "offset_before_named_step"
SUPPORTED_QUERY_IDS = (OFFSET_AFTER_QUERY_ID, OFFSET_BEFORE_QUERY_ID)

_LOOKUP_MODE_BY_QUERY_ID = {
    OFFSET_AFTER_QUERY_ID: _lifecycle.OFFSET_AFTER_TITLE_MODE,
    OFFSET_BEFORE_QUERY_ID: _lifecycle.OFFSET_BEFORE_TITLE_MODE,
}


@register_task
class PagesStepListRelativeOffsetStepLabelTask:
    """Return the visible title reached by moving a few steps before or after a named source step."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'topology')
    domain = _lifecycle.DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int):
        del max_attempts
        selected_branch, branch_probabilities, task_params = _lifecycle.select_public_branch(
            instance_seed=int(instance_seed),
            params=params,
            supported=SUPPORTED_QUERY_IDS,
            default=OFFSET_AFTER_QUERY_ID,
            public_task=TASK_ID,
        )
        query_id = str(selected_branch)
        return _lifecycle.build_step_list_response(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=query_id,
            branch_probabilities=branch_probabilities,
            lookup_mode=_LOOKUP_MODE_BY_QUERY_ID[query_id],
            source_query_id=query_id,
            prompt_query_key=query_id,
            question_format="step_list_relative_offset_step_label",
        )


__all__ = [
    "OFFSET_AFTER_QUERY_ID",
    "OFFSET_BEFORE_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesStepListRelativeOffsetStepLabelTask",
]
