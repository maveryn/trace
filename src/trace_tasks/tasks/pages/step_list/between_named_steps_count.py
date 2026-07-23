"""Step-list task for counting steps between two named boundary steps."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from . import _lifecycle


TASK_ID = "task_pages__step_list__between_named_steps_count"
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)
PROMPT_QUERY_KEY = "between_named_steps_count"
SOURCE_QUERY_ID = "between_named_steps_count"


@register_task
class PagesStepListBetweenNamedStepsCountTask:
    """Return the number of numbered steps strictly between two named step titles."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology')
    domain = _lifecycle.DOMAIN
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
        return _lifecycle.build_step_list_response(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            lookup_mode=_lifecycle.BETWEEN_NAMED_STEPS_COUNT_MODE,
            source_query_id=SOURCE_QUERY_ID,
            prompt_query_key=PROMPT_QUERY_KEY,
            question_format="step_list_between_named_steps_count",
        )


__all__ = [
    "PROMPT_QUERY_KEY",
    "SOURCE_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesStepListBetweenNamedStepsCountTask",
]
