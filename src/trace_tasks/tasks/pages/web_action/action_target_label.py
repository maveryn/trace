"""Web-action task selecting the target control for an action instruction."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from . import _lifecycle


TASK_ID = "task_pages__web_action__action_target_label"
SUPPORTED_QUERY_IDS = _lifecycle.SUPPORTED_ACTION_TARGET_QUERY_IDS
DEFAULT_QUERY_ID = SUPPORTED_QUERY_IDS[0]


@register_task
class PagesWebActionActionTargetLabelTask:
    """Identify the candidate-labeled web control described by the visible instruction."""

    task_id = TASK_ID
    reasoning_operations = ('matching',)
    domain = "pages"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int):
        del max_attempts
        selected_branch, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=DEFAULT_QUERY_ID,
            task_id=TASK_ID,
            namespace=f"{TASK_ID}.query",
        )
        return _lifecycle.build_web_action_response(
            instance_seed=int(instance_seed),
            params=task_params,
            task_id=TASK_ID,
            prompt_query_key=str(selected_branch),
            public_query_id=str(selected_branch),
            query_id_probabilities=dict(branch_probabilities),
            control_family_key=str(selected_branch),
            answer_mode="selection",
            question_format="web_action_action_target_label",
        )


__all__ = [
    "DEFAULT_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesWebActionActionTargetLabelTask",
]
