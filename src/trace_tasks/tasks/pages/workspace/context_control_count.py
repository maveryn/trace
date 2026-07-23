"""Workspace task counting controls in one context row by visible state."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from . import _lifecycle


TASK_ID = "task_pages__workspace__context_control_count"
OBJECTIVE_KEY = _lifecycle.PROMPT_CONTEXT_COUNT_KEY
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)


@register_task
class PagesWorkspaceContextControlCountTask:
    """Count controls with a requested visible state in one workspace context row."""

    task_id = TASK_ID
    reasoning_operations = ('counting',)
    domain = "pages"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int):
        del max_attempts
        return _lifecycle.build_workspace_response(
            instance_seed=int(instance_seed),
            params=params,
            task_id=TASK_ID,
            objective_key=OBJECTIVE_KEY,
            question_format="workspace_context_control_count",
        )


__all__ = [
    "OBJECTIVE_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesWorkspaceContextControlCountTask",
]
