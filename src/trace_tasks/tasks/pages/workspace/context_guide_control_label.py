"""Workspace task identifying a target control from a context guide cue and Key column."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.registry import register_task

from . import _lifecycle


TASK_ID = "task_pages__workspace__context_guide_control_label"
OBJECTIVE_KEY = _lifecycle.PROMPT_CONTEXT_GUIDE_LABEL_KEY
SUPPORTED_QUERY_IDS = (SINGLE_QUERY_ID,)


@register_task
class PagesWorkspaceContextGuideControlLabelTask:
    """Identify the control matching one context guide cue and a directly stated Key column."""

    task_id = TASK_ID
    reasoning_operations = ('matching',)
    domain = "pages"
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int):
        del max_attempts
        task_params = dict(params)
        task_params.setdefault("context_count_min", 3)
        task_params.setdefault("context_count_max", 3)
        return _lifecycle.build_workspace_response(
            instance_seed=int(instance_seed),
            params=task_params,
            task_id=TASK_ID,
            objective_key=OBJECTIVE_KEY,
            question_format="workspace_context_guide_control_label",
        )


__all__ = [
    "OBJECTIVE_KEY",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesWorkspaceContextGuideControlLabelTask",
]
