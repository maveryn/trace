"""Timeline task for identifying an event at a relative timeline offset."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from . import _lifecycle


TASK_ID = "task_pages__timeline__relative_position_event_label"
BEFORE_QUERY_ID = "event_before_dated_event_label"
AFTER_QUERY_ID = "event_after_dated_event_label"
SUPPORTED_QUERY_IDS = (BEFORE_QUERY_ID, AFTER_QUERY_ID)


def _relative_relation_for_branch(selected_branch: str) -> str:
    """Map one task-owned query id to its timeline direction."""

    if str(selected_branch) == BEFORE_QUERY_ID:
        return "before"
    if str(selected_branch) == AFTER_QUERY_ID:
        return "after"
    raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_branch}")


@register_task
class PagesTimelineRelativePositionEventLabelTask:
    """Identify the event label at a fixed offset from a dated event card."""

    task_id = TASK_ID
    reasoning_operations = ('ranking',)
    domain = _lifecycle.DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int):
        del max_attempts
        selected_branch, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=AFTER_QUERY_ID,
            task_id=TASK_ID,
        )
        relative_relation = _relative_relation_for_branch(str(selected_branch))
        return _lifecycle.build_timeline_response(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            program_mode=_lifecycle.RELATIVE_POSITION_EVENT_LABEL_MODE,
            interval_relation=str(relative_relation),
            prompt_query_key=str(selected_branch),
            source_query_name=str(selected_branch),
        )


__all__ = [
    "AFTER_QUERY_ID",
    "BEFORE_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesTimelineRelativePositionEventLabelTask",
]
