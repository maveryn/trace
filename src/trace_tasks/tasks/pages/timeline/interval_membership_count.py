"""Timeline task for counting events inside or outside a reference interval."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import select_task_query_id

from . import _lifecycle


TASK_ID = "task_pages__timeline__interval_membership_count"
BETWEEN_QUERY_ID = "between_reference_events_count"
OUTSIDE_QUERY_ID = "outside_reference_interval_count"
SUPPORTED_QUERY_IDS = (BETWEEN_QUERY_ID, OUTSIDE_QUERY_ID)


def _interval_relation_for_branch(selected_branch: str) -> str:
    if str(selected_branch) == BETWEEN_QUERY_ID:
        return "between"
    if str(selected_branch) == OUTSIDE_QUERY_ID:
        return "outside"
    raise ValueError(f"unsupported query_id for {TASK_ID}: {selected_branch}")


@register_task
class PagesTimelineIntervalMembershipCountTask:
    """Count event cards selected by a highlighted timeline interval relation."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison', 'logical_composition')
    domain = _lifecycle.DOMAIN
    supported_query_ids = SUPPORTED_QUERY_IDS
    default_dataset_enabled = True

    def generate(self, instance_seed: int, *, params: Mapping[str, Any], max_attempts: int):
        del max_attempts
        selected_branch, branch_probabilities, task_params = select_task_query_id(
            instance_seed=int(instance_seed),
            params=params,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=BETWEEN_QUERY_ID,
            task_id=TASK_ID,
        )
        interval_relation = _interval_relation_for_branch(str(selected_branch))
        return _lifecycle.build_timeline_response(
            instance_seed=int(instance_seed),
            params=task_params,
            selected_branch=str(selected_branch),
            branch_probabilities=branch_probabilities,
            program_mode=_lifecycle.INTERVAL_EVENT_MODE,
            interval_relation=str(interval_relation),
            prompt_query_key=str(selected_branch),
            source_query_name=str(selected_branch),
        )


__all__ = [
    "BETWEEN_QUERY_ID",
    "OUTSIDE_QUERY_ID",
    "SUPPORTED_QUERY_IDS",
    "TASK_ID",
    "PagesTimelineIntervalMembershipCountTask",
]
