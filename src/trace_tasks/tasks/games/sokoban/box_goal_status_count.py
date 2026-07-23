"""Sokoban paired box-goal status count task."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task

from ._lifecycle import SokobanLifecycleTask, build_box_goal_status_count_objective, run_sokoban_lifecycle
from .shared.sampling import (
    sample_box_goal_status_dataset,
    select_box_goal_status_answer_count,
)
from .shared.state import BOX_GOAL_STATUS_MODE_OFF, BOX_GOAL_STATUS_MODE_ON


TASK_ID = "task_games__sokoban__box_goal_status_count"
BOX_ON_GOAL_QUERY_ID = "box_on_goal_count"
BOX_OFF_GOAL_QUERY_ID = "box_off_goal_count"
SUPPORTED_QUERY_IDS = (BOX_ON_GOAL_QUERY_ID, BOX_OFF_GOAL_QUERY_ID)


def _status_mode_for_public_query(public_query: str) -> str:
    """Map public count queries to the corresponding Sokoban box status."""

    if str(public_query) == BOX_ON_GOAL_QUERY_ID:
        return BOX_GOAL_STATUS_MODE_ON
    if str(public_query) == BOX_OFF_GOAL_QUERY_ID:
        return BOX_GOAL_STATUS_MODE_OFF
    raise ValueError(f"unsupported Sokoban box-goal status query: {public_query}")


def _prepare_box_goal_status_count_objective(
    attempt_seed: int,
    task_params: Mapping[str, Any],
    public_query: str,
):
    """Construct a paired-color box/goal board and bind the counted boxes."""

    answer_count, count_support, count_probabilities = select_box_goal_status_answer_count(
        task_params,
        instance_seed=int(attempt_seed),
        namespace=TASK_ID,
    )
    status_mode = _status_mode_for_public_query(str(public_query))
    dataset = sample_box_goal_status_dataset(
        status_mode=str(status_mode),
        answer_count=int(answer_count),
        params=task_params,
        instance_seed=int(attempt_seed),
        namespace=TASK_ID,
    )
    return build_box_goal_status_count_objective(
        dataset=dataset,
        prompt_query_key=str(public_query),
        answer_count_support=[int(value) for value in count_support],
        answer_count_probabilities=count_probabilities,
        trace_extra_params={
            "status_mode": str(status_mode),
            "answer_count_support": [int(value) for value in count_support],
            "answer_count_probabilities": dict(count_probabilities),
        },
    )


@register_task
class GamesSokobanBoxGoalStatusCountTask(SokobanLifecycleTask):
    """Count boxes that are on or off their matching colored goal dots."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'matching')
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_sokoban_lifecycle(
            namespace=TASK_ID,
            supported_queries=SUPPORTED_QUERY_IDS,
            default_query=BOX_ON_GOAL_QUERY_ID,
            task_params=params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            build_objective=_prepare_box_goal_status_count_objective,
        )


__all__ = ["GamesSokobanBoxGoalStatusCountTask"]
