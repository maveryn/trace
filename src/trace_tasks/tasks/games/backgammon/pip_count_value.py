"""Compute the active player's Backgammon pip count."""

from __future__ import annotations

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import (
    BackgammonObjectivePlan,
    integer_count_attempt_result,
    resolve_backgammon_count_target,
    run_backgammon_lifecycle,
)
from .shared.sampling import sample_pip_count_scene


TASK_ID = "task_games__backgammon__pip_count_value"
QUERY_ID = "single"
PIP_COUNT_SUPPORT = (1, 2, 3, 4, 5, 6, 7, 8)
SUPPORTED_QUERY_IDS = (QUERY_ID,)


def _prepare_pip_count_objective(instance_seed, task_params, query_id):
    """Resolve the exact pip-count target and bind sparse race-position construction."""

    if str(query_id) != QUERY_ID:
        raise ValueError(f"unsupported Backgammon pip-count query: {query_id}")
    target = resolve_backgammon_count_target(
        instance_seed=int(instance_seed),
        task_params=task_params,
        task_id=TASK_ID,
        support_key="pip_count_support",
        fallback_support=PIP_COUNT_SUPPORT,
        namespace="backgammon.pip_count_value.target_answer",
    )

    def construct_attempt(rng, axes):
        sample = sample_pip_count_scene(
            rng,
            axes=axes,
            target_answer=int(target.target_answer),
        )
        return integer_count_attempt_result(
            sample=sample,
            target_points=tuple(int(point) for point in sample.target_points),
            construction_mode="exact_pip_count",
        )

    return BackgammonObjectivePlan(
        attempt_namespace="games.backgammon.pip_count_value",
        prompt_query_key="pip_count_value",
        prompt_dynamic_slot_builder=lambda sample: {
            "active_player_label": str(sample.active_player),
        },
        query_params={
            "target_answer_support": [int(value) for value in target.target_answer_support],
            "target_answer_probabilities": dict(target.target_answer_probabilities),
            "pip_count_support": [int(value) for value in target.target_answer_support],
        },
        construct_attempt=construct_attempt,
    )


@register_task
class GamesBackgammonPipCountValueTask:
    """Compute the active player's total pips to bear off."""

    task_id = TASK_ID
    reasoning_operations = ('aggregation', 'topology', 'formula_evaluation')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int) -> TaskOutput:
        """Generate a sparse exact-answer pip-count position."""

        return run_backgammon_lifecycle(
            task_id=self.task_id,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_pip_count_objective,
        )


__all__ = ["GamesBackgammonPipCountValueTask", "PIP_COUNT_SUPPORT"]
