"""Space-shooter enemy ship hit count task."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import SpaceShooterLifecycleTask, SpaceShooterObjective, run_space_shooter_lifecycle
from .shared.defaults import DEFAULTS
from .shared.sampling import resolve_target_answer, sample_enemy_ship_hit_scene
from .shared.state import SceneAxes


TASK_ID = "task_games__space_shooter__enemy_ship_hit_count"
PROMPT_QUERY_KEY = "enemy_ship_hit_count"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
JSON_EXAMPLE = '{"annotation":[[140,130,202,178],[420,250,482,298],[700,170,762,218]],"answer":3}'
JSON_EXAMPLE_ANSWER_ONLY = '{"answer":3}'


def _prepare_enemy_ship_hit_objective(rng, params: Mapping[str, Any], axes: SceneAxes, instance_seed: int) -> SpaceShooterObjective:
    """Construct a scene and count enemy ships destroyable by blue shots."""

    target, support, probabilities = resolve_target_answer(
        namespace=TASK_ID,
        instance_seed=int(instance_seed),
        params=params,
        support_key="enemy_ship_hit_count_support",
        fallback_support=DEFAULTS.enemy_ship_hit_count_support,
    )
    if int(target) > (int(axes.lane_count) * 3) and params.get("lane_count") is not None:
        raise ValueError("target_answer cannot exceed three hits per lane for enemy_ship_hit_count")
    sample = sample_enemy_ship_hit_scene(rng=rng, axes=axes, target_answer=int(target))
    sample = replace(
        sample,
        metadata={
            **dict(sample.metadata),
            "target_answer_support": [int(value) for value in support],
            "target_answer_probabilities": dict(probabilities),
        },
    )
    return SpaceShooterObjective(
        sample=sample,
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        prompt_query_key=PROMPT_QUERY_KEY,
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
        show_enemy_labels=False,
    )


@register_task
class GamesSpaceShooterEnemyShipHitCountTask(SpaceShooterLifecycleTask):
    """Count enemy ships that can be destroyed by current blue shots."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'ranking', 'aggregation')
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_space_shooter_lifecycle(
            namespace=TASK_ID,
            prompt_query_key=PROMPT_QUERY_KEY,
            supported_queries=SUPPORTED_QUERY_IDS,
            default_query=DEFAULT_QUERY_ID,
            task_params=params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            build_objective=_prepare_enemy_ship_hit_objective,
        )


__all__ = ["GamesSpaceShooterEnemyShipHitCountTask"]
