"""Space-shooter visible enemy ship count task."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import SpaceShooterLifecycleTask, SpaceShooterObjective, run_space_shooter_lifecycle
from .shared.sampling import sample_enemy_ship_count_scene
from .shared.state import SceneAxes


TASK_ID = "task_games__space_shooter__enemy_ship_count"
PROMPT_QUERY_KEY = "enemy_ship_count"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
JSON_EXAMPLE = '{"annotation":[[120,180,182,228],[320,260,382,308],[520,340,582,388]],"answer":3}'
JSON_EXAMPLE_ANSWER_ONLY = '{"answer":3}'


def _prepare_enemy_ship_count_objective(rng, params: Mapping[str, Any], axes: SceneAxes, instance_seed: int) -> SpaceShooterObjective:
    """Construct a scene and count every visible enemy ship."""

    sample = sample_enemy_ship_count_scene(rng=rng, axes=axes)
    return SpaceShooterObjective(
        sample=sample,
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        prompt_query_key=PROMPT_QUERY_KEY,
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
        show_enemy_labels=False,
    )


@register_task
class GamesSpaceShooterEnemyShipCountTask(SpaceShooterLifecycleTask):
    """Count every visible enemy ship in the playfield."""

    task_id = TASK_ID
    reasoning_operations = ('counting',)
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
            build_objective=_prepare_enemy_ship_count_objective,
        )


__all__ = ["GamesSpaceShooterEnemyShipCountTask"]
