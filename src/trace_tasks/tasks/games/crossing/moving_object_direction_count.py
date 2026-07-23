"""Count crossing moving objects by visible arrow direction."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.games.crossing._lifecycle import (
    CrossingCountObjectiveSpec,
    CrossingObjectivePlan,
    prepare_count_objective_from_spec,
    run_crossing_lifecycle,
)
from trace_tasks.tasks.games.crossing.shared.defaults import SCENE_ID
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults


TASK_ID = "task_games__crossing__moving_object_direction_count"
LEFT_QUERY_ID = "left_moving_object_count"
RIGHT_QUERY_ID = "right_moving_object_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = (LEFT_QUERY_ID, RIGHT_QUERY_ID)
LEFT_MOVING_OBJECT_COUNT_SUPPORT: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
RIGHT_MOVING_OBJECT_COUNT_SUPPORT: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _direction_count_objective_spec(selected_query_id: str) -> CrossingCountObjectiveSpec:
    """Resolve one supported direction-count query into semantic arguments."""

    if str(selected_query_id) == LEFT_QUERY_ID:
        return CrossingCountObjectiveSpec(
            prompt_query_key=LEFT_QUERY_ID,
            count_mode="left_movers",
            support_key="left_moving_object_count_support",
            fallback_support=LEFT_MOVING_OBJECT_COUNT_SUPPORT,
            include_route_in_description=False,
        )
    if str(selected_query_id) == RIGHT_QUERY_ID:
        return CrossingCountObjectiveSpec(
            prompt_query_key=RIGHT_QUERY_ID,
            count_mode="right_movers",
            support_key="right_moving_object_count_support",
            fallback_support=RIGHT_MOVING_OBJECT_COUNT_SUPPORT,
            include_route_in_description=False,
        )
    raise ValueError(f"unsupported crossing direction query: {selected_query_id}")


def _prepare_direction_count_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    selected_query_id: str,
    _query_probabilities: Mapping[str, float],
) -> CrossingObjectivePlan:
    """Bind left/right direction-count semantics and exact-answer construction."""

    return prepare_count_objective_from_spec(
        task_id=TASK_ID,
        spec=_direction_count_objective_spec(str(selected_query_id)),
        instance_seed=int(instance_seed),
        task_params=task_params,
        selected_query_id=str(selected_query_id),
        gen_defaults=_GEN_DEFAULTS,
    )


@register_task
class GamesCrossingMovingObjectDirectionCountTask:
    """Count moving objects by whether their visible arrows point left or right."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict[str, Any], max_attempts: int) -> TaskOutput:
        """Generate an exact direction-filtered moving-object count."""

        return run_crossing_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=LEFT_QUERY_ID,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_direction_count_objective,
        )


__all__ = ["GamesCrossingMovingObjectDirectionCountTask"]
