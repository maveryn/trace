"""Count Bubble-shooter board bubbles that drop after the marked shot."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    BubbleShooterObjectivePlan,
    prepare_integer_outcome_objective,
    run_bubble_shooter_lifecycle,
)
from .shared.defaults import SCENE_ID
from .shared.sampling import ResolvedBubbleShooterBoardAxes, ResolvedBubbleShooterSceneAxes, sample_drop_state
from .shared.state import BubbleShooterState


TASK_ID = "task_games__bubble_shooter__drop_count"
QUERY_ID = "drop_count"
PROMPT_QUERY_KEY = "drop_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
ROW_COUNT_SUPPORT = (7,)
COL_COUNT_SUPPORT = (8, 9, 10)
DROP_COUNT_SUPPORT = (0, 1, 2, 3, 4)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _prepare_drop_count_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    _query_id: str,
    _query_probabilities: Mapping[str, float],
) -> BubbleShooterObjectivePlan:
    """Prepare the drop-count target and attempt constructor for one sample."""

    return prepare_integer_outcome_objective(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        task_id=TASK_ID,
        row_count_support=ROW_COUNT_SUPPORT,
        col_count_support=COL_COUNT_SUPPORT,
        support_key="drop_count_support",
        fallback_support=DROP_COUNT_SUPPORT,
        target_namespace=f"{TASK_ID}.target_answer",
        attempt_namespace="games.bubble_shooter.drop_count",
        prompt_query_key=PROMPT_QUERY_KEY,
        sample_state=_sample_drop_count_state,
        outcome_coords=_dropped_coords,
    )


def _sample_drop_count_state(
    rng,
    scene_axes: ResolvedBubbleShooterSceneAxes,
    board_axes: ResolvedBubbleShooterBoardAxes,
    target_value: int,
) -> BubbleShooterState:
    """Construct a board whose marked shot drops the requested number of bubbles."""

    return sample_drop_state(
        rng=rng,
        scene_axes=scene_axes,
        board_axes=board_axes,
        target_drop_count=int(target_value),
    )


def _dropped_coords(state: BubbleShooterState):
    """Return the objective-owned dropped-bubble annotation coordinates."""

    return state.outcome.dropped_coords


@register_task
class GamesBubbleShooterDropCountTask:
    """Count bubbles that drop because they are no longer connected to the top."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'topology', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_bubble_shooter_lifecycle(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_drop_count_objective,
            domain=self.domain,
        )


__all__ = ["GamesBubbleShooterDropCountTask"]
