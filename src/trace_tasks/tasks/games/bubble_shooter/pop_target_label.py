"""Choose the labeled Bubble-shooter landing target that would make bubbles pop."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)

from ._lifecycle import (
    BubbleShooterObjectivePlan,
    prepare_landing_target_label_objective,
    run_bubble_shooter_lifecycle,
)
from .shared.defaults import SCENE_ID
from .shared.sampling import ResolvedBubbleShooterSceneAxes, sample_pop_target_state
from .shared.state import BUBBLE_OPTION_LABELS

TASK_ID = "task_games__bubble_shooter__pop_target_label"
QUERY_ID = "pop_target_label"
PROMPT_QUERY_KEY = "pop_target_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
ROW_COUNT_SUPPORT = (7, 8, 9)
COL_COUNT_SUPPORT = (8, 9, 10)
OPTION_LABELS = BUBBLE_OPTION_LABELS[:4]
TARGET_LABEL_SUPPORT = OPTION_LABELS
POSITIVE_POP_COUNT_SUPPORT = (2, 3, 4, 5)
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
        task_id=TASK_ID,
    )
)


def _prepare_pop_target_label_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    _query_id: str,
    _query_probabilities: Mapping[str, float],
) -> BubbleShooterObjectivePlan:
    """Prepare the target label and landing-option constructor for one sample."""

    return prepare_landing_target_label_objective(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        task_id=TASK_ID,
        row_count_support=ROW_COUNT_SUPPORT,
        col_count_support=COL_COUNT_SUPPORT,
        prompt_query_key=PROMPT_QUERY_KEY,
        target_label_support_key="pop_target_label_support",
        fallback_target_label_support=TARGET_LABEL_SUPPORT,
        displayed_option_labels=OPTION_LABELS,
        positive_pop_count_support_key="positive_pop_count_support",
        fallback_positive_pop_count_support=POSITIVE_POP_COUNT_SUPPORT,
        target_namespace=f"{TASK_ID}.target_label",
        pop_count_namespace=f"{TASK_ID}.positive_pop_count",
        attempt_namespace="games.bubble_shooter.pop_target_label",
        sample_state=_sample_pop_target_state,
    )


def _sample_pop_target_state(
    rng,
    scene_axes: ResolvedBubbleShooterSceneAxes,
    board_axes,
    target_label: str,
    option_labels,
    positive_pop_count: int,
):
    """Construct a target-label board using task-owned semantic arguments."""

    return sample_pop_target_state(
        rng=rng,
        scene_axes=scene_axes,
        board_axes=board_axes,
        target_option_label=str(target_label),
        option_labels=tuple(str(label) for label in option_labels),
        target_pop_count=int(positive_pop_count),
    )


@register_task
class GamesBubbleShooterPopTargetLabelTask:
    """Choose the labeled landing target that would pop bubbles."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'comparison', 'topology', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(
        self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int
    ) -> TaskOutput:
        return run_bubble_shooter_lifecycle(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_pop_target_label_objective,
            domain=self.domain,
        )


__all__ = ["GamesBubbleShooterPopTargetLabelTask"]
