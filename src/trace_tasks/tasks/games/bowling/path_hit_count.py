"""Count Bowling pins intersected by the shown straight path."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    BowlingObjectivePlan,
    bowling_integer_axis_spec,
    pin_bbox_set_count_attempt,
    resolve_bowling_integer_axis_specs,
    run_bowling_lifecycle,
)
from .shared.defaults import SCENE_ID
from .shared.rules import sample_path_hit_count_scene
from .shared.sampling import ResolvedBowlingSceneAxes


TASK_ID = "task_games__bowling__path_hit_count"
QUERY_ID = "path_hit_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
PATH_HIT_COUNT_SUPPORT = (1, 2, 3, 4, 5)
PATH_HIT_COUNT_AXIS_SPECS = (
    bowling_integer_axis_spec(
        "target_hit_count",
        PATH_HIT_COUNT_SUPPORT,
        support_key="path_hit_count_support",
        explicit_key="target_answer",
        balanced_flag_key="balanced_path_hit_count_sampling",
        trace_aliases=("target_answer",),
    ),
)
_GEN_DEFAULTS = load_scene_generation_rendering_prompt_defaults("games", SCENE_ID, task_id=TASK_ID)[0]


def _prepare_path_hit_count_objective(
    instance_seed,
    task_params,
    _query_id,
    _query_probabilities,
):
    """Resolve the hit-count target and bind strict-clearance construction."""

    axis_values, query_params = resolve_bowling_integer_axis_specs(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        task_id=TASK_ID,
        axis_specs=PATH_HIT_COUNT_AXIS_SPECS,
    )
    target_axis = axis_values["target_hit_count"]

    def construct_attempt(rng, axes: ResolvedBowlingSceneAxes):
        sample = sample_path_hit_count_scene(
            rng=rng,
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            target_hit_count=int(target_axis.value),
        )
        return pin_bbox_set_count_attempt(
            sample=sample,
            answer_value=int(target_axis.value),
            execution_extra={
                "target_hit_count": int(target_axis.value),
                "path_hit_pin_ids": [str(entity_id) for entity_id in sample.annotation_entity_ids],
            },
        )

    return BowlingObjectivePlan(
        attempt_namespace="games.bowling.path_hit_count",
        prompt_query_key=QUERY_ID,
        render_mode="first_pin_path",
        query_params=query_params,
        construct_attempt=construct_attempt,
    )


@register_task
class GamesBowlingPathHitCountTask:
    """Count standing pins whose visible body is crossed by the shown path."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations', 'topology')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_bowling_lifecycle(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_path_hit_count_objective,
        )


__all__ = ["GamesBowlingPathHitCountTask", "PATH_HIT_COUNT_SUPPORT"]
