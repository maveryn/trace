"""Identify the first Bowling pin reached by the shown path."""

from __future__ import annotations

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    BowlingObjectivePlan,
    bowling_integer_axis_spec,
    pin_point_label_attempt,
    resolve_bowling_integer_axis_specs,
    run_bowling_lifecycle,
)
from .shared.defaults import SCENE_ID
from .shared.rules import sample_first_pin_hit_scene
from .shared.sampling import ResolvedBowlingSceneAxes


TASK_ID = "task_games__bowling__first_pin_hit_label"
QUERY_ID = "first_pin_hit_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
FIRST_PIN_AXIS_SPECS = (
    bowling_integer_axis_spec(
        "visible_pin_count",
        (4, 5, 6, 7, 8, 9),
        balanced_flag_key="balanced_visible_pin_count_sampling",
    ),
    bowling_integer_axis_spec(
        "target_pin_index",
        tuple(range(10)),
        balanced_flag_key="balanced_target_pin_sampling",
        trace_aliases=("target_pin_label_index",),
    ),
)
_GEN_DEFAULTS = load_scene_generation_rendering_prompt_defaults("games", SCENE_ID, task_id=TASK_ID)[0]


def _prepare_first_pin_objective(
    instance_seed,
    task_params,
    _query_id,
    _query_probabilities,
):
    """Resolve first-pin target axes and bind the collision-path constructor."""

    axis_values, query_params = resolve_bowling_integer_axis_specs(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        task_id=TASK_ID,
        axis_specs=FIRST_PIN_AXIS_SPECS,
    )
    visible_axis = axis_values["visible_pin_count"]
    target_axis = axis_values["target_pin_index"]

    def construct_attempt(rng, axes: ResolvedBowlingSceneAxes):
        sample = sample_first_pin_hit_scene(
            rng=rng,
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            target_pin_label_index=int(target_axis.value),
            visible_pin_count=int(visible_axis.value),
        )
        return pin_point_label_attempt(
            sample=sample,
            answer_value=str(sample.target_pin_label),
            execution_extra={
                "target_pin_index": int(target_axis.value),
            },
        )

    return BowlingObjectivePlan(
        attempt_namespace="games.bowling.first_pin_hit_label",
        prompt_query_key=QUERY_ID,
        render_mode="first_pin_path",
        query_params=query_params,
        construct_attempt=construct_attempt,
    )


@register_task
class GamesBowlingFirstPinHitLabelTask:
    """Identify the labeled pin hit first by the visible ball path."""

    task_id = TASK_ID
    reasoning_operations = ('ranking', 'spatial_relations', 'topology')
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
            prepare_objective=_prepare_first_pin_objective,
        )


__all__ = ["GamesBowlingFirstPinHitLabelTask"]
