"""Identify the bottom catch lane reached by the visible ball trajectory."""

from __future__ import annotations

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    BrickBreakerObjectivePlan,
    point_attempt,
    resolve_brick_breaker_playfield_axis_specs,
    run_brick_breaker_lifecycle,
)
from .shared.defaults import SCENE_ID
from .shared.rules import sample_paddle_catch_scene
from .shared.sampling import ResolvedBrickBreakerSceneAxes


TASK_ID = "task_games__brick_breaker__paddle_catch_label"
QUERY_ID = "paddle_catch_label"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
BRICK_ROW_COUNT_SUPPORT = (4, 5)
BRICK_COL_COUNT_SUPPORT = (5, 6)
CATCH_LANE_COUNT_SUPPORT = (5, 6)
_GEN_DEFAULTS = load_scene_generation_rendering_prompt_defaults("games", SCENE_ID, task_id=TASK_ID)[0]


def _prepare_paddle_catch_objective(
    instance_seed,
    task_params,
    _query_id,
    _query_probabilities,
):
    """Resolve playfield axes and bind the bottom catch-lane constructor."""

    playfield_axes, query_params = resolve_brick_breaker_playfield_axis_specs(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        task_id=TASK_ID,
        brick_row_count_support=BRICK_ROW_COUNT_SUPPORT,
        brick_col_count_support=BRICK_COL_COUNT_SUPPORT,
        catch_lane_count_support=CATCH_LANE_COUNT_SUPPORT,
    )

    def construct_attempt(rng, axes: ResolvedBrickBreakerSceneAxes):
        sample = sample_paddle_catch_scene(
            rng=rng,
            scene_variant=str(axes.scene_variant),
            brick_rows=int(playfield_axes.brick_rows.value),
            brick_cols=int(playfield_axes.brick_cols.value),
            lane_count=int(playfield_axes.lane_count.value),
        )
        if sample.target_lane_label is None:
            raise ValueError("paddle-catch Brick-breaker sample is missing target lane label")
        return point_attempt(
            sample=sample,
            answer_gt=TypedValue(type="string", value=str(sample.target_lane_label)),
        )

    return BrickBreakerObjectivePlan(
        attempt_namespace="games.brick_breaker.paddle_catch_label",
        prompt_query_key=QUERY_ID,
        render_mode="paddle_catch_path",
        query_params=dict(query_params),
        construct_attempt=construct_attempt,
    )


@register_task
class GamesBrickBreakerPaddleCatchLabelTask:
    """Identify the labeled bottom catch lane reached by the visible ball path."""

    task_id = TASK_ID
    reasoning_operations = ('spatial_relations',)
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_brick_breaker_lifecycle(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_paddle_catch_objective,
        )


__all__ = ["GamesBrickBreakerPaddleCatchLabelTask"]
