"""Count bricks remaining in the hit row after one Brick-breaker shot."""

from __future__ import annotations

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults

from ._lifecycle import (
    BrickBreakerObjectivePlan,
    bbox_set_attempt,
    brick_breaker_integer_axis_spec,
    resolve_brick_breaker_integer_axis_spec,
    resolve_brick_breaker_playfield_axis_specs,
    run_brick_breaker_lifecycle,
)
from .shared.defaults import SCENE_ID
from .shared.rules import sample_hit_row_remaining_scene
from .shared.sampling import ResolvedBrickBreakerSceneAxes


TASK_ID = "task_games__brick_breaker__hit_row_remaining_count"
QUERY_ID = "hit_row_remaining_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
BRICK_ROW_COUNT_SUPPORT = (4, 5)
BRICK_COL_COUNT_SUPPORT = (5, 6)
CATCH_LANE_COUNT_SUPPORT = (5, 6)
ROW_REMAINING_AXIS_SPEC = brick_breaker_integer_axis_spec(
    "row_remaining_count",
    (1, 2, 3, 4, 5),
    balanced_flag_key="balanced_row_remaining_count_sampling",
)
_GEN_DEFAULTS = load_scene_generation_rendering_prompt_defaults("games", SCENE_ID, task_id=TASK_ID)[0]


def _prepare_hit_row_remaining_objective(
    instance_seed,
    task_params,
    _query_id,
    _query_probabilities,
):
    """Resolve playfield/count axes and bind the same-row survivor constructor."""

    playfield_axes, playfield_query_params = resolve_brick_breaker_playfield_axis_specs(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        task_id=TASK_ID,
        brick_row_count_support=BRICK_ROW_COUNT_SUPPORT,
        brick_col_count_support=BRICK_COL_COUNT_SUPPORT,
        catch_lane_count_support=CATCH_LANE_COUNT_SUPPORT,
    )
    row_axis, row_query_params = resolve_brick_breaker_integer_axis_spec(
        instance_seed=int(instance_seed),
        task_params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        task_id=TASK_ID,
        spec=ROW_REMAINING_AXIS_SPEC,
    )
    brick_cols = max(int(playfield_axes.brick_cols.value), int(row_axis.value) + 1)

    def construct_attempt(rng, axes: ResolvedBrickBreakerSceneAxes):
        sample = sample_hit_row_remaining_scene(
            rng=rng,
            scene_variant=str(axes.scene_variant),
            brick_rows=int(playfield_axes.brick_rows.value),
            brick_cols=int(brick_cols),
            lane_count=int(playfield_axes.lane_count.value),
            row_remaining_count=int(row_axis.value),
        )
        if sample.target_row_remaining_count is None:
            raise ValueError("hit-row Brick-breaker sample is missing row remaining count")
        return bbox_set_attempt(
            sample=sample,
            answer_gt=TypedValue(type="integer", value=int(sample.target_row_remaining_count)),
            annotation_entity_ids=tuple(str(entity_id) for entity_id in sample.target_row_remaining_brick_ids),
            execution_extra={
                "row_remaining_count": int(row_axis.value),
            },
        )

    return BrickBreakerObjectivePlan(
        attempt_namespace="games.brick_breaker.hit_row_remaining_count",
        prompt_query_key=QUERY_ID,
        render_mode="brick_hit_path",
        query_params={
            **dict(playfield_query_params),
            **dict(row_query_params),
            "brick_cols": int(brick_cols),
        },
        construct_attempt=construct_attempt,
    )


@register_task
class GamesBrickBreakerHitRowRemainingCountTask:
    """Count same-row bricks remaining after the visible ball path hits one brick."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'state_update')
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
            prepare_objective=_prepare_hit_row_remaining_objective,
        )


__all__ = ["GamesBrickBreakerHitRowRemainingCountTask"]
