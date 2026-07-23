"""Count pool balls blocking a marked shot lane."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import AttemptPoolResult, ObjectivePoolPlan, run_pool_lifecycle
from .shared.annotations import bbox_set_for_ball_ids
from .shared.defaults import DEFAULTS, SCENE_ID
from .shared.rules import balls_on_segment, pocket_by_id, sorted_ids
from .shared.sampling import (
    PoolVisualAxes,
    resolve_pool_integer_axis,
    sample_marked_shot_pool_scene,
)
from .shared.state import PoolBall, PoolSceneState


TASK_ID = "task_games__pool__blocking_ball_count"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "blocking_ball_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _json_examples() -> tuple[str, str]:
    """Return valid format examples for pool blocker-count output."""

    return (
        json.dumps({"annotation": [[242, 282, 278, 318], [502, 242, 538, 278]], "answer": 2}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": 2}, separators=(",", ":"), ensure_ascii=False),
    )


def _ball_by_id(state: PoolSceneState, ball_id: str) -> PoolBall:
    """Return a generated pool ball by stable id."""

    for ball in state.balls:
        if str(ball.ball_id) == str(ball_id):
            return ball
    raise ValueError(f"missing pool ball {ball_id!r}")


def _blocking_ball_ids(state: PoolSceneState, *, clearance: float) -> tuple[str, ...]:
    """Recompute blockers on the two marked shot segments."""

    if state.marked_ball_id is None or state.marked_pocket_id is None:
        raise ValueError("pool blocker-count state requires marked ball and pocket")
    cue = _ball_by_id(state, str(state.cue_ball_id))
    target = _ball_by_id(state, str(state.marked_ball_id))
    pocket = pocket_by_id(state.pockets, str(state.marked_pocket_id))
    first_segment = balls_on_segment(
        balls=state.balls,
        start=cue.center,
        end=target.center,
        ignore_ball_ids=(str(cue.ball_id), str(target.ball_id)),
        clearance=float(clearance),
    )
    second_segment = balls_on_segment(
        balls=state.balls,
        start=target.center,
        end=pocket.center,
        ignore_ball_ids=(str(target.ball_id),),
        clearance=float(clearance),
    )
    return sorted_ids(ball.ball_id for ball in (*first_segment, *second_segment))


def _validate_blocking_count_state(
    state: PoolSceneState,
    *,
    target_answer: int,
    clearance: float,
) -> tuple[str, ...]:
    """Verify blocker-count construction and return annotation entity ids."""

    blocker_ids = _blocking_ball_ids(state, clearance=float(clearance))
    if len(blocker_ids) != int(target_answer):
        raise ValueError("constructed pool blocker count did not match target")
    return blocker_ids


def _prepare_blocking_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    _query_id: str,
    axes: PoolVisualAxes,
) -> ObjectivePoolPlan:
    """Resolve blocker-count target axes and bind the scene constructor."""

    target_axis = resolve_pool_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="blocking_ball_count_support",
        explicit_key="target_answer",
        fallback_support=DEFAULTS.blocking_ball_count_support,
        namespace=f"{TASK_ID}.target_answer",
    )

    def construct_attempt(rng: Any, attempt_axes: PoolVisualAxes) -> AttemptPoolResult:
        state = sample_marked_shot_pool_scene(
            rng=rng,
            axes=attempt_axes,
            target_answer=int(target_axis.target_value),
        )
        blocker_ids = _validate_blocking_count_state(
            state,
            target_answer=int(target_axis.target_value),
            clearance=float(attempt_axes.line_clearance),
        )
        return AttemptPoolResult(
            state=state,
            answer_gt=TypedValue(type="integer", value=len(blocker_ids)),
            annotation_entity_ids=tuple(blocker_ids),
            build_annotation=lambda rendered: bbox_set_for_ball_ids(rendered.rendered_scene, blocker_ids),
            witness_type="object_set",
            show_shot_path=True,
            query_params={
                "target_answer": int(target_axis.target_value),
                "target_answer_support": [int(value) for value in target_axis.target_value_support],
                "target_answer_probabilities": dict(target_axis.target_value_probabilities),
            },
            relations_extra={
                "target_answer": int(target_axis.target_value),
                "blocking_ball_ids": list(blocker_ids),
            },
            execution_extra={
                "target_answer": int(target_axis.target_value),
                "target_answer_support": [int(value) for value in target_axis.target_value_support],
                "blocking_ball_ids": list(blocker_ids),
                "annotation_ball_ids": list(blocker_ids),
                "annotation_pocket_ids": [],
            },
        )

    json_example, json_example_answer_only = _json_examples()
    return ObjectivePoolPlan(
        attempt_namespace="games.pool.marked_shot_blockers",
        prompt_query_key=PROMPT_QUERY_KEY,
        answer_hint_key="answer_hint_blocking_ball_count",
        annotation_hint_key="annotation_hint_blocking_ball_count",
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
        query_params={"query_id_probabilities": dict(query_probabilities)},
        construct_attempt=construct_attempt,
    )


@register_task
class GamesPoolBlockingBallCountTask:
    """Count balls blocking the marked direct shot lane."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'spatial_relations')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: Dict[str, Any], max_attempts: int) -> TaskOutput:
        return run_pool_lifecycle(
            task_id=TASK_ID,
            domain=self.domain,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=QUERY_ID,
            gen_defaults=_GEN_DEFAULTS,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_blocking_objective,
        )


__all__ = ["GamesPoolBlockingBallCountTask"]
