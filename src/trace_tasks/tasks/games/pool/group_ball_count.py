"""Count visible pool balls in the current player's group."""

from __future__ import annotations

import json
from typing import Any, Dict, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import load_scene_generation_rendering_prompt_defaults
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import AttemptPoolResult, ObjectivePoolPlan, run_pool_lifecycle
from .shared.annotations import bbox_set_for_ball_ids
from .shared.defaults import DEFAULTS, SCENE_ID
from .shared.rules import group_display_name, object_balls, sorted_ids
from .shared.sampling import (
    PoolVisualAxes,
    resolve_pool_integer_axis,
    sample_numbered_pool_scene,
)
from .shared.state import PoolSceneState


TASK_ID = "task_games__pool__group_ball_count"
QUERY_ID = DEFAULT_QUERY_ID
PROMPT_QUERY_KEY = "current_group_ball_count"
SUPPORTED_QUERY_IDS = (QUERY_ID,)
_GEN_DEFAULTS, _RENDER_DEFAULTS, _PROMPT_DEFAULTS_UNUSED = load_scene_generation_rendering_prompt_defaults(
    "games",
    SCENE_ID,
    task_id=TASK_ID,
)


def _json_examples() -> tuple[str, str]:
    """Return valid format examples for pool group-count output."""

    return (
        json.dumps({"annotation": [[202, 232, 238, 268], [412, 312, 448, 348], [592, 262, 628, 298]], "answer": 3}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": 3}, separators=(",", ":"), ensure_ascii=False),
    )


def _resolve_current_player_group(instance_seed: int, params: Mapping[str, Any]) -> str:
    """Resolve the current player group once for the generated instance."""

    explicit_group = params.get("current_player_group")
    if explicit_group is not None:
        group = str(explicit_group)
    else:
        rng = spawn_rng(int(instance_seed), f"{TASK_ID}.current_player_group")
        group = str(rng.choice(("solid", "stripe")))
    if group not in {"solid", "stripe"}:
        raise ValueError(f"unsupported current_player_group={group!r}")
    return group


def _matching_group_ids(state: PoolSceneState) -> tuple[str, ...]:
    """Return sorted ids for visible balls in the current player group."""

    if state.current_player_group is None:
        raise ValueError("pool group-count state requires current player group")
    return sorted_ids(
        ball.ball_id
        for ball in object_balls(state.balls)
        if str(ball.group) == str(state.current_player_group)
    )


def _number_pool_for_group(group: str) -> tuple[int, ...]:
    """Return the object-ball numbers that belong to one player group."""

    if str(group) == "solid":
        return tuple(range(1, 8))
    if str(group) == "stripe":
        return tuple(range(9, 16))
    raise ValueError(f"unsupported current_player_group={group!r}")


def _distractor_number_pool(group: str) -> tuple[int, ...]:
    """Return object-ball numbers outside the current player's group."""

    opponent_numbers = tuple(range(9, 16)) if str(group) == "solid" else tuple(range(1, 8))
    return (*opponent_numbers, 8)


def _select_visible_object_numbers(
    *,
    rng: Any,
    current_group: str,
    target_answer: int,
    total_object_balls: int,
) -> tuple[int, ...]:
    """Select exact current-group and distractor ball numbers for the task."""

    group_numbers = _number_pool_for_group(str(current_group))
    distractor_numbers = _distractor_number_pool(str(current_group))
    if int(target_answer) > len(group_numbers):
        raise ValueError("pool group-count answer exceeds group size")
    distractor_count = int(total_object_balls) - int(target_answer)
    if int(distractor_count) < 0 or int(distractor_count) > len(distractor_numbers):
        raise ValueError("pool group-count distractor count is infeasible")
    selected = [
        *rng.sample(group_numbers, int(target_answer)),
        *rng.sample(distractor_numbers, int(distractor_count)),
    ]
    rng.shuffle(selected)
    return tuple(int(number) for number in selected)


def _validate_group_number_mix(
    numbers: tuple[int, ...],
    *,
    current_group: str,
    target_answer: int,
) -> None:
    """Verify selected numbers encode the requested group-count answer."""

    group_numbers = set(_number_pool_for_group(str(current_group)))
    matching_count = sum(1 for number in numbers if int(number) in group_numbers)
    if int(matching_count) != int(target_answer):
        raise ValueError("selected pool numbers do not encode target group count")


def _validate_group_count_state(state: PoolSceneState, *, target_answer: int) -> tuple[str, ...]:
    """Verify group-count construction and return annotation entity ids."""

    annotation_ids = _matching_group_ids(state)
    if len(annotation_ids) != int(target_answer):
        raise ValueError("pool group-count construction did not match target")
    return annotation_ids


def _prepare_group_objective(
    instance_seed: int,
    params: Mapping[str, Any],
    query_probabilities: Mapping[str, float],
    _query_id: str,
    axes: PoolVisualAxes,
) -> ObjectivePoolPlan:
    """Resolve group-count target axes and bind the scene constructor."""

    target_axis = resolve_pool_integer_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        support_key="current_group_ball_count_support",
        explicit_key="target_answer",
        fallback_support=DEFAULTS.current_group_ball_count_support,
        namespace=f"{TASK_ID}.target_answer",
    )
    current_group = _resolve_current_player_group(int(instance_seed), params)

    def construct_attempt(rng: Any, attempt_axes: PoolVisualAxes) -> AttemptPoolResult:
        """Build a table whose selected object numbers encode the requested group count."""

        selected_numbers = _select_visible_object_numbers(
            rng=rng,
            current_group=str(current_group),
            target_answer=int(target_axis.target_value),
            total_object_balls=int(attempt_axes.object_ball_count),
        )
        _validate_group_number_mix(
            selected_numbers,
            current_group=str(current_group),
            target_answer=int(target_axis.target_value),
        )
        state = sample_numbered_pool_scene(
            rng=rng,
            axes=attempt_axes,
            object_numbers=selected_numbers,
            current_player_group=str(current_group),
        )
        annotation_ids = _validate_group_count_state(state, target_answer=int(target_axis.target_value))
        group_text = group_display_name(str(current_group))
        return AttemptPoolResult(
            state=state,
            answer_gt=TypedValue(type="integer", value=len(annotation_ids)),
            annotation_entity_ids=tuple(annotation_ids),
            build_annotation=lambda rendered: bbox_set_for_ball_ids(rendered.rendered_scene, annotation_ids),
            witness_type="object_set",
            badge_text=f"Current player: {group_text.upper()}",
            dynamic_prompt_slots={"current_player_group": group_text},
            query_params={
                "current_player_group": str(current_group),
                "selected_object_numbers": [int(number) for number in selected_numbers],
                "target_answer": int(target_axis.target_value),
                "target_answer_support": [int(value) for value in target_axis.target_value_support],
                "target_answer_probabilities": dict(target_axis.target_value_probabilities),
            },
            relations_extra={
                "current_player_group": str(current_group),
                "target_answer": int(target_axis.target_value),
            },
            execution_extra={
                "target_answer": int(target_axis.target_value),
                "target_answer_support": [int(value) for value in target_axis.target_value_support],
                "annotation_ball_ids": list(annotation_ids),
                "annotation_pocket_ids": [],
            },
        )

    json_example, json_example_answer_only = _json_examples()
    return ObjectivePoolPlan(
        attempt_namespace="games.pool.visible_group_count",
        prompt_query_key=PROMPT_QUERY_KEY,
        answer_hint_key="answer_hint_current_group_ball_count",
        annotation_hint_key="annotation_hint_current_group_ball_count",
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
        query_params={"query_id_probabilities": dict(query_probabilities)},
        construct_attempt=construct_attempt,
    )


@register_task
class GamesPoolGroupBallCountTask:
    """Count visible balls in the current player's pool group."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
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
            prepare_objective=_prepare_group_objective,
        )


__all__ = ["GamesPoolGroupBallCountTask"]
