"""Count Backgammon points matching one checker stack-state predicate."""

from __future__ import annotations
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.games.shared.sampling import resolve_games_named_axis
from ._lifecycle import (
    BackgammonObjectivePlan,
    integer_count_attempt_result,
    resolve_backgammon_count_target,
    run_backgammon_lifecycle,
)
from .shared.sampling import sample_point_state_count_scene
from .shared.state import (
    PLAYER_BLACK,
    PLAYER_WHITE,
    SCENE_ID,
    STACK_STATE_SINGLE,
    STACK_STATE_TWO_OR_MORE,
)

TASK_ID = "task_games__backgammon__point_state_count"
QUERY_ID = "single"
POINT_STATE_COUNT_SUPPORT = (0, 1, 2, 3, 4, 5, 6)
SUPPORTED_CHECKER_COLORS = (PLAYER_BLACK, PLAYER_WHITE)
SUPPORTED_STACK_STATES = (STACK_STATE_SINGLE, STACK_STATE_TWO_OR_MORE)
SUPPORTED_QUERY_IDS = (QUERY_ID,)
STACK_STATE_PHRASES = {
    STACK_STATE_SINGLE: "exactly one {color} checker",
    STACK_STATE_TWO_OR_MORE: "two or more {color} checkers",
}
_GEN_DEFAULTS, _RENDER_DEFAULTS_UNUSED, _PROMPT_DEFAULTS_UNUSED = (
    load_scene_generation_rendering_prompt_defaults(
        "games",
        SCENE_ID,
        task_id=TASK_ID,
    )
)


def _resolve_named_operand(
    instance_seed,
    task_params,
    *,
    explicit_key,
    weights_key,
    balance_flag_key,
    namespace,
    supported,
):
    return resolve_games_named_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=_GEN_DEFAULTS,
        namespace=str(namespace),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_flag_key),
        supported_variants=tuple(str(value) for value in supported),
    )


def _prepare_point_state_objective(instance_seed, task_params, query_id):
    """Resolve the checker-stack predicate and bind exact-count sample construction."""
    if str(query_id) != QUERY_ID:
        raise ValueError(f"unsupported Backgammon point-state query: {query_id}")
    checker_color, checker_color_probabilities = _resolve_named_operand(
        instance_seed,
        task_params,
        explicit_key="target_checker_color",
        weights_key="target_checker_color_weights",
        balance_flag_key="balanced_target_checker_color_sampling",
        namespace=f"{TASK_ID}.target_checker_color",
        supported=SUPPORTED_CHECKER_COLORS,
    )
    stack_state, stack_state_probabilities = _resolve_named_operand(
        instance_seed,
        task_params,
        explicit_key="target_stack_state",
        weights_key="target_stack_state_weights",
        balance_flag_key="balanced_target_stack_state_sampling",
        namespace=f"{TASK_ID}.target_stack_state",
        supported=SUPPORTED_STACK_STATES,
    )
    target = resolve_backgammon_count_target(
        instance_seed=int(instance_seed),
        task_params=task_params,
        task_id=TASK_ID,
        support_key="point_state_count_support",
        fallback_support=POINT_STATE_COUNT_SUPPORT,
        namespace="backgammon.point_state_count.target_answer",
    )
    color_text = str(checker_color)
    state_phrase = str(STACK_STATE_PHRASES[str(stack_state)]).format(color=color_text)

    def construct_attempt(rng, axes):
        sample = sample_point_state_count_scene(
            rng,
            axes=axes,
            checker_color=str(checker_color),
            stack_state=str(stack_state),
            target_answer=int(target.target_answer),
        )
        return integer_count_attempt_result(
            sample=sample,
            target_points=tuple((int(point) for point in sample.target_points)),
            construction_mode="exact_point_state_count",
        )

    return BackgammonObjectivePlan(
        attempt_namespace=f"games.backgammon.point_state_count.{color_text}.{str(stack_state)}",
        prompt_query_key="point_state_count",
        prompt_dynamic_slots={
            "point_state_phrase": state_phrase,
        },
        query_params={
            "target_answer_support": [
                int(value) for value in target.target_answer_support
            ],
            "target_answer_probabilities": dict(target.target_answer_probabilities),
            "target_checker_color": color_text,
            "target_checker_color_probabilities": dict(checker_color_probabilities),
            "target_stack_state": str(stack_state),
            "target_stack_state_probabilities": dict(stack_state_probabilities),
        },
        construct_attempt=construct_attempt,
    )


@register_task
class GamesBackgammonPointStateCountTask:
    """Count numbered Backgammon points by checker color and stack state."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(
        self, instance_seed: int, *, params: dict, max_attempts: int
    ) -> TaskOutput:
        """Generate a stack-state count board by binding color and stack predicate locally."""
        return run_backgammon_lifecycle(
            task_id=self.task_id,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_point_state_objective,
        )


__all__ = ["GamesBackgammonPointStateCountTask"]
