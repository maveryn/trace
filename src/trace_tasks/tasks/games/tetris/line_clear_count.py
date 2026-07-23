"""Maximum line-clear count for a shown Tetris board and next piece."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import TetrisObjectivePlan, resolve_tetris_integer_target, run_tetris_lifecycle
from .shared.defaults import DEFAULTS
from .shared.prompts import format_json_examples
from .shared.rendering import RENDER_MODE_LINE_CLEAR
from .shared.sampling import build_line_clear_sample


TASK_ID = "task_games__tetris__line_clear_count"
PROMPT_QUERY_KEY = "max_clear_with_next_piece"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
JSON_EXAMPLE, JSON_EXAMPLE_ANSWER_ONLY = format_json_examples(
    annotation={"board": [40, 120, 320, 640], "next_piece": [420, 140, 520, 260]},
    answer=2,
)


def _prepare_line_clear_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    _selected_query: str,
    _query_probabilities: Mapping[str, float],
    _axes,
) -> TetrisObjectivePlan:
    """Resolve the target max-clear count and bind a legal-placement constructor."""

    target_clear_count, target_probabilities, target_support = resolve_tetris_integer_target(
        instance_seed=int(instance_seed),
        params=task_params,
        support_key="line_clear_count_support",
        explicit_key="target_clear_count",
        fallback_support=DEFAULTS.line_clear_count_support,
        namespace=f"{TASK_ID}.target_clear_count",
    )

    def construct_attempt(rng, axes):
        return build_line_clear_sample(
            rng,
            scene_variant=str(axes.scene_variant),
            board_rows=int(axes.board_rows),
            board_cols=int(axes.board_cols),
            target_clear_count=int(target_clear_count),
        )

    return TetrisObjectivePlan(
        attempt_namespace=f"games.tetris.line_clear.{int(target_clear_count)}",
        prompt_query_key=PROMPT_QUERY_KEY,
        answer_hint_key="answer_hint_line_clear_count",
        annotation_hint_key="annotation_hint_line_clear_count",
        json_example=JSON_EXAMPLE,
        json_example_answer_only=JSON_EXAMPLE_ANSWER_ONLY,
        render_mode=RENDER_MODE_LINE_CLEAR,
        query_params={
            "target_clear_count": int(target_clear_count),
            "target_clear_count_support": [int(value) for value in target_support],
            "target_clear_count_probabilities": dict(target_probabilities),
        },
        construct_attempt=construct_attempt,
    )


@register_task
class GamesTetrisLineClearCountTask:
    """Count the best possible row clears for the visible next piece."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'ranking', 'state_update')
    domain = "games"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_tetris_lifecycle(
            task_id=TASK_ID,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            instance_seed=int(instance_seed),
            params=params,
            max_attempts=int(max_attempts),
            prepare_objective=_prepare_line_clear_objective,
        )


__all__ = ["GamesTetrisLineClearCountTask"]
