"""Remaining-square distance from token to finish in Snakes and Ladders."""

from __future__ import annotations

import json
from typing import Any, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import SnakesLaddersLifecycleTask, SnakesLaddersObjective, run_snakes_ladders_task
from .shared.annotations import bbox_map_annotation_artifacts
from .shared.rules import board_last_square, square_to_cell_id, validate_snakes_ladders_sample
from .shared.sampling import add_random_jumps, select_integer_axis, target_jump_counts
from .shared.state import SnakesLaddersAxes, SnakesLaddersSample


TASK_ID = "task_games__snakes_ladders__remaining_to_finish_value"
PROMPT_QUERY_KEY = "remaining_to_finish_value"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
REMAINING_TO_FINISH_SUPPORT = tuple(range(1, 26))
BOARD_SIDE_SUPPORT = (6, 7)


def _examples() -> tuple[str, str]:
    annotation = {"token_square": [356, 598, 446, 688], "finish_square": [542, 132, 632, 222]}
    return (
        json.dumps({"annotation": annotation, "answer": 12}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": 12}, separators=(",", ":"), ensure_ascii=False),
    )


def _construct_remaining_sample(
    *,
    rng: Any,
    axes: SnakesLaddersAxes,
    target_remaining: int,
) -> SnakesLaddersSample:
    """Construct a board where finish-square distance is the answer."""

    board_side = int(axes.board_side)
    last_square = board_last_square(board_side)
    start_square = int(last_square) - int(target_remaining)
    if int(start_square) < 1:
        raise ValueError("remaining distance is incompatible with board side")
    for _attempt in range(140):
        target_ladders, target_snakes = target_jump_counts(int(board_side))
        jumps = add_random_jumps(
            rng=rng,
            jumps=tuple(),
            board_side=int(board_side),
            protected_starts=(int(start_square), int(last_square)),
            target_ladders=int(target_ladders),
            target_snakes=int(target_snakes),
        )
        protected = {int(start_square), int(last_square)}
        if any(int(jump.start_square) in protected or int(jump.end_square) in protected for jump in jumps):
            continue
        sample = SnakesLaddersSample(
            mode=PROMPT_QUERY_KEY,
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            board_side=int(board_side),
            answer=int(target_remaining),
            start_square=int(start_square),
            jumps=tuple(jumps),
            move=None,
            horizon_roll_count=None,
            optimal_route=tuple(),
            annotation_entity_ids=(
                square_to_cell_id(int(start_square)),
                square_to_cell_id(int(last_square)),
            ),
            construction_mode="remaining_to_finish_distance",
        )
        validate_snakes_ladders_sample(sample)
        return sample
    raise ValueError("failed to sample Snakes and Ladders remaining-distance scene")


def _prepare_objective(
    attempt_seed: int,
    task_params: Mapping[str, Any],
    axes: SnakesLaddersAxes,
    _selected_query: str,
    _query_probabilities: Mapping[str, float],
) -> SnakesLaddersObjective:
    """Bind a remaining-square answer and its token/finish witnesses."""

    target, target_probs = select_integer_axis(
        task_params,
        support_key="remaining_to_finish_support",
        explicit_key="target_answer",
        fallback_support=REMAINING_TO_FINISH_SUPPORT,
        instance_seed=int(attempt_seed),
        namespace=f"{TASK_ID}.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
        max_value=board_last_square(int(axes.board_side)) - 1,
    )
    sample = _construct_remaining_sample(
        rng=spawn_rng(int(attempt_seed), f"{TASK_ID}.sample"),
        axes=axes,
        target_remaining=int(target),
    )
    role_ids = {
        "token_square": square_to_cell_id(int(sample.start_square)),
        "finish_square": square_to_cell_id(int(board_last_square(int(axes.board_side)))),
    }
    json_example, json_example_answer_only = _examples()
    return SnakesLaddersObjective(
        sample=sample,
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        prompt_query_key=PROMPT_QUERY_KEY,
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
        answer_support=[int(value) for value in REMAINING_TO_FINISH_SUPPORT],
        build_annotation=lambda rendered: bbox_map_annotation_artifacts(rendered.render_map, role_ids),
        trace_extra_params={
            "target_answer": int(target),
            "target_answer_probabilities": dict(target_probs),
            "remaining_to_finish": int(sample.answer),
        },
        execution_extra={"annotation_role_entity_ids": dict(role_ids), "remaining_to_finish": int(sample.answer)},
        show_roll_panel=False,
    )


@register_task
class GamesSnakesLaddersRemainingToFinishValueTask(SnakesLaddersLifecycleTask):
    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation',)
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        task_params = dict(params)
        task_params.setdefault("board_side_support", list(BOARD_SIDE_SUPPORT))
        return run_snakes_ladders_task(
            self,
            int(instance_seed),
            task_params,
            int(max_attempts),
            _prepare_objective,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id=DEFAULT_QUERY_ID,
        )


__all__ = ["GamesSnakesLaddersRemainingToFinishValueTask"]
