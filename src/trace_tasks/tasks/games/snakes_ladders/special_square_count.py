"""Count all visible snake-head or ladder-start squares."""

from __future__ import annotations

import json
from typing import Any, Mapping, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task

from ._lifecycle import SnakesLaddersLifecycleTask, SnakesLaddersObjective, run_snakes_ladders_task
from .shared.annotations import bbox_set_annotation_for_entities
from .shared.rules import board_last_square, square_to_cell_id, validate_snakes_ladders_sample
from .shared.sampling import append_jumps_from_allowed_starts, jump_start_entity_ids, select_integer_axis, valid_jump_starts
from .shared.state import SnakesLaddersAxes, SnakesLaddersJump, SnakesLaddersSample


TASK_ID = "task_games__snakes_ladders__special_square_count"
SUPPORTED_QUERY_IDS: Tuple[str, ...] = ("ladder_count", "snake_count")
SPECIAL_SQUARE_COUNT_SUPPORT = (1, 2, 3, 4)


def special_query_kind(query_id: str) -> str:
    """Return the jump kind counted by one public query branch."""

    if str(query_id) == "ladder_count":
        return "ladder"
    if str(query_id) == "snake_count":
        return "snake"
    raise ValueError(f"unsupported special-square query_id: {query_id}")


def _json_examples() -> tuple[str, str]:
    """Return examples whose annotation shape matches this task."""

    annotation = [[188, 682, 278, 772], [374, 496, 464, 586]]
    return (
        json.dumps({"annotation": annotation, "answer": 2}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": 2}, separators=(",", ":"), ensure_ascii=False),
    )


def _sample_special_square_scene(
    *,
    rng: Any,
    axes: SnakesLaddersAxes,
    query_id: str,
    target_answer: int,
) -> SnakesLaddersSample:
    """Construct a board with exactly the requested visible jump count."""

    board_side = int(axes.board_side)
    last_square = board_last_square(board_side)
    query_kind = special_query_kind(str(query_id))
    opposite_kind = "snake" if query_kind == "ladder" else "ladder"
    valid_query_starts = valid_jump_starts(kind=query_kind, board_side=int(board_side))
    valid_opposite_starts = valid_jump_starts(kind=opposite_kind, board_side=int(board_side))

    for _attempt in range(220):
        start_square = int(rng.randint(1, int(last_square) - 1))
        jumps: Tuple[SnakesLaddersJump, ...] = tuple()
        jumps = append_jumps_from_allowed_starts(
            rng=rng,
            jumps=jumps,
            board_side=int(board_side),
            kind=query_kind,
            allowed_starts=tuple(int(value) for value in valid_query_starts if int(value) != int(start_square)),
            count=int(target_answer),
            required=True,
        )
        if jumps is None:
            continue
        allowed_opposite_starts = [int(value) for value in valid_opposite_starts if int(value) != int(start_square)]
        opposite_count = int(rng.randint(1, min(4, max(1, len(allowed_opposite_starts)))))
        jumps = append_jumps_from_allowed_starts(
            rng=rng,
            jumps=jumps,
            board_side=int(board_side),
            kind=opposite_kind,
            allowed_starts=tuple(allowed_opposite_starts),
            count=int(opposite_count),
            required=False,
        )
        if jumps is None:
            continue
        annotation_ids = jump_start_entity_ids(
            jumps=tuple(jumps),
            kind=query_kind,
        )
        if len(annotation_ids) != int(target_answer):
            continue
        if square_to_cell_id(int(start_square)) in annotation_ids:
            continue
        sample = SnakesLaddersSample(
            mode=str(query_id),
            scene_variant=str(axes.scene_variant),
            style_variant=str(axes.style_variant),
            board_side=int(board_side),
            answer=int(target_answer),
            start_square=int(start_square),
            jumps=tuple(jumps),
            move=None,
            horizon_roll_count=None,
            optimal_route=tuple(),
            annotation_entity_ids=tuple(annotation_ids),
            construction_mode="special_square_total_count",
        )
        validate_snakes_ladders_sample(sample)
        return sample
    raise ValueError("failed to sample Snakes and Ladders special-square scene")


def _prepare_objective(
    attempt_seed: int,
    task_params: Mapping[str, Any],
    axes: SnakesLaddersAxes,
    query_id: str,
    query_probabilities: Mapping[str, float],
) -> SnakesLaddersObjective:
    """Bind query kind and target count for the special-square objective."""

    target, target_probs = select_integer_axis(
        task_params,
        support_key="special_square_count_support",
        explicit_key="target_answer",
        fallback_support=SPECIAL_SQUARE_COUNT_SUPPORT,
        instance_seed=int(attempt_seed),
        namespace=f"{TASK_ID}.{str(query_id)}.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
    )
    rng = spawn_rng(int(attempt_seed), f"{TASK_ID}.{str(query_id)}.sample")
    sample = _sample_special_square_scene(
        rng=rng,
        axes=axes,
        query_id=str(query_id),
        target_answer=int(target),
    )
    json_example, json_example_answer_only = _json_examples()
    return SnakesLaddersObjective(
        sample=sample,
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        prompt_query_key=str(query_id),
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
        answer_support=[int(value) for value in SPECIAL_SQUARE_COUNT_SUPPORT],
        build_annotation=lambda rendered: bbox_set_annotation_for_entities(rendered.render_map, sample.annotation_entity_ids),
        trace_extra_params={
            "target_answer": int(target),
            "target_answer_probabilities": dict(target_probs),
            "special_square_kind": special_query_kind(str(query_id)),
        },
        execution_extra={
            "special_square_kind": special_query_kind(str(query_id)),
            "count_scope": "all_visible_jumps_of_kind",
        },
        show_roll_panel=False,
        highlight_token_square=False,
    )


@register_task
class GamesSnakesLaddersSpecialSquareCountTask(SnakesLaddersLifecycleTask):
    """Count visible ladder starts or snake heads."""

    task_id = TASK_ID
    reasoning_operations = ('counting',)
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_snakes_ladders_task(
            self,
            int(instance_seed),
            params,
            int(max_attempts),
            _prepare_objective,
            supported_query_ids=SUPPORTED_QUERY_IDS,
            default_query_id="ladder_count",
        )


__all__ = ["GamesSnakesLaddersSpecialSquareCountTask", "special_query_kind"]
