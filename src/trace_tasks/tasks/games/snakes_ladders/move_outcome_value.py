"""Final-square value after one shown Snakes and Ladders die roll."""

from __future__ import annotations

import json
from typing import Any, Mapping

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import SnakesLaddersLifecycleTask, SnakesLaddersObjective, run_snakes_ladders_task
from .shared.annotations import bbox_map_annotation_artifacts
from .shared.defaults import DEFAULTS
from .shared.rules import board_last_square, square_to_cell_id
from .shared.sampling import construct_single_roll_outcome_sample, move_jump_probability, select_integer_axis
from .shared.state import SUPPORTED_DIE_VALUES, SnakesLaddersAxes


TASK_ID = "task_games__snakes_ladders__move_outcome_value"
PROMPT_QUERY_KEY = "move_outcome_value"
SUPPORTED_QUERY_IDS = (DEFAULT_QUERY_ID,)
MOVE_OUTCOME_SUPPORT = tuple(range(7, 50))


def _examples() -> tuple[str, str]:
    annotation = {"start_square": [88, 682, 178, 772], "end_square": [462, 398, 552, 488]}
    return (
        json.dumps({"annotation": annotation, "answer": 31}, separators=(",", ":"), ensure_ascii=False),
        json.dumps({"answer": 31}, separators=(",", ":"), ensure_ascii=False),
    )


def _prepare_objective(
    attempt_seed: int,
    task_params: Mapping[str, Any],
    axes: SnakesLaddersAxes,
    _selected_query: str,
    _query_probabilities: Mapping[str, float],
) -> SnakesLaddersObjective:
    # This public task owns the die value and target final-square axes.
    target, target_probs = select_integer_axis(
        task_params,
        support_key="move_outcome_support",
        explicit_key="target_answer",
        fallback_support=MOVE_OUTCOME_SUPPORT,
        instance_seed=int(attempt_seed),
        namespace=f"{TASK_ID}.target_answer",
        balanced_flag_key="balanced_target_answer_sampling",
        max_value=board_last_square(int(axes.board_side)),
    )
    die_value, die_probs = select_integer_axis(
        task_params,
        support_key="die_value_support",
        explicit_key="die_value",
        fallback_support=SUPPORTED_DIE_VALUES,
        instance_seed=int(attempt_seed),
        namespace=f"{TASK_ID}.die_value",
        balanced_flag_key="balanced_die_value_sampling",
    )
    sample = construct_single_roll_outcome_sample(
        rng=spawn_rng(int(attempt_seed), f"{TASK_ID}.sample"),
        axes=axes,
        target_final=int(target),
        die_value=int(die_value),
        jump_probability=move_jump_probability(task_params, DEFAULTS.move_outcome_jump_probability),
    )
    role_ids = {
        "start_square": square_to_cell_id(int(sample.start_square)),
        "end_square": square_to_cell_id(int(sample.move.final_square if sample.move is not None else sample.answer)),
    }
    json_example, json_example_answer_only = _examples()
    return SnakesLaddersObjective(
        sample=sample,
        answer_gt=TypedValue(type="integer", value=int(sample.answer)),
        prompt_query_key=PROMPT_QUERY_KEY,
        json_example=json_example,
        json_example_answer_only=json_example_answer_only,
        answer_support=[int(value) for value in MOVE_OUTCOME_SUPPORT if int(value) <= board_last_square(int(axes.board_side))],
        build_annotation=lambda rendered: bbox_map_annotation_artifacts(rendered.render_map, role_ids),
        trace_extra_params={"target_answer": int(target), "target_answer_probabilities": dict(target_probs), "die_value": int(die_value), "die_value_probabilities": dict(die_probs)},
        execution_extra={"annotation_role_entity_ids": dict(role_ids)},
        die_value=int(die_value),
        show_roll_panel=True,
    )


@register_task
class GamesSnakesLaddersMoveOutcomeValueTask(SnakesLaddersLifecycleTask):
    task_id = TASK_ID
    reasoning_operations = ('state_update',)
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed: int, *, params: dict, max_attempts: int):
        return run_snakes_ladders_task(self, int(instance_seed), params, int(max_attempts), _prepare_objective, supported_query_ids=SUPPORTED_QUERY_IDS, default_query_id=DEFAULT_QUERY_ID)


__all__ = ["GamesSnakesLaddersMoveOutcomeValueTask"]
