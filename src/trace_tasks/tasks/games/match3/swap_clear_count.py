"""Count immediate clears for one marked match-3 swap."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import support_probability_map
from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.support_sampling import (
    resolve_integer_choice,
    resolve_integer_support,
)

from ._lifecycle import (
    Match3ObjectivePlan,
    Match3SingleQueryTaskBase,
    match3_bbox_set_attempt,
    run_match3_registered_task,
)
from .shared.defaults import SCENE_ID
from .shared.prompts import make_match3_prompt_slots
from .shared.rules import cell_entity_id, external_same_color_neighbors_for_clear
from .shared.sampling import all_outcomes_for_board, make_base_board
from .shared.state import Match3Sample, Match3SceneAxes, MoveOutcome, SwapOption

TASK_ID = "task_games__match3__swap_clear_count"
ANSWER_SUPPORT = (0, 3, 4, 5, 6)
PROMPT_SLOTS = make_match3_prompt_slots(
    prompt_query_key="swap_clear_count",
    object_description_key="object_description_match3_grid",
    answer_hint_key="answer_hint_swap_clear_count",
    annotation_hint_key="annotation_hint_swap_clear_count",
    example_annotation=[[214, 180, 262, 228], [270, 180, 318, 228]],
    example_answer=3,
)


def _target_clear_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
) -> tuple[int, tuple[int, ...], dict[str, float]]:
    """Resolve the target clear-count answer with finite-support cycling."""

    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key="swap_clear_count_answer_support",
        fallback=ANSWER_SUPPORT,
    )
    explicit = params.get("target_answer")
    if explicit is not None:
        selected = int(explicit)
        if selected not in set(support):
            raise ValueError(f"unsupported target_answer: {selected}")
        return (
            int(selected),
            tuple(int(value) for value in support),
            support_probability_map(support, selected=int(selected), sort_keys=True),
        )
    sample_cursor = params.get("_sample_cursor")
    if sample_cursor is not None:
        selected = support[abs(int(sample_cursor)) % len(support)]
        return (
            int(selected),
            tuple(int(value) for value in support),
            support_probability_map(support, sort_keys=True),
        )
    selected, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key="swap_clear_count_answer_support",
        explicit_key="target_answer",
        fallback_support=ANSWER_SUPPORT,
        namespace=f"{namespace}.answer",
        balanced_flag_key="balanced_target_answer_sampling",
        namespace_support_permutation=True,
    )
    return (
        int(selected),
        tuple(int(value) for value in support),
        dict(probabilities),
    )


def _select_marked_outcome(
    *,
    board: Sequence[Sequence[str]],
    outcomes: Sequence[MoveOutcome],
    target_clear_count: int,
    rng: Any,
) -> MoveOutcome:
    """Choose one swap outcome with the requested immediate clear count."""

    matching = [
        outcome
        for outcome in outcomes
        if int(outcome.clear_count) == int(target_clear_count)
        and not external_same_color_neighbors_for_clear(board, outcome)
    ]
    if not matching:
        raise ValueError("no unambiguous swap outcome with requested immediate clear count")
    rng.shuffle(matching)
    return matching[0]


def _prepare_swap_clear_count_objective(
    instance_seed: int,
    task_params: Mapping[str, Any],
    _selected_branch: str,
    _branch_probabilities: Mapping[str, float],
    _axes: Match3SceneAxes,
    gen_defaults: Mapping[str, Any],
) -> Match3ObjectivePlan:
    """Resolve target clear-count semantics for one marked swap."""

    namespace = f"{SCENE_ID}.swap_clear_count"
    target_answer, answer_support, answer_probabilities = _target_clear_axis(
        instance_seed=int(instance_seed),
        params=task_params,
        gen_defaults=gen_defaults,
        namespace=namespace,
    )

    def _construct_attempt(rng: Any, axes: Match3SceneAxes):
        """Build one board whose marked swap clears the sampled target count."""

        board_spec = make_base_board(
            rng,
            gen_defaults=gen_defaults,
            namespace=namespace,
            instance_seed=int(instance_seed),
            params=task_params,
            scene_variant=str(axes.scene_variant),
        )
        outcomes = tuple(all_outcomes_for_board(board_spec.board))
        marked_outcome = _select_marked_outcome(
            board=board_spec.board,
            outcomes=outcomes,
            target_clear_count=int(target_answer),
            rng=rng,
        )
        option = SwapOption(label="A", outcome=marked_outcome, is_answer=True)
        sample = Match3Sample(
            scene_variant=str(axes.scene_variant),
            board=board_spec.board,
            answer=int(marked_outcome.clear_count),
            answer_type="integer",
            option_specs=(option,),
            annotation_entity_ids=tuple(
                str(cell_entity_id(coord)) for coord in marked_outcome.cleared_cells
            ),
            metadata={
                **dict(board_spec.metadata),
                "target_answer": int(target_answer),
                "target_answer_probabilities": dict(answer_probabilities),
                "answer_support": [int(value) for value in answer_support],
                "marked_swap_label": str(option.label),
                "marked_swap_entity_id": str(option.entity_id),
                "marked_swap_from_cell": [
                    int(marked_outcome.move.a[0] + 1),
                    int(marked_outcome.move.a[1] + 1),
                ],
                "marked_swap_to_cell": [
                    int(marked_outcome.move.b[0] + 1),
                    int(marked_outcome.move.b[1] + 1),
                ],
                "marked_swap_clear_count": int(marked_outcome.clear_count),
                "marked_swap_run_count": int(marked_outcome.run_count),
                "marked_swap_cleared_cells": [
                    [int(row + 1), int(col + 1)]
                    for row, col in marked_outcome.cleared_cells
                ],
                "marked_swap_runs": [
                    [[int(row + 1), int(col + 1)] for row, col in run]
                    for run in marked_outcome.runs
                ],
            },
        )
        return match3_bbox_set_attempt(
            answer_gt=TypedValue(type="integer", value=int(sample.answer)),
            sample=sample,
            prompt_slots=PROMPT_SLOTS,
            execution_extra={
                "marked_swap_label": str(option.label),
                "marked_swap_clear_count": int(marked_outcome.clear_count),
            },
        )

    return Match3ObjectivePlan(
        attempt_namespace=namespace,
        construct_attempt=_construct_attempt,
    )


@register_task
class GamesMatch3SwapClearCountTask(Match3SingleQueryTaskBase):
    """Count how many gems immediately clear after the marked swap."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'state_update')
    _namespace = f"{SCENE_ID}.swap_clear_count"
    _default_branch = "single"
    _prepare_objective = staticmethod(_prepare_swap_clear_count_objective)

    def generate(
        self,
        instance_seed: int,
        *,
        params: dict[str, Any] | None = None,
        max_attempts: int = 100,
    ) -> TaskOutput:
        return run_match3_registered_task(
            self,
            instance_seed,
            params=params,
            max_attempts=max_attempts,
        )


__all__ = ["ANSWER_SUPPORT", "GamesMatch3SwapClearCountTask", "TASK_ID"]
