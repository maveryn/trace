"""Public Sudoku task for finding one wrong filled cell."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import SudokuOptionAxes, run_sudoku_single_query_option_lifecycle
from .shared.rules import peer_coords
from .shared.sampling import (
    SudokuDefaults,
    add_random_solution_givens,
    freeze_board,
    make_sudoku_sample,
    mutable_empty_board,
    target_visible_count,
)
from .shared.state import DOMAIN, Board, Coord, SIZE, SudokuSample

TASK_ID = "task_puzzles__sudoku__mistake_cell_label"
PROMPT_QUERY_KEY = "mistake_cell_label"
ATTEMPT_NAMESPACE = "puzzles.sudoku.mistake_cell_label"


def _option_specs_for_mistake(
    *,
    labels: Sequence[str],
    answer_label: str,
    wrong_coord: Coord,
    other_option_coords: Sequence[Coord],
    board: Board,
    solution: Board,
    conflict_peer: Coord,
) -> tuple[dict[str, Any], ...]:
    """Build labeled filled-cell specs for the mistake task."""

    by_label: dict[str, Coord] = {str(answer_label): wrong_coord}
    other_iter = iter(tuple(other_option_coords))
    for label in labels:
        if str(label) == str(answer_label):
            continue
        by_label[str(label)] = next(other_iter)

    specs: list[dict[str, Any]] = []
    for label in labels:
        coord = by_label[str(label)]
        shown_value = int(board[int(coord[0])][int(coord[1])])
        correct_value = int(solution[int(coord[0])][int(coord[1])])
        specs.append(
            {
                "label": str(label),
                "row": int(coord[0]),
                "col": int(coord[1]),
                "value": int(shown_value),
                "correct_value": int(correct_value),
                "is_correct": bool(str(label) == str(answer_label)),
                "is_wrong_cell": bool(shown_value != correct_value),
                "conflict_peer": (
                    [int(conflict_peer[0]), int(conflict_peer[1])]
                    if str(label) == str(answer_label)
                    else None
                ),
            }
        )
    return tuple(specs)


def _sample_mistake_cell(
    *,
    rng,
    solution: Board,
    axes: SudokuOptionAxes,
    defaults: SudokuDefaults,
) -> SudokuSample:
    """Construct a board with one wrong lettered filled cell."""

    wrong_coord = (int(rng.randrange(SIZE)), int(rng.randrange(SIZE)))
    peer_candidates = [
        coord
        for coord in peer_coords(wrong_coord)
        if int(solution[int(coord[0])][int(coord[1])])
        != int(solution[int(wrong_coord[0])][int(wrong_coord[1])])
    ]
    if not peer_candidates:
        raise ValueError("could not find a Sudoku peer for injected mistake")
    conflict_peer = tuple(rng.choice(peer_candidates))
    wrong_value = int(solution[int(conflict_peer[0])][int(conflict_peer[1])])
    wrong_value_peers = {
        coord
        for coord in peer_coords(wrong_coord)
        if int(solution[int(coord[0])][int(coord[1])]) == wrong_value
    }
    forbidden_options = {wrong_coord, conflict_peer, *wrong_value_peers}
    other_option_candidates = [
        (row, col)
        for row in range(SIZE)
        for col in range(SIZE)
        if (row, col) not in forbidden_options
    ]
    other_option_coords = tuple(rng.sample(other_option_candidates, k=3))

    board = mutable_empty_board()
    board[int(wrong_coord[0])][int(wrong_coord[1])] = int(wrong_value)
    board[int(conflict_peer[0])][int(conflict_peer[1])] = int(wrong_value)
    for coord in other_option_coords:
        board[int(coord[0])][int(coord[1])] = int(
            solution[int(coord[0])][int(coord[1])]
        )

    target_visible = target_visible_count(
        rng=rng,
        scene_variant=str(axes.scene_variant),
        defaults=defaults,
        minimum_floor=sum(1 for row in board for cell in row if int(cell) != 0),
    )
    add_random_solution_givens(
        rng=rng,
        board=board,
        solution=solution,
        excluded_coords=(wrong_coord,),
        target_visible_count=int(target_visible),
    )
    frozen = freeze_board(board)
    conflict_witnesses = [
        coord
        for coord in peer_coords(wrong_coord)
        if int(frozen[int(coord[0])][int(coord[1])]) == wrong_value
    ]
    if conflict_peer not in set(conflict_witnesses):
        raise ValueError("injected mistake lost its visible conflict witness")
    option_specs = _option_specs_for_mistake(
        labels=axes.option_labels,
        answer_label=str(axes.answer_label),
        wrong_coord=wrong_coord,
        other_option_coords=other_option_coords,
        board=frozen,
        solution=solution,
        conflict_peer=conflict_peer,
    )
    wrong_labels = [
        str(spec["label"]) for spec in option_specs if spec["is_wrong_cell"]
    ]
    if wrong_labels != [str(axes.answer_label)]:
        raise ValueError("constructed mistake task does not have one wrong option")

    return make_sudoku_sample(
        board=frozen,
        solution=solution,
        answer=str(axes.answer_label),
        annotation_coords=(wrong_coord,),
        option_specs=option_specs,
        correct_option_label=str(axes.answer_label),
        construction_mode="mistake_cell_label",
    )


@register_task
class PuzzlesSudokuMistakeCellLabelTask:
    """Choose the lettered filled cell whose digit must be changed."""

    task_id = TASK_ID
    reasoning_operations = ('matching',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = (SINGLE_QUERY_ID,)

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate one wrong-cell option-letter Sudoku puzzle."""

        task_params = dict(params)
        output = run_sudoku_single_query_option_lifecycle(
            source_id=TASK_ID,
            prompt_query_key=PROMPT_QUERY_KEY,
            attempt_namespace=ATTEMPT_NAMESPACE,
            build_sample=_sample_mistake_cell,
            params=task_params,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
        )
        return output


__all__ = [
    "TASK_ID",
    "PuzzlesSudokuMistakeCellLabelTask",
]
