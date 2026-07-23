"""Public Sudoku task for legal candidate count in a marked empty cell."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.base import TaskOutput
from trace_tasks.tasks.registry import register_task

from ._lifecycle import SudokuTaskRuntime, run_sudoku_single_query_lifecycle
from .shared.rules import candidate_digits
from .shared.sampling import (
    SudokuAxes,
    SudokuDefaults,
    build_marked_peer_construction,
    make_sudoku_sample,
)
from .shared.state import DIGITS, DOMAIN, SCENE_ID, Board, SIZE, SudokuSample

TASK_ID = "task_puzzles__sudoku__marked_cell_candidate_count"
_RUNTIME = SudokuTaskRuntime(
    source_id=TASK_ID,
    support_key="marked_cell_candidate_count_support",
    include_unit_type=False,
    prompt_query_key="marked_cell_candidate_count",
    attempt_namespace="puzzles.sudoku.marked_cell_candidate_count",
)


@register_task
class PuzzlesSudokuMarkedCellCandidateCountTask:
    """Count legal candidate digits for a marked empty Sudoku cell."""

    task_id = TASK_ID
    reasoning_operations = ('counting',)
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = (SINGLE_QUERY_ID,)

    def _build_candidate_count_sample(
        self, *, rng, solution: Board, axes: SudokuAxes, defaults: SudokuDefaults
    ) -> SudokuSample:
        """Build the candidate-count sample from the resolved answer axis."""

        return _sample_marked_cell_candidate_count(
            rng=rng,
            solution=solution,
            target_count=int(axes.target_answer),
            scene_variant=str(axes.scene_variant),
            defaults=defaults,
        )

    def generate(
        self,
        instance_seed: int,
        *,
        params: Dict[str, Any],
        max_attempts: int,
    ) -> TaskOutput:
        """Generate one marked-cell candidate-count Sudoku puzzle."""

        return run_sudoku_single_query_lifecycle(
            runtime=_RUNTIME,
            params=params,
            build_sample=self._build_candidate_count_sample,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
        )


def _sample_marked_cell_candidate_count(
    *,
    rng,
    solution: Board,
    target_count: int,
    scene_variant: str,
    defaults: SudokuDefaults,
) -> SudokuSample:
    """Construct a board where the marked cell has the target candidate count."""

    if target_count < 1 or target_count > len(DIGITS):
        raise ValueError(f"unsupported Sudoku candidate count: {target_count}")
    marked_cell = (int(rng.randrange(SIZE)), int(rng.randrange(SIZE)))
    solution_digit = int(solution[marked_cell[0]][marked_cell[1]])
    eliminable_digits = [
        int(digit) for digit in DIGITS if int(digit) != int(solution_digit)
    ]
    eliminated_digits = set(
        int(digit)
        for digit in rng.sample(
            eliminable_digits,
            k=int(len(DIGITS) - target_count),
        )
    )
    construction = build_marked_peer_construction(
        rng=rng,
        solution=solution,
        marked_cell=marked_cell,
        peer_digits=tuple(eliminated_digits),
        scene_variant=str(scene_variant),
        defaults=defaults,
        exclude_all_peers=True,
    )
    candidates = candidate_digits(construction.board, marked_cell)
    if len(candidates) != int(target_count):
        raise ValueError("constructed Sudoku marked cell has wrong candidate count")
    return make_sudoku_sample(
        board=construction.board,
        solution=solution,
        answer=int(target_count),
        annotation_coords=(marked_cell,),
        marked_cell=marked_cell,
        construction_mode="marked_cell_candidate_count",
    )


__all__ = [
    "TASK_ID",
    "PuzzlesSudokuMarkedCellCandidateCountTask",
]
