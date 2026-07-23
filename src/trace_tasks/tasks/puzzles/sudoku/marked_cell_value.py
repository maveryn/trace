"""Public Sudoku task for the unique value of a marked empty cell."""

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
    coords_with_solution_value,
    make_sudoku_sample,
)
from .shared.state import DIGITS, DOMAIN, SCENE_ID, Board, SudokuSample

TASK_ID = "task_puzzles__sudoku__marked_cell_value"
_RUNTIME = SudokuTaskRuntime(
    source_id=TASK_ID,
    support_key="marked_cell_value_support",
    include_unit_type=False,
    prompt_query_key="marked_cell_value",
    attempt_namespace="puzzles.sudoku.marked_cell_value",
)


def _sample_marked_cell_value(
    *,
    rng,
    solution: Board,
    target_digit: int,
    scene_variant: str,
    defaults: SudokuDefaults,
) -> SudokuSample:
    """Construct a board where the marked empty cell has one valid digit."""

    marked_cell = tuple(rng.choice(coords_with_solution_value(solution, target_digit)))
    construction = build_marked_peer_construction(
        rng=rng,
        solution=solution,
        marked_cell=marked_cell,
        peer_digits=[digit for digit in DIGITS if int(digit) != int(target_digit)],
        scene_variant=str(scene_variant),
        defaults=defaults,
        exclude_all_peers=False,
    )
    candidates = candidate_digits(construction.board, marked_cell)
    if candidates != (int(target_digit),):
        raise ValueError("constructed Sudoku marked cell is not uniquely solved")
    return make_sudoku_sample(
        board=construction.board,
        solution=solution,
        answer=int(target_digit),
        annotation_coords=(marked_cell,),
        marked_cell=marked_cell,
        construction_mode="unique_marked_cell",
    )


@register_task
class PuzzlesSudokuMarkedCellValueTask:
    """Find the unique digit for a marked empty Sudoku cell."""

    task_id = TASK_ID
    reasoning_operations = ('formula_evaluation', 'matching')
    domain = DOMAIN
    default_dataset_enabled = True
    supported_query_ids = (SINGLE_QUERY_ID,)

    def _build_unique_value_sample(
        self, *, rng, solution: Board, axes: SudokuAxes, defaults: SudokuDefaults
    ) -> SudokuSample:
        """Build the value-specific sample from the resolved answer axis."""

        return _sample_marked_cell_value(
            rng=rng,
            solution=solution,
            target_digit=int(axes.target_answer),
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
        """Generate one marked-cell value puzzle with role-bound annotation."""

        return run_sudoku_single_query_lifecycle(
            runtime=_RUNTIME,
            params=params,
            build_sample=self._build_unique_value_sample,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
        )


__all__ = [
    "TASK_ID",
    "PuzzlesSudokuMarkedCellValueTask",
]
