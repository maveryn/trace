"""Count cells that violate a sampled mirror-symmetry axis."""

from __future__ import annotations

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
)
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import run_single_query_cell_board_task
from .shared.sampling import (
    all_coords,
    cycled_named_color,
    sample_answer,
    sample_dimensions,
    sample_palette,
)
from .shared.state import CellBoardCase, SCENE_ID
from .shared.topology import sort_coords

TASK_ID = "task_puzzles__cell_board__symmetry_violation_count"
QUERY_ID = DEFAULT_QUERY_ID
SUPPORTED_QUERY_IDS = (QUERY_ID,)
PROMPT_QUERY_KEY = "symmetry_violation_count"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _ = load_scene_generation_rendering_prompt_defaults(
    "puzzles", SCENE_ID, task_id=TASK_ID
)


def _mirror(coord, *, rows: int, cols: int, axis: str):
    row, col = int(coord[0]), int(coord[1])
    if str(axis) == "vertical":
        return row, int(cols) - 1 - col
    return int(rows) - 1 - row, col


def _counted_side_coords(*, rows: int, cols: int, axis: str):
    if str(axis) == "vertical":
        return [(row, col) for row in range(int(rows)) for col in range(int(cols) // 2)]
    return [(row, col) for row in range(int(rows) // 2) for col in range(int(cols))]


def _build_symmetry_case(*, instance_seed: int, params) -> CellBoardCase:
    """Construct a board with exact mismatches on the counted mirror side."""

    rows, cols = sample_dimensions(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        namespace="puzzles.cell_board.symmetry.dimensions",
        fallback_rows_min=3,
        fallback_rows_max=5,
        fallback_cols_min=3,
        fallback_cols_max=5,
    )
    answer, support = sample_answer(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        namespace="puzzles.cell_board.symmetry.answer",
        fallback_min=1,
        fallback_max=5,
        min_key="target_violation_count_min",
        max_key="target_violation_count_max",
    )
    rng = spawn_rng(int(instance_seed), "puzzles.cell_board.symmetry.case")
    palette = sample_palette(rng=rng, palette_size=4)
    axis = "vertical" if bool(rng.randint(0, 1)) else "horizontal"
    counted_side = "left" if axis == "vertical" else "top"
    counted_side_coords = _counted_side_coords(rows=rows, cols=cols, axis=axis)
    if len(counted_side_coords) < int(answer):
        raise ValueError("not enough mirror pairs for requested violations")
    board = {}
    for coord in all_coords(rows=rows, cols=cols):
        board[coord] = cycled_named_color(
            palette,
            offset=int(coord[0]) + int(coord[1]),
        )
    for coord in counted_side_coords:
        mirror = _mirror(coord, rows=rows, cols=cols, axis=axis)
        color = palette[rng.randrange(len(palette))]
        board[coord] = color
        board[mirror] = color
    rng.shuffle(counted_side_coords)
    violations = set(counted_side_coords[: int(answer)])
    for index, coord in enumerate(violations):
        mirror = _mirror(coord, rows=rows, cols=cols, axis=axis)
        base_name = str(board[mirror][0])
        replacement = next(color for color in palette if str(color[0]) != base_name)
        board[coord] = replacement
    actual = [
        coord
        for coord in counted_side_coords
        if board[coord][0] != board[_mirror(coord, rows=rows, cols=cols, axis=axis)][0]
    ]
    if len(actual) != int(answer):
        raise ValueError("symmetry violation count mismatch")
    actual_sorted = sort_coords(actual)
    violation_pairs = tuple(
        (coord, _mirror(coord, rows=rows, cols=cols, axis=axis))
        for coord in actual_sorted
    )
    return CellBoardCase(
        rows=int(rows),
        cols=int(cols),
        board_colors=board,
        answer_value=int(answer),
        annotation_kind="cell_pair_segment_set",
        annotation_coords=tuple(actual_sorted),
        annotation_coord_pairs=violation_pairs,
        prompt_task_key="cell_board_symmetry_query",
        prompt_query_key=PROMPT_QUERY_KEY,
        prompt_slots={"mirror_axis": str(axis), "counted_side": str(counted_side)},
        execution_trace={
            "mirror_axis": str(axis),
            "counted_side": str(counted_side),
            "violating_cells": [[int(row), int(col)] for row, col in actual_sorted],
            "violating_mirror_pairs": [
                [
                    [int(first[0]), int(first[1])],
                    [int(second[0]), int(second[1])],
                ]
                for first, second in violation_pairs
            ],
            "answer_support": [int(value) for value in support],
        },
    )


@register_task
class PuzzlesCellBoardSymmetryViolationCountTask:
    """Count mismatching cells on one side of a mirror-symmetry check."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'transformation', 'matching')
    domain = "puzzles"
    default_dataset_enabled = True
    supported_query_ids = SUPPORTED_QUERY_IDS

    def generate(self, instance_seed, *, params, max_attempts):
        return run_single_query_cell_board_task(
            task_id=TASK_ID,
            domain=self.domain,
            params=params,
            render_defaults=_RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            max_attempts=int(max_attempts),
            prompt_query_key=PROMPT_QUERY_KEY,
            construct_case=lambda seed: _build_symmetry_case(
                instance_seed=int(seed),
                params=params,
            ),
            namespace="puzzles.cell_board.symmetry.query",
        )


__all__ = ["PuzzlesCellBoardSymmetryViolationCountTask"]
