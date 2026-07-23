"""Count cells reachable from a start cell through open cells."""

from __future__ import annotations

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.registry import register_task
from trace_tasks.tasks.shared.config_defaults import (
    load_scene_generation_rendering_prompt_defaults,
    resolve_required_int_bounds,
)
from trace_tasks.tasks.shared.fixed_query import DEFAULT_QUERY_ID

from ._lifecycle import run_single_query_cell_board_task
from .shared.sampling import (
    all_coords,
    grow_connected_region,
    sample_answer,
    sample_dimensions,
)
from .shared.state import CellBoardCase, OPEN_COLOR, SCENE_ID, START_COLOR, WALL_COLOR
from .shared.topology import bfs_distances, coords_with_neighbors, sort_coords

TASK_ID = "task_puzzles__cell_board__reachable_region_size"
QUERY_ID = DEFAULT_QUERY_ID
SUPPORTED_QUERY_IDS = (QUERY_ID,)
PROMPT_QUERY_KEY = "reachable_region_size"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _ = load_scene_generation_rendering_prompt_defaults(
    "puzzles", SCENE_ID, task_id=TASK_ID
)


def _distractor_bounds(*, params) -> tuple[int, int]:
    """Resolve how many disconnected passable cells should distract the task."""

    lo, hi = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key="distractor_open_min",
        max_key="distractor_open_max",
        fallback_min=4,
        fallback_max=10,
        context="reachable-region distractor passable cells",
    )
    return max(0, int(lo)), max(0, int(hi))


def _diagonal_touch_candidates(
    *,
    region,
    rows: int,
    cols: int,
    blocked,
) -> list[tuple[int, int]]:
    """Return cells that touch the reachable region diagonally but not by side."""

    unavailable = {(int(row), int(col)) for row, col in blocked}
    side_blocked = coords_with_neighbors(region, rows=int(rows), cols=int(cols))
    candidates = set()
    for row, col in region:
        for dr, dc in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            coord = (int(row) + dr, int(col) + dc)
            if not (0 <= coord[0] < int(rows) and 0 <= coord[1] < int(cols)):
                continue
            if coord in unavailable or coord in side_blocked:
                continue
            candidates.add(coord)
    return sort_coords(candidates)


def _grow_disconnected_distractor(
    *,
    rng,
    rows: int,
    cols: int,
    start,
    size: int,
    existing_passable,
) -> list[tuple[int, int]]:
    """Grow one passable distractor component disconnected from existing passable cells."""

    blocked = coords_with_neighbors(
        existing_passable,
        rows=int(rows),
        cols=int(cols),
    )
    return grow_connected_region(
        rng=rng,
        rows=int(rows),
        cols=int(cols),
        size=int(size),
        start=start,
        blocked=blocked,
    )


def _build_region_size_case(*, instance_seed: int, params) -> CellBoardCase:
    """Construct an exact-size reachable open region from one start cell."""

    rows, cols = sample_dimensions(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        namespace="puzzles.cell_board.region_size.dimensions",
        fallback_rows_min=6,
        fallback_rows_max=8,
        fallback_cols_min=6,
        fallback_cols_max=8,
    )
    answer, support = sample_answer(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        namespace="puzzles.cell_board.region_size.answer",
        fallback_min=1,
        fallback_max=8,
    )
    rng = spawn_rng(int(instance_seed), "puzzles.cell_board.region_size.case")
    distractor_min, distractor_max = _distractor_bounds(params=params)
    region = grow_connected_region(
        rng=rng,
        rows=int(rows),
        cols=int(cols),
        size=int(answer),
    )
    start = rng.choice(region)
    board = {coord: WALL_COLOR for coord in all_coords(rows=int(rows), cols=int(cols))}
    for coord in region:
        board[coord] = OPEN_COLOR
    board[start] = START_COLOR

    distractors: list[tuple[int, int]] = []
    diagonal_distractors: list[tuple[int, int]] = []
    target_distractor_count = int(rng.randint(distractor_min, distractor_max))
    diagonal_candidates = _diagonal_touch_candidates(
        region=region,
        rows=int(rows),
        cols=int(cols),
        blocked=region,
    )
    rng.shuffle(diagonal_candidates)
    for candidate in diagonal_candidates:
        remaining = max(1, int(target_distractor_count) - len(distractors))
        sizes = list(range(1, min(3, remaining) + 1))
        rng.shuffle(sizes)
        for component_size in sorted(sizes, reverse=True):
            try:
                component = _grow_disconnected_distractor(
                    rng=rng,
                    rows=int(rows),
                    cols=int(cols),
                    start=candidate,
                    size=int(component_size),
                    existing_passable=region,
                )
            except ValueError:
                continue
            distractors.extend(component)
            diagonal_distractors.extend(component)
            break
        if diagonal_distractors:
            break

    for _ in range(80):
        if len(distractors) >= int(target_distractor_count):
            break
        existing_passable = list(region) + list(distractors)
        unavailable = coords_with_neighbors(
            existing_passable,
            rows=int(rows),
            cols=int(cols),
        )
        candidates = [
            coord
            for coord in all_coords(rows=int(rows), cols=int(cols))
            if coord not in unavailable
        ]
        if not candidates:
            break
        rng.shuffle(candidates)
        remaining = max(1, int(target_distractor_count) - len(distractors))
        size = int(rng.randint(1, min(3, remaining)))
        try:
            component = _grow_disconnected_distractor(
                rng=rng,
                rows=int(rows),
                cols=int(cols),
                start=candidates[0],
                size=int(size),
                existing_passable=existing_passable,
            )
        except ValueError:
            continue
        distractors.extend(component)

    if len(distractors) < int(distractor_min):
        raise ValueError("not enough disconnected passable distractors")
    for coord in distractors:
        board[coord] = OPEN_COLOR
    passable = [
        coord
        for coord, color in board.items()
        if color[0] in {OPEN_COLOR[0], START_COLOR[0]}
    ]
    distances = bfs_distances(start=start, passable=passable, rows=rows, cols=cols)
    if len(distances) != int(answer):
        raise ValueError("reachable region construction changed answer")
    return CellBoardCase(
        rows=int(rows),
        cols=int(cols),
        board_colors=board,
        answer_value=int(answer),
        annotation_kind="bbox_set",
        annotation_coords=tuple(sort_coords(distances.keys())),
        cell_text={start: "S"},
        prompt_task_key="cell_board_topology_query",
        prompt_query_key=PROMPT_QUERY_KEY,
        prompt_slots={},
        execution_trace={
            "start_cell": [int(start[0]), int(start[1])],
            "reachable_cells": [
                [int(row), int(col)] for row, col in sort_coords(distances.keys())
            ],
            "distractor_open_cells": [
                [int(row), int(col)] for row, col in sort_coords(distractors)
            ],
            "diagonal_distractor_cells": [
                [int(row), int(col)]
                for row, col in sort_coords(diagonal_distractors)
            ],
            "passable_cell_count": int(len(passable)),
            "wall_cell_color": str(WALL_COLOR[0]),
            "passable_cell_color": str(OPEN_COLOR[0]),
            "answer_support": [int(value) for value in support],
        },
    )


@register_task
class PuzzlesCellBoardReachableRegionSizeTask:
    """Count the open region reachable by orthogonal moves from start."""

    task_id = TASK_ID
    reasoning_operations = ('filtering', 'counting', 'topology')
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
            construct_case=lambda seed: _build_region_size_case(
                instance_seed=int(seed),
                params=params,
            ),
            namespace="puzzles.cell_board.region_size.query",
        )


__all__ = ["PuzzlesCellBoardReachableRegionSizeTask"]
