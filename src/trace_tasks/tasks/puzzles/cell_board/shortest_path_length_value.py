"""Find the shortest path length between a start and goal cell."""

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
    grow_connected_region,
    sample_answer,
    sample_dimensions,
)
from .shared.state import (
    CellBoardCase,
    GOAL_COLOR,
    OPEN_COLOR,
    SCENE_ID,
    START_COLOR,
    WALL_COLOR,
)
from .shared.topology import (
    coord_distance,
    coords_with_neighbors,
    four_neighbors,
    reconstruct_shortest_path,
    sort_coords,
)

TASK_ID = "task_puzzles__cell_board__shortest_path_length_value"
QUERY_ID = DEFAULT_QUERY_ID
SUPPORTED_QUERY_IDS = (QUERY_ID,)
PROMPT_QUERY_KEY = "shortest_path_length_value"

_GEN_DEFAULTS, _RENDER_DEFAULTS, _ = load_scene_generation_rendering_prompt_defaults(
    "puzzles", SCENE_ID, task_id=TASK_ID
)


def _valid_path_next(*, candidate, path, rows: int, cols: int) -> bool:
    """Reject cells that would create a shortcut inside the corridor path."""

    if candidate in path:
        return False
    path_set = set(path)
    return all(
        neighbor == path[-1] or neighbor not in path_set
        for neighbor in four_neighbors(candidate, rows=int(rows), cols=int(cols))
    )


def _sample_detour_path(*, rng, rows: int, cols: int, length: int):
    """Sample a unique corridor path whose length exceeds direct grid distance."""

    coords = all_coords(rows=int(rows), cols=int(cols))
    for _ in range(400):
        start = rng.choice(coords)
        path = [start]
        for _step in range(int(length)):
            candidates = [
                neighbor
                for neighbor in four_neighbors(path[-1], rows=int(rows), cols=int(cols))
                if _valid_path_next(
                    candidate=neighbor,
                    path=path,
                    rows=int(rows),
                    cols=int(cols),
                )
            ]
            if not candidates:
                break
            rng.shuffle(candidates)
            path.append(candidates[0])
        if len(path) != int(length) + 1:
            continue
        if coord_distance(path[0], path[-1]) < int(length):
            return path
    raise ValueError("could not sample non-Manhattan shortest path")


def _path_neighbors_on_path(*, coord, path_set, rows: int, cols: int):
    """Return existing path cells that touch coord orthogonally."""

    return [
        neighbor
        for neighbor in four_neighbors(coord, rows=int(rows), cols=int(cols))
        if neighbor in path_set
    ]


def _sample_alternate_bulge(*, rng, rows: int, cols: int, path):
    """Add a two-cell side loop around one path edge without shortening it."""

    path_list = list(path)
    path_set = set(path_list)
    edge_indices = list(range(len(path_list) - 1))
    rng.shuffle(edge_indices)
    for index in edge_indices:
        start = path_list[index]
        end = path_list[index + 1]
        dr = int(end[0]) - int(start[0])
        dc = int(end[1]) - int(start[1])
        if abs(dr) + abs(dc) != 1:
            continue
        side_offsets = [(0, 1), (0, -1)] if dr else [(1, 0), (-1, 0)]
        rng.shuffle(side_offsets)
        for sr, sc in side_offsets:
            first = (int(start[0]) + int(sr), int(start[1]) + int(sc))
            second = (int(end[0]) + int(sr), int(end[1]) + int(sc))
            if not (
                0 <= first[0] < int(rows)
                and 0 <= first[1] < int(cols)
                and 0 <= second[0] < int(rows)
                and 0 <= second[1] < int(cols)
            ):
                continue
            if first in path_set or second in path_set:
                continue
            if coord_distance(first, second) != 1:
                continue
            first_path_neighbors = _path_neighbors_on_path(
                coord=first,
                path_set=path_set,
                rows=int(rows),
                cols=int(cols),
            )
            second_path_neighbors = _path_neighbors_on_path(
                coord=second,
                path_set=path_set,
                rows=int(rows),
                cols=int(cols),
            )
            if first_path_neighbors != [start] or second_path_neighbors != [end]:
                continue
            return [first, second], (start, end)
    raise ValueError("could not sample alternate S-to-G side loop")


def _add_disconnected_passable_distractors(*, rng, rows: int, cols: int, path):
    """Add light-cell distractor components that are not reachable from S."""

    distractors: list[tuple[int, int]] = []
    target_count = int(rng.randint(4, 10))
    for _ in range(100):
        if len(distractors) >= target_count:
            break
        existing_passable = list(path) + list(distractors)
        blocked = coords_with_neighbors(
            existing_passable,
            rows=int(rows),
            cols=int(cols),
        )
        candidates = [
            coord
            for coord in all_coords(rows=int(rows), cols=int(cols))
            if coord not in blocked
        ]
        if not candidates:
            break
        rng.shuffle(candidates)
        remaining = max(1, target_count - len(distractors))
        component_size = int(rng.randint(1, min(3, remaining)))
        try:
            component = grow_connected_region(
                rng=rng,
                rows=int(rows),
                cols=int(cols),
                size=component_size,
                start=candidates[0],
                blocked=blocked,
            )
        except ValueError:
            continue
        distractors.extend(component)
    if len(distractors) < 4:
        raise ValueError("not enough shortest-path passable distractors")
    return sort_coords(distractors)


def _build_shortest_path_case(*, instance_seed: int, params) -> CellBoardCase:
    """Construct a corridor where the unique shortest path has target length."""

    rows, cols = sample_dimensions(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        namespace="puzzles.cell_board.shortest_path.dimensions",
        fallback_rows_min=6,
        fallback_rows_max=8,
        fallback_cols_min=6,
        fallback_cols_max=8,
    )
    answer, support = sample_answer(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=_GEN_DEFAULTS,
        namespace="puzzles.cell_board.shortest_path.answer",
        fallback_min=4,
        fallback_max=8,
        min_key="target_shortest_len_min",
        max_key="target_shortest_len_max",
    )
    rng = spawn_rng(int(instance_seed), "puzzles.cell_board.shortest_path.case")
    path = _sample_detour_path(rng=rng, rows=rows, cols=cols, length=answer)
    alternate_cells, bypassed_edge = _sample_alternate_bulge(
        rng=rng,
        rows=int(rows),
        cols=int(cols),
        path=path,
    )
    start, goal = path[0], path[-1]
    board = {coord: WALL_COLOR for coord in all_coords(rows=rows, cols=cols)}
    for coord in list(path) + list(alternate_cells):
        board[coord] = OPEN_COLOR
    distractors = _add_disconnected_passable_distractors(
        rng=rng,
        rows=int(rows),
        cols=int(cols),
        path=list(path) + list(alternate_cells),
    )
    for coord in distractors:
        board[coord] = OPEN_COLOR
    board[start] = START_COLOR
    board[goal] = GOAL_COLOR
    resolved_path = reconstruct_shortest_path(
        start=start,
        goal=goal,
        passable=set(path) | set(alternate_cells) | set(distractors),
        rows=int(rows),
        cols=int(cols),
    )
    if len(resolved_path) - 1 != int(answer):
        raise ValueError("shortest path length mismatch")
    direct_distance = coord_distance(start, goal)
    if int(direct_distance) >= int(answer):
        raise ValueError("shortest path should require a detour")
    return CellBoardCase(
        rows=int(rows),
        cols=int(cols),
        board_colors=board,
        answer_value=int(answer),
        annotation_kind="segment_set",
        annotation_path=tuple(resolved_path),
        cell_text={start: "S", goal: "G"},
        prompt_task_key="cell_board_topology_query",
        prompt_query_key=PROMPT_QUERY_KEY,
        prompt_slots={},
        execution_trace={
            "start_cell": [int(start[0]), int(start[1])],
            "goal_cell": [int(goal[0]), int(goal[1])],
            "shortest_path_cells": [
                [int(row), int(col)] for row, col in resolved_path
            ],
            "distractor_open_cells": [
                [int(row), int(col)] for row, col in distractors
            ],
            "alternate_path_cells": [
                [int(row), int(col)] for row, col in alternate_cells
            ],
            "alternate_bypassed_edge": [
                [int(row), int(col)] for row, col in bypassed_edge
            ],
            "direct_manhattan_distance": int(direct_distance),
            "passable_cell_count": int(
                len(path) + len(alternate_cells) + len(distractors)
            ),
            "wall_cell_color": str(WALL_COLOR[0]),
            "passable_cell_color": str(OPEN_COLOR[0]),
            "answer_support": [int(value) for value in support],
        },
    )


@register_task
class PuzzlesCellBoardShortestPathLengthValueTask:
    """Return the orthogonal shortest path length from start to goal."""

    task_id = TASK_ID
    reasoning_operations = ('counting', 'ranking', 'topology')
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
            construct_case=lambda seed: _build_shortest_path_case(
                instance_seed=int(seed),
                params=params,
            ),
            namespace="puzzles.cell_board.shortest_path.query",
        )


__all__ = ["PuzzlesCellBoardShortestPathLengthValueTask"]
