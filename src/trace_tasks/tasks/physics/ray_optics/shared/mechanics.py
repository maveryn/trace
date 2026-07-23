"""Ray-path construction mechanics for ray-optics scenes."""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

from trace_tasks.tasks.shared.config_defaults import group_default

from .sampling import answer_support
from .state import (
    DIRECTION_STEP,
    RAY_EVENT_BOUNCE,
    RAY_EVENT_TARGET_HIT,
    REFLECT_BACKSLASH,
    REFLECT_SLASH,
    SCENE_MIRROR_COUNT,
    MirrorPlacement,
    RayOpticsTaskDefaults,
    RaySceneLayout,
    TargetPlacement,
)


def simulate_path(
    *,
    board_cols: int,
    board_rows: int,
    source_row: int,
    mirrors: Mapping[Tuple[int, int], str],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]], str]:
    """Return visited cells, hit mirror cells, and the exit direction."""

    col = 0
    row = int(source_row)
    direction = "E"
    visited: List[Tuple[int, int]] = []
    bounces: List[Tuple[int, int]] = []
    seen_states: set[Tuple[int, int, str]] = set()
    max_steps = int(board_cols) * int(board_rows) * 4
    for _ in range(max_steps):
        if not (0 <= int(col) < int(board_cols) and 0 <= int(row) < int(board_rows)):
            break
        state = (int(col), int(row), str(direction))
        if state in seen_states:
            raise ValueError("ray path looped unexpectedly")
        seen_states.add(state)
        visited.append((int(col), int(row)))
        orientation = mirrors.get((int(col), int(row)))
        if orientation is not None:
            bounces.append((int(col), int(row)))
            direction = (
                REFLECT_SLASH[str(direction)]
                if str(orientation) == "/"
                else REFLECT_BACKSLASH[str(direction)]
            )
        step_col, step_row = DIRECTION_STEP[str(direction)]
        col = int(col + step_col)
        row = int(row + step_row)
    return visited, bounces, str(direction)


def compute_path_points(
    *,
    render_defaults: Mapping[str, Any],
    source_row: int,
    path_cells: Sequence[Tuple[int, int]],
    exit_direction: str,
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Return the outside source and exit points for the rendered ray."""

    board_left = float(render_defaults["board_left_px"])
    board_top = float(render_defaults["board_top_px"])
    cell_size = float(render_defaults["cell_size_px"])
    board_cols = int(render_defaults["board_cols"])
    board_rows = int(render_defaults["board_rows"])
    source_center_y = float(board_top + ((int(source_row) + 0.5) * cell_size))
    source_point = (float(board_left - (0.75 * cell_size)), float(source_center_y))
    if not path_cells:
        raise ValueError("ray path must include at least one board cell")
    last_col, last_row = path_cells[-1]
    last_center_x = float(board_left + ((int(last_col) + 0.5) * cell_size))
    last_center_y = float(board_top + ((int(last_row) + 0.5) * cell_size))
    if str(exit_direction) == "E":
        exit_point = (
            float(board_left + (board_cols * cell_size) + (0.55 * cell_size)),
            float(last_center_y),
        )
    elif str(exit_direction) == "W":
        exit_point = (
            float(board_left - (0.55 * cell_size)),
            float(last_center_y),
        )
    elif str(exit_direction) == "N":
        exit_point = (
            float(last_center_x),
            float(board_top - (0.55 * cell_size)),
        )
    else:
        exit_point = (
            float(last_center_x),
            float(board_top + (board_rows * cell_size) + (0.55 * cell_size)),
        )
    return source_point, exit_point


def _choose_two_distinct_rows(rng, *, board_rows: int) -> Tuple[int, int]:
    """Return two distinct interior rows."""

    rows = list(range(1, int(board_rows) - 1))
    rng.shuffle(rows)
    if len(rows) < 2:
        raise ValueError("ray-optics scenes require at least two interior rows")
    return int(rows[0]), int(rows[1])


def construct_hit_mirrors(
    rng,
    *,
    board_cols: int,
    board_rows: int,
    bounce_count: int,
) -> Tuple[int, List[Tuple[int, int, str]]]:
    """Construct one non-self-intersecting sequence of hit mirrors."""

    start_row = int(rng.randint(1, int(board_rows) - 2))
    if int(bounce_count) == 0:
        return int(start_row), []
    if int(bounce_count) == 1:
        col1 = int(rng.randint(2, int(board_cols) - 3))
        exit_vertical = (
            "N"
            if int(start_row) > 1
            and (int(start_row) >= int(board_rows) - 2 or rng.random() < 0.5)
            else "S"
        )
        orientation1 = "/" if str(exit_vertical) == "N" else "\\"
        return int(start_row), [(col1, int(start_row), str(orientation1))]
    if int(bounce_count) == 2:
        col1 = int(rng.randint(2, int(board_cols) - 3))
        row0, row1 = _choose_two_distinct_rows(rng, board_rows=int(board_rows))
        start_row = int(row0)
        vertical_dir = "N" if int(row1) < int(row0) else "S"
        orientation1 = "/" if str(vertical_dir) == "N" else "\\"
        orientation2 = "/" if str(vertical_dir) == "N" else "\\"
        return int(start_row), [
            (col1, int(row0), str(orientation1)),
            (col1, int(row1), str(orientation2)),
        ]
    row0, row1 = _choose_two_distinct_rows(rng, board_rows=int(board_rows))
    start_row = int(row0)
    if int(bounce_count) == 3:
        col1 = int(rng.randint(2, int(board_cols) - 5))
        col2 = int(rng.randint(int(col1) + 2, int(board_cols) - 3))
        first_vertical = "N" if int(row1) < int(row0) else "S"
        second_vertical = (
            "N"
            if int(row1) > 1 and (int(row1) >= int(board_rows) - 2 or rng.random() < 0.5)
            else "S"
        )
        orientation1 = "/" if str(first_vertical) == "N" else "\\"
        orientation2 = "/" if str(first_vertical) == "N" else "\\"
        orientation3 = "/" if str(second_vertical) == "N" else "\\"
        return int(start_row), [
            (col1, int(row0), str(orientation1)),
            (col1, int(row1), str(orientation2)),
            (col2, int(row1), str(orientation3)),
        ]
    rows = list(range(1, int(board_rows) - 1))
    rows.remove(int(row0))
    rows.remove(int(row1))
    rng.shuffle(rows)
    if not rows:
        raise ValueError("multi-bounce optics path needs a third interior row")
    row2 = int(rows[0])
    if int(bounce_count) == 4:
        col1 = int(rng.randint(2, int(board_cols) - 5))
        col2 = int(rng.randint(int(col1) + 2, int(board_cols) - 3))
    else:
        if int(board_cols) < 8:
            raise ValueError("five-bounce optics path needs at least eight columns")
        col1 = int(rng.randint(1, max(1, int(board_cols) - 6)))
        col2 = int(col1 + 2)
        col3 = int(col2 + 2)
    first_vertical = "N" if int(row1) < int(row0) else "S"
    second_vertical = "N" if int(row2) < int(row1) else "S"
    orientation1 = "/" if str(first_vertical) == "N" else "\\"
    orientation2 = "/" if str(first_vertical) == "N" else "\\"
    orientation3 = "/" if str(second_vertical) == "N" else "\\"
    orientation4 = "/" if str(second_vertical) == "N" else "\\"
    mirrors = [
        (col1, int(row0), str(orientation1)),
        (col1, int(row1), str(orientation2)),
        (col2, int(row1), str(orientation3)),
        (col2, int(row2), str(orientation4)),
    ]
    if int(bounce_count) == 4:
        return int(start_row), mirrors
    exit_vertical = "N" if int(row2) >= int(board_rows // 2) else "S"
    orientation5 = "/" if str(exit_vertical) == "N" else "\\"
    mirrors.append((int(col3), int(row2), str(orientation5)))
    return int(start_row), mirrors


def place_unused_mirrors(
    rng,
    *,
    total_mirror_count: int,
    hit_mirrors: Sequence[Tuple[int, int, str]],
    path_cells: Sequence[Tuple[int, int]],
    board_cols: int,
    board_rows: int,
) -> List[Tuple[int, int, str]]:
    """Add off-path mirrors so the scene has the requested mirror count."""

    occupied = {(int(col), int(row)) for col, row in path_cells}
    used = {(int(col), int(row)) for col, row, _ in hit_mirrors}
    available: List[Tuple[int, int]] = [
        (col, row)
        for row in range(1, int(board_rows) - 1)
        for col in range(1, int(board_cols) - 1)
        if (int(col), int(row)) not in occupied
        and (int(col), int(row)) not in used
    ]
    rng.shuffle(available)
    mirrors = list(hit_mirrors)
    while len(mirrors) < int(total_mirror_count) and available:
        col, row = available.pop()
        orientation = "/" if rng.random() < 0.5 else "\\"
        mirrors.append((int(col), int(row), str(orientation)))
    if len(mirrors) != int(total_mirror_count):
        raise ValueError("failed to place all non-hit mirrors off the ray path")
    return mirrors


def choose_targets(
    rng,
    *,
    target_answer: int,
    board_cols: int,
    board_rows: int,
    path_cells: Sequence[Tuple[int, int]],
    mirror_cells: Sequence[Tuple[int, int]],
    target_count_min: int,
    target_count_max: int,
) -> List[TargetPlacement]:
    """Place hit and distractor targets after the path has been fixed."""

    mirror_set = {(int(col), int(row)) for col, row in mirror_cells}
    path_target_cells = [cell for cell in path_cells if tuple(cell) not in mirror_set]
    if int(target_answer) > len(path_target_cells):
        raise ValueError("not enough non-mirror path cells to place requested targets")
    hit_cells: List[Tuple[int, int]] = []
    if int(target_answer) > 0:
        hit_cells = list(path_target_cells)
        rng.shuffle(hit_cells)
        hit_cells = hit_cells[: int(target_answer)]
    occupied = set(path_cells) | mirror_set | set(hit_cells)
    target_total = int(rng.randint(int(target_count_min), int(target_count_max)))
    target_total = max(int(target_total), int(target_answer))
    target_cells = list(hit_cells)
    all_distractor_slots: List[Tuple[int, int]] = [
        (col, row)
        for row in range(int(board_rows))
        for col in range(int(board_cols))
        if (int(col), int(row)) not in occupied
    ]
    path_set = {(int(col), int(row)) for col, row in path_cells}
    preferred_distractor_slots = [
        (col, row)
        for col, row in all_distractor_slots
        if all(
            abs(int(col) - int(path_col)) + abs(int(row) - int(path_row)) >= 2
            for path_col, path_row in path_set
        )
    ]
    distractor_slots = (
        list(preferred_distractor_slots)
        if len(preferred_distractor_slots) >= max(0, int(target_total) - len(target_cells))
        else list(all_distractor_slots)
    )
    rng.shuffle(distractor_slots)
    while len(target_cells) < int(target_total) and distractor_slots:
        target_cells.append(tuple(distractor_slots.pop()))
    if len(target_cells) < int(target_total):
        raise ValueError("failed to place the requested number of optics targets")
    targets: List[TargetPlacement] = []
    hit_cell_set = {tuple(cell) for cell in hit_cells}
    for label_index, (col, row) in enumerate(target_cells, start=1):
        targets.append(
            TargetPlacement(
                target_id=f"target_{int(label_index)}",
                col=int(col),
                row=int(row),
                label=int(label_index),
                hit=(int(col), int(row)) in hit_cell_set,
            )
        )
    return targets


def sample_scene_layout(
    rng,
    *,
    scene_variant: str,
    ray_event_kind: str,
    target_answer: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback_defaults: RayOpticsTaskDefaults,
    render_defaults: Mapping[str, Any],
) -> RaySceneLayout:
    """Construct one ray-trace board matching the requested answer."""

    board_cols = int(render_defaults["board_cols"])
    board_rows = int(render_defaults["board_rows"])
    total_mirror_count = int(SCENE_MIRROR_COUNT[str(scene_variant)])
    target_count_min = int(
        params.get(
            "target_count_min",
            group_default(gen_defaults, "target_count_min", fallback_defaults.target_count_min),
        )
    )
    target_count_max = int(
        params.get(
            "target_count_max",
            group_default(gen_defaults, "target_count_max", fallback_defaults.target_count_max),
        )
    )

    if str(ray_event_kind) == RAY_EVENT_BOUNCE:
        bounce_count = int(target_answer)
    else:
        feasible_bounces = [
            int(value)
            for value in answer_support(
                params=params,
                gen_defaults=gen_defaults,
                fallback_defaults=fallback_defaults,
                scene_variant=str(scene_variant),
                ray_event_kind=RAY_EVENT_BOUNCE,
            )
            if 0 <= int(value) <= int(total_mirror_count)
        ]
        rng.shuffle(feasible_bounces)
        bounce_count = int(feasible_bounces[0] if feasible_bounces else 0)

    source_row, hit_mirrors = construct_hit_mirrors(
        rng,
        board_cols=int(board_cols),
        board_rows=int(board_rows),
        bounce_count=int(bounce_count),
    )
    mirror_map = {
        (int(col), int(row)): str(orientation)
        for col, row, orientation in hit_mirrors
    }
    path_cells, hit_bounce_cells, exit_direction = simulate_path(
        board_cols=int(board_cols),
        board_rows=int(board_rows),
        source_row=int(source_row),
        mirrors=mirror_map,
    )
    if int(len(hit_bounce_cells)) != int(bounce_count):
        raise ValueError("constructed optics path did not realize requested bounce count")

    mirrors = place_unused_mirrors(
        rng,
        total_mirror_count=int(total_mirror_count),
        hit_mirrors=hit_mirrors,
        path_cells=path_cells,
        board_cols=int(board_cols),
        board_rows=int(board_rows),
    )
    mirror_cells = [(int(col), int(row)) for col, row, _ in mirrors]
    if str(ray_event_kind) == RAY_EVENT_TARGET_HIT:
        targets = choose_targets(
            rng,
            target_answer=int(target_answer),
            board_cols=int(board_cols),
            board_rows=int(board_rows),
            path_cells=path_cells,
            mirror_cells=mirror_cells,
            target_count_min=int(target_count_min),
            target_count_max=int(target_count_max),
        )
    else:
        targets = []

    actual_hit_target_count = sum(1 for target in targets if bool(target.hit))
    if (
        str(ray_event_kind) == RAY_EVENT_TARGET_HIT
        and int(actual_hit_target_count) != int(target_answer)
    ):
        raise ValueError("constructed optics targets did not realize requested hit count")

    source_point_px, exit_point_px = compute_path_points(
        render_defaults=render_defaults,
        source_row=int(source_row),
        path_cells=path_cells,
        exit_direction=str(exit_direction),
    )
    if str(ray_event_kind) == RAY_EVENT_BOUNCE:
        annotation_entity_ids = tuple(
            f"bounce_{int(index)}" for index in range(1, len(hit_bounce_cells) + 1)
        )
    else:
        annotation_entity_ids = tuple(
            str(target.target_id) for target in targets if bool(target.hit)
        )
    mirror_specs = tuple(
        MirrorPlacement(
            mirror_id=f"mirror_{int(index)}",
            col=int(col),
            row=int(row),
            orientation=str(orientation),
            hit=(int(col), int(row)) in set(hit_bounce_cells),
        )
        for index, (col, row, orientation) in enumerate(mirrors, start=1)
    )
    return RaySceneLayout(
        scene_variant=str(scene_variant),
        ray_event_kind=str(ray_event_kind),
        target_answer=int(target_answer),
        source_row=int(source_row),
        mirrors=mirror_specs,
        targets=tuple(targets),
        path_cells=tuple((int(col), int(row)) for col, row in path_cells),
        bounce_cells=tuple((int(col), int(row)) for col, row in hit_bounce_cells),
        source_point_px=tuple(float(value) for value in source_point_px),
        exit_point_px=tuple(float(value) for value in exit_point_px),
        annotation_entity_ids=tuple(str(item) for item in annotation_entity_ids),
    )


__all__ = [
    "choose_targets",
    "compute_path_points",
    "construct_hit_mirrors",
    "place_unused_mirrors",
    "sample_scene_layout",
    "simulate_path",
]
