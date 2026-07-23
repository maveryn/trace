"""Sampling primitives for pipe-flow repair puzzles."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.core.sampling import integer_range_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default, resolve_required_int_bounds

from .defaults import DEFAULTS
from .rules import (
    block_cells,
    cell_add,
    connected_to_destination,
    direction_between,
    local_cells,
    localize_block,
    normalize_openings,
    option_connecting_rotation_turns,
    option_connects,
    option_signature,
    parse_gap_size_variant,
    parse_grid_size_variant,
    path_openings,
    rotate_local_option,
    rotate_openings,
)
from .state import (
    LABEL_POOL,
    Cell,
    Openings,
    OptionSpec,
    PipeFlowDataset,
    PipeFlowMisrotatedDataset,
    RotatedTileCandidateSpec,
    TileSpec,
)


def _local_internal_edges(gap_size: int) -> tuple[tuple[Cell, str, Cell, str], ...]:
    """Return bidirectional internal edges for a square local option grid."""

    size = parse_gap_size_variant(int(gap_size))
    edges: list[tuple[Cell, str, Cell, str]] = []
    for row in range(size):
        for col in range(size):
            if col + 1 < size:
                edges.append(((row, col), "E", (row, col + 1), "W"))
            if row + 1 < size:
                edges.append(((row, col), "S", (row + 1, col), "N"))
    return tuple(edges)


def _local_boundary_stubs(gap_size: int) -> tuple[tuple[Cell, str], ...]:
    """Return outward-facing boundary openings for a square local option grid."""

    size = parse_gap_size_variant(int(gap_size))
    stubs: list[tuple[Cell, str]] = []
    for col in range(size):
        stubs.append(((0, col), "N"))
        stubs.append(((size - 1, col), "S"))
    for row in range(size):
        stubs.append(((row, 0), "W"))
        stubs.append(((row, size - 1), "E"))
    return tuple(stubs)


def find_path(
    rng,
    *,
    rows: int,
    cols: int,
    min_length: int,
    max_length: int,
) -> tuple[Cell, ...]:
    """Sample a simple start-marker-to-finish-flag path across the grid."""

    for _ in range(260):
        start = (int(rng.randrange(int(rows))), 0)
        goal = (int(rng.randrange(int(rows))), int(cols - 1))
        stack: list[tuple[Cell, list[Cell], set[Cell]]] = [(start, [start], {start})]
        step_limit = max(1, int(rows * cols * 8))
        while stack and step_limit > 0:
            step_limit -= 1
            cell, path, visited = stack.pop()
            if cell == goal and int(min_length) <= len(path) <= int(max_length):
                return tuple(path)
            if len(path) >= int(max_length):
                continue
            directions = ["N", "E", "S", "W"]
            rng.shuffle(directions)
            directions.sort(
                key=lambda direction: (
                    0
                    if direction == "E" and cell[1] < goal[1]
                    else (1 if direction in {"N", "S"} else 2)
                )
            )
            for direction in directions:
                nxt = cell_add(cell, direction)
                if not (0 <= nxt[0] < int(rows) and 0 <= nxt[1] < int(cols)):
                    continue
                if nxt in visited:
                    continue
                if nxt[1] < cell[1] - 1:
                    continue
                new_path = [*path, nxt]
                remaining_min = abs(int(goal[0] - nxt[0])) + abs(int(goal[1] - nxt[1]))
                if len(new_path) + remaining_min > int(max_length):
                    continue
                stack.append((nxt, new_path, {*visited, nxt}))
    raise RuntimeError("failed to sample pipe-flow path")


def select_missing_origin(
    path: tuple[Cell, ...],
    *,
    rows: int,
    cols: int,
    gap_size: int,
    rng,
) -> Cell:
    """Choose an interior missing region that a contiguous path segment crosses."""

    size = parse_gap_size_variant(int(gap_size))
    candidates: list[tuple[int, Cell]] = []
    indexed_path = {tuple(cell): index for index, cell in enumerate(path)}
    for row in range(1, max(1, int(rows) - size)):
        for col in range(1, max(1, int(cols) - size)):
            origin = (int(row), int(col))
            block = set(block_cells(origin, gap_size=size))
            inside_indices = sorted(
                indexed_path[cell]
                for cell in block
                if cell in indexed_path
            )
            if len(inside_indices) < 2:
                continue
            if inside_indices[-1] - inside_indices[0] + 1 != len(inside_indices):
                continue
            if inside_indices[0] <= 0 or inside_indices[-1] >= len(path) - 1:
                continue
            if tuple(path[inside_indices[0] - 1]) in block:
                continue
            if tuple(path[inside_indices[-1] + 1]) in block:
                continue
            candidates.append((len(inside_indices), origin))
    if not candidates:
        raise RuntimeError(f"failed to choose a {size}x{size} pipe-flow missing block")
    max_inside = max(score for score, _origin in candidates)
    preferred = [origin for score, origin in candidates if int(score) == int(max_inside)]
    return tuple(preferred[int(rng.randrange(len(preferred)))])


def add_opening(openings_by_cell: dict[Cell, set[str]], left: Cell, right: Cell) -> None:
    """Add a bidirectional pipe connection between adjacent cells."""

    direction = direction_between(tuple(left), tuple(right))
    opposite = {"N": "S", "S": "N", "E": "W", "W": "E"}
    openings_by_cell.setdefault(tuple(left), set()).add(direction)
    openings_by_cell.setdefault(tuple(right), set()).add(opposite[direction])


def is_boundary_cell(cell: Cell, *, rows: int, cols: int) -> bool:
    """Return whether a cell touches any grid side."""

    row, col = int(cell[0]), int(cell[1])
    return row == 0 or row == int(rows) - 1 or col == 0 or col == int(cols) - 1


def boundary_distance(cell: Cell, *, rows: int, cols: int) -> int:
    """Return Manhattan distance from a cell to the nearest grid side."""

    row, col = int(cell[0]), int(cell[1])
    return min(row, int(rows) - 1 - row, col, int(cols) - 1 - col)


def find_branch_path_to_side(
    *,
    anchor: Cell,
    occupied: set[Cell],
    rows: int,
    cols: int,
    min_length: int,
    max_length: int,
    rng,
) -> tuple[Cell, ...] | None:
    """Find an offshoot path from a main-path anchor to any grid side."""

    anchor = tuple(anchor)
    stack: list[tuple[Cell, list[Cell], set[Cell]]] = [(anchor, [], {anchor})]
    step_limit = max(1, int(rows * cols * 10))
    while stack and step_limit > 0:
        step_limit -= 1
        cell, branch_path, visited = stack.pop()
        if (
            branch_path
            and len(branch_path) >= int(min_length)
            and is_boundary_cell(branch_path[-1], rows=int(rows), cols=int(cols))
        ):
            return tuple(branch_path)
        if len(branch_path) >= int(max_length):
            continue
        directions = ["N", "E", "S", "W"]
        rng.shuffle(directions)
        directions.sort(
            key=lambda direction: boundary_distance(
                cell_add(cell, direction),
                rows=int(rows),
                cols=int(cols),
            )
        )
        for direction in directions:
            nxt = cell_add(cell, direction)
            if not (0 <= nxt[0] < int(rows) and 0 <= nxt[1] < int(cols)):
                continue
            if nxt in occupied or nxt in visited:
                continue
            stack.append((nxt, [*branch_path, nxt], {*visited, nxt}))
    return None


def add_branch_paths(
    openings_by_cell: dict[Cell, set[str]],
    *,
    path: tuple[Cell, ...],
    missing_cells: set[Cell],
    rows: int,
    cols: int,
    branch_count: int,
    branch_length_min: int,
    branch_length_max: int,
    rng,
) -> tuple[tuple[Cell, ...], tuple[Cell, ...]]:
    """Attach visible offshoots to the main path, ending each on a grid side."""

    occupied = {tuple(cell) for cell in path} | set(missing_cells)
    branch_cells: set[Cell] = set()
    branch_terminals: list[Cell] = []
    anchors = [tuple(cell) for cell in path[1:-1] if tuple(cell) not in missing_cells]
    for _ in range(max(0, int(branch_count))):
        rng.shuffle(anchors)
        started = False
        for anchor in anchors:
            max_length = max(int(branch_length_max), int(min(rows, cols) - 1))
            branch_path = find_branch_path_to_side(
                anchor=anchor,
                occupied=set(occupied),
                rows=int(rows),
                cols=int(cols),
                min_length=max(1, int(branch_length_min)),
                max_length=max(1, int(max_length)),
                rng=rng,
            )
            if not branch_path:
                continue
            prev = anchor
            for current in branch_path:
                current = tuple(current)
                if current in occupied:
                    break
                add_opening(openings_by_cell, tuple(prev), current)
                occupied.add(current)
                branch_cells.add(current)
                prev = current
            else:
                branch_terminals.append(tuple(branch_path[-1]))
                started = True
            break
        if not started:
            break
    return tuple(sorted(branch_cells)), tuple(branch_terminals)


def random_local_option(rng, *, gap_size: int) -> dict[Cell, Openings]:
    """Build one internally consistent random square pipe option."""

    size = parse_gap_size_variant(int(gap_size))
    openings: dict[Cell, set[str]] = {cell: set() for cell in local_cells(gap_size=size)}
    for left, left_dir, right, right_dir in _local_internal_edges(size):
        if rng.random() < 0.46:
            openings[left].add(left_dir)
            openings[right].add(right_dir)
    stubs = list(_local_boundary_stubs(size))
    rng.shuffle(stubs)
    max_stubs = min(len(stubs), 5)
    stub_count = int(rng.randint(2, max(2, max_stubs)))
    for cell, direction in stubs[:stub_count]:
        openings[cell].add(direction)
    if sum(1 for value in openings.values() if value) < 2:
        cell, direction = stubs[-1]
        openings[cell].add(direction)
    return {cell: normalize_openings(value) for cell, value in openings.items()}


def _normalized_local_map(
    local_openings: Mapping[Cell, Openings],
    *,
    gap_size: int,
) -> dict[Cell, Openings]:
    """Return a complete normalized local opening map."""

    return {
        cell: normalize_openings(local_openings.get(cell, ()))
        for cell in local_cells(gap_size=int(gap_size))
    }


def _toggle(openings: dict[Cell, set[str]], cell: Cell, direction: str) -> None:
    """Toggle one local-cell opening."""

    if str(direction) in openings[tuple(cell)]:
        openings[tuple(cell)].remove(str(direction))
    else:
        openings[tuple(cell)].add(str(direction))


def _similar_local_option(
    correct_local_openings: Mapping[Cell, Openings],
    rng,
    *,
    gap_size: int,
) -> dict[Cell, Openings]:
    """Return a small mutation of the correct square option."""

    size = parse_gap_size_variant(int(gap_size))
    openings: dict[Cell, set[str]] = {
        cell: set(correct_local_openings.get(cell, ()))
        for cell in local_cells(gap_size=size)
    }
    edits = 1 if rng.random() < 0.72 else 2
    for _ in range(int(edits)):
        internal_edges = _local_internal_edges(size)
        if internal_edges and rng.random() < 0.58:
            left, left_dir, right, right_dir = internal_edges[int(rng.randrange(len(internal_edges)))]
            _toggle(openings, left, left_dir)
            _toggle(openings, right, right_dir)
        else:
            boundary_stubs = _local_boundary_stubs(size)
            cell, direction = boundary_stubs[int(rng.randrange(len(boundary_stubs)))]
            _toggle(openings, cell, direction)
    return {cell: normalize_openings(value) for cell, value in openings.items()}


def _opening_difference_score(
    left: Mapping[Cell, Openings],
    right: Mapping[Cell, Openings],
    *,
    gap_size: int,
) -> int:
    """Score how visually close two local opening maps are."""

    score = 0
    for cell in local_cells(gap_size=int(gap_size)):
        score += len(set(left.get(cell, ())) ^ set(right.get(cell, ())))
    return int(score)


def _is_valid_option_piece(
    local_openings: Mapping[Cell, Openings],
    *,
    gap_size: int,
    min_occupied_cells: int = 2,
) -> bool:
    """Return whether occupied option cells are complete pipe tiles."""

    occupied = 0
    for cell in local_cells(gap_size=int(gap_size)):
        opening_count = len(normalize_openings(local_openings.get(cell, ())))
        if opening_count == 1:
            return False
        if opening_count > 0:
            occupied += 1
    return occupied >= int(min_occupied_cells)


def build_option_specs(
    *,
    correct_local_openings: Mapping[Cell, Openings],
    visible_map: Mapping[Cell, Openings],
    origin: Cell,
    gap_size: int,
    rows: int,
    cols: int,
    start_cell: Cell,
    destination_cell: Cell,
    candidate_labels: tuple[str, ...],
    answer_label: str,
    rng,
) -> tuple[OptionSpec, ...]:
    """Create one in-place correct option and distinct close distractors."""

    size = parse_gap_size_variant(int(gap_size))
    min_occupied_cells = 1 if size == 1 else 2
    labels = tuple(str(label) for label in candidate_labels)
    if str(answer_label) not in set(labels):
        raise ValueError(f"answer_label {answer_label!r} outside candidate labels")
    answer_label_index = int(labels.index(str(answer_label)))
    correct_display_turns = 0
    correct_display_openings = _normalized_local_map(correct_local_openings, gap_size=size)
    correct_occupied_cells = sum(1 for openings in correct_display_openings.values() if openings)
    if not _is_valid_option_piece(
        correct_display_openings,
        gap_size=size,
        min_occupied_cells=max(int(min_occupied_cells), int(correct_occupied_cells)),
    ):
        raise RuntimeError("displayed correct pipe-flow option contains partial pipe cells")
    correct_signature = option_signature(correct_display_openings, gap_size=size)
    correct_connects_in_place = option_connects(
        visible_map=visible_map,
        local_openings=correct_display_openings,
        origin=origin,
        gap_size=size,
        rows=int(rows),
        cols=int(cols),
        start_cell=start_cell,
        destination_cell=destination_cell,
    )
    if not correct_connects_in_place:
        raise RuntimeError("displayed correct pipe-flow option is not solvable as drawn")
    correct_connecting_turns = option_connecting_rotation_turns(
        visible_map=visible_map,
        local_openings=correct_display_openings,
        origin=origin,
        gap_size=size,
        rows=int(rows),
        cols=int(cols),
        start_cell=start_cell,
        destination_cell=destination_cell,
    )

    distractor_candidates: list[tuple[int, int, dict[Cell, Openings]]] = []
    seen = {correct_signature}

    def maybe_add(candidate: Mapping[Cell, Openings]) -> None:
        normalized = _normalized_local_map(candidate, gap_size=size)
        signature = option_signature(normalized, gap_size=size)
        if signature in seen:
            return
        seen.add(signature)
        if not _is_valid_option_piece(
            normalized,
            gap_size=size,
            min_occupied_cells=max(int(min_occupied_cells), int(correct_occupied_cells) - 1),
        ):
            return
        if option_connects(
            visible_map=visible_map,
            local_openings=normalized,
            origin=origin,
            gap_size=size,
            rows=int(rows),
            cols=int(cols),
            start_cell=start_cell,
            destination_cell=destination_cell,
        ):
            return
        score = _opening_difference_score(correct_display_openings, normalized, gap_size=size)
        distractor_candidates.append(
            (int(score), int(rng.randrange(1_000_000_000)), normalized)
        )

    rotated_variants = list(range(1, 4))
    rng.shuffle(rotated_variants)
    for turns in rotated_variants:
        maybe_add(
            rotate_local_option(
                correct_display_openings,
                turns=int(turns),
                gap_size=size,
            )
        )
    for _ in range(1600):
        if len(distractor_candidates) >= max(24, int((len(labels) - 1) * 6)):
            break
        maybe_add(_similar_local_option(correct_display_openings, rng, gap_size=size))
    for _ in range(1200):
        if len(distractor_candidates) >= max(36, int((len(labels) - 1) * 8)):
            break
        maybe_add(random_local_option(rng, gap_size=size))

    distractor_candidates.sort(key=lambda item: (int(item[0]), int(item[1])))
    distractors = [candidate for _score, _tie, candidate in distractor_candidates]
    if len(distractors) < len(labels) - 1:
        raise RuntimeError("failed to build enough pipe-flow option distractors")

    options: list[OptionSpec] = []
    distractor_iter = iter(distractors)
    for option_index, label in enumerate(labels):
        is_correct = bool(option_index == int(answer_label_index))
        local_openings = (
            _normalized_local_map(correct_display_openings, gap_size=size)
            if is_correct
            else next(distractor_iter)
        )
        connects_in_place = option_connects(
            visible_map=visible_map,
            local_openings=local_openings,
            origin=origin,
            gap_size=size,
            rows=int(rows),
            cols=int(cols),
            start_cell=start_cell,
            destination_cell=destination_cell,
        )
        connecting_turns = option_connecting_rotation_turns(
            visible_map=visible_map,
            local_openings=local_openings,
            origin=origin,
            gap_size=size,
            rows=int(rows),
            cols=int(cols),
            start_cell=start_cell,
            destination_cell=destination_cell,
        )
        if bool(is_correct) != bool(connects_in_place):
            raise RuntimeError("pipe-flow option uniqueness check failed")
        if not _is_valid_option_piece(
            local_openings,
            gap_size=size,
            min_occupied_cells=int(min_occupied_cells),
        ):
            raise RuntimeError("pipe-flow option has invalid partial pipe cells")
        options.append(
            OptionSpec(
                option_id=f"option_panel_{int(option_index + 1)}",
                label=str(label),
                local_openings=option_signature(local_openings, gap_size=size),
                is_correct=is_correct,
                connects_in_place=bool(connects_in_place),
                connects_after_rotation_turns=tuple(int(value) for value in connecting_turns),
                display_rotation_turns=int(correct_display_turns if is_correct else 0),
            )
        )
    return tuple(options)


def sample_pipe_flow_dataset(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    grid_size_variant: str,
    gap_size_variant: str,
    scene_variant: str,
    candidate_count: int,
    answer_label: str,
) -> PipeFlowDataset:
    """Sample a complete pipe-flow scene with one unique in-place repair option."""

    rows, cols = parse_grid_size_variant(str(grid_size_variant))
    gap_size = parse_gap_size_variant(str(gap_size_variant))
    path_min, path_max = resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="path_length_min",
        max_key="path_length_max",
        fallback_min=DEFAULTS.path_length_min,
        fallback_max=DEFAULTS.path_length_max,
        context="pipe_flow path length",
    )
    branch_min, branch_max = resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="branch_count_min",
        max_key="branch_count_max",
        fallback_min=DEFAULTS.branch_count_min,
        fallback_max=DEFAULTS.branch_count_max,
        context="pipe_flow branch count",
    )
    branch_length_min, branch_length_max = resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="branch_length_min",
        max_key="branch_length_max",
        fallback_min=DEFAULTS.branch_length_min,
        fallback_max=DEFAULTS.branch_length_max,
        context="pipe_flow branch length",
    )
    labels = tuple(LABEL_POOL[index] for index in range(int(candidate_count)))
    if str(answer_label) not in set(labels):
        raise ValueError(f"answer_label {answer_label!r} outside candidate labels")
    rng = spawn_rng(int(instance_seed), "puzzles.pipe_flow.dataset")

    for _ in range(180):
        path = find_path(
            rng,
            rows=int(rows),
            cols=int(cols),
            min_length=max(8, int(path_min)),
            max_length=min(int(rows * cols), int(path_max)),
        )
        if len(path) < 8:
            continue
        try:
            missing_origin = select_missing_origin(
                path,
                rows=int(rows),
                cols=int(cols),
                gap_size=int(gap_size),
                rng=rng,
            )
        except RuntimeError:
            continue
        missing_cells = set(block_cells(missing_origin, gap_size=int(gap_size)))
        path_opening_map = path_openings(tuple(path))
        full_openings: dict[Cell, set[str]] = {
            tuple(cell): set(openings)
            for cell, openings in path_opening_map.items()
        }
        branch_count, _branch_count_probabilities = integer_range_choice(
            rng,
            int(branch_min),
            int(branch_max),
            weights=group_default(gen_defaults, "branch_count_weights", None),
        )
        branch_cells, branch_terminal_cells = add_branch_paths(
            full_openings,
            path=tuple(path),
            missing_cells=set(missing_cells),
            rows=int(rows),
            cols=int(cols),
            branch_count=int(branch_count),
            branch_length_min=int(branch_length_min),
            branch_length_max=int(branch_length_max),
            rng=rng,
        )
        required_branch_terminals = max(0, min(int(branch_count), int(branch_min)))
        if (
            int(required_branch_terminals) > 0
            and len(branch_terminal_cells) < int(required_branch_terminals)
        ):
            continue
        full_map: dict[Cell, Openings] = {
            tuple(cell): normalize_openings(openings)
            for cell, openings in full_openings.items()
            if openings
        }
        visible_map = {
            tuple(cell): tuple(openings)
            for cell, openings in full_map.items()
            if tuple(cell) not in missing_cells
        }
        if connected_to_destination(
            visible_map,
            rows=rows,
            cols=cols,
            start_cell=path[0],
            destination_cell=path[-1],
        ):
            continue
        correct_local_openings = localize_block(
            full_map,
            origin=missing_origin,
            gap_size=int(gap_size),
        )
        if not option_connects(
            visible_map=visible_map,
            local_openings=correct_local_openings,
            origin=missing_origin,
            gap_size=int(gap_size),
            rows=int(rows),
            cols=int(cols),
            start_cell=path[0],
            destination_cell=path[-1],
        ):
            continue
        try:
            options = build_option_specs(
                correct_local_openings=correct_local_openings,
                visible_map=visible_map,
                origin=missing_origin,
                gap_size=int(gap_size),
                rows=int(rows),
                cols=int(cols),
                start_cell=path[0],
                destination_cell=path[-1],
                candidate_labels=labels,
                answer_label=str(answer_label),
                rng=rng,
            )
        except RuntimeError:
            continue

        path_set = set(path)
        branch_set = set(branch_cells)
        tiles = tuple(
            TileSpec(
                tile_id=f"tile_r{cell[0]}c{cell[1]}",
                row=int(cell[0]),
                col=int(cell[1]),
                required_openings=tuple(path_opening_map.get(cell, tuple())),
                current_openings=tuple(visible_map[cell]),
                is_path=bool(cell in path_set),
                is_branch=bool(cell in branch_set),
                label="",
            )
            for cell in sorted(visible_map)
        )
        correct_option = next(option for option in options if option.is_correct)
        return PipeFlowDataset(
            rows=int(rows),
            cols=int(cols),
            grid_size_variant=str(grid_size_variant),
            gap_size_variant=str(gap_size_variant),
            gap_size=int(gap_size),
            scene_variant=str(scene_variant),
            path_cells=tuple(path),
            branch_cells=tuple(branch_cells),
            branch_terminal_cells=tuple(branch_terminal_cells),
            start_cell=tuple(path[0]),
            destination_cell=tuple(path[-1]),
            missing_origin=tuple(missing_origin),
            missing_cells=tuple(sorted(missing_cells)),
            missing_region_id="missing_region",
            correct_option_panel_id=str(correct_option.option_id),
            answer_label=str(answer_label),
            candidate_count=int(candidate_count),
            tiles=tiles,
            options=tuple(options),
        )
    raise RuntimeError("failed to construct unique pipe-flow option puzzle")


def _tile_map_with_rotated_cell(
    tile_map: Mapping[Cell, Openings],
    *,
    cell: Cell,
    turns: int,
) -> dict[Cell, Openings]:
    """Return tile openings after rotating one cell's current pipe."""

    rotated_map = {
        tuple(tile_cell): normalize_openings(openings)
        for tile_cell, openings in tile_map.items()
    }
    rotated_map[tuple(cell)] = rotate_openings(rotated_map.get(tuple(cell), ()), int(turns))
    return {
        tuple(tile_cell): normalize_openings(openings)
        for tile_cell, openings in rotated_map.items()
        if openings
    }


def _repair_rotation_turns(
    tile_map: Mapping[Cell, Openings],
    *,
    cell: Cell,
    rows: int,
    cols: int,
    start_cell: Cell,
    destination_cell: Cell,
) -> tuple[int, ...]:
    """Return rotations of a candidate tile that reconnect start to finish."""

    turns: list[int] = []
    for turn_count in range(1, 4):
        if connected_to_destination(
            _tile_map_with_rotated_cell(
                tile_map,
                cell=tuple(cell),
                turns=int(turn_count),
            ),
            rows=int(rows),
            cols=int(cols),
            start_cell=tuple(start_cell),
            destination_cell=tuple(destination_cell),
        ):
            turns.append(int(turn_count))
    return tuple(turns)


def _wrong_tile_orientations(required_openings: Openings) -> tuple[Openings, ...]:
    """Return distinct non-original rotations for one pipe tile."""

    required = normalize_openings(required_openings)
    wrong: list[Openings] = []
    seen = {required}
    for turn_count in range(1, 4):
        rotated = rotate_openings(required, int(turn_count))
        if rotated in seen:
            continue
        seen.add(rotated)
        wrong.append(rotated)
    return tuple(wrong)


def _build_misrotated_candidates(
    *,
    labels: tuple[str, ...],
    answer_label: str,
    path: tuple[Cell, ...],
    required_map: Mapping[Cell, Openings],
    current_map: Mapping[Cell, Openings],
    correct_cell: Cell,
    rows: int,
    cols: int,
    rng,
) -> tuple[RotatedTileCandidateSpec, ...] | None:
    """Build labeled candidate tiles with exactly one repairable answer."""

    interior_cells = [tuple(cell) for cell in path[1:-1] if tuple(cell) != tuple(correct_cell)]
    rng.shuffle(interior_cells)
    distractors = interior_cells[: max(0, len(labels) - 1)]
    if len(distractors) < len(labels) - 1:
        return None
    candidate_by_label: dict[str, Cell] = {str(answer_label): tuple(correct_cell)}
    distractor_iter = iter(distractors)
    for label in labels:
        if str(label) == str(answer_label):
            continue
        candidate_by_label[str(label)] = tuple(next(distractor_iter))

    candidates: list[RotatedTileCandidateSpec] = []
    for label in labels:
        cell = tuple(candidate_by_label[str(label)])
        repair_turns = _repair_rotation_turns(
            current_map,
            cell=cell,
            rows=int(rows),
            cols=int(cols),
            start_cell=path[0],
            destination_cell=path[-1],
        )
        is_correct = str(label) == str(answer_label)
        if bool(is_correct) != bool(repair_turns):
            return None
        candidates.append(
            RotatedTileCandidateSpec(
                candidate_id=f"candidate_tile_{str(label).lower()}",
                label=str(label),
                tile_id=f"tile_r{cell[0]}c{cell[1]}",
                row=int(cell[0]),
                col=int(cell[1]),
                required_openings=normalize_openings(required_map[cell]),
                current_openings=normalize_openings(current_map[cell]),
                repair_rotation_turns=tuple(int(value) for value in repair_turns),
                is_correct=bool(is_correct),
                connects_after_rotation=bool(repair_turns),
            )
        )
    return tuple(candidates)


def sample_pipe_flow_misrotated_dataset(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    grid_size_variant: str,
    scene_variant: str,
    candidate_count: int,
    answer_label: str,
) -> PipeFlowMisrotatedDataset:
    """Sample a compact pipe-flow scene with one unique misrotated tile."""

    rows, cols = parse_grid_size_variant(str(grid_size_variant))
    path_min, path_max = resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="path_length_min",
        max_key="path_length_max",
        fallback_min=DEFAULTS.path_length_min,
        fallback_max=DEFAULTS.path_length_max,
        context="pipe_flow misrotated path length",
    )
    labels = tuple(LABEL_POOL[index] for index in range(int(candidate_count)))
    if str(answer_label) not in set(labels):
        raise ValueError(f"answer_label {answer_label!r} outside candidate labels")
    rng = spawn_rng(int(instance_seed), "puzzles.pipe_flow.misrotated_dataset")

    for _ in range(240):
        path = find_path(
            rng,
            rows=int(rows),
            cols=int(cols),
            min_length=max(8, int(path_min)),
            max_length=min(int(rows * cols), int(path_max)),
        )
        if len(path) < int(candidate_count) + 2:
            continue
        required_map = path_openings(tuple(path))
        eligible_cells = [
            tuple(cell)
            for cell in path[1:-1]
            if len(required_map.get(tuple(cell), ())) >= 2
            and _wrong_tile_orientations(required_map[tuple(cell)])
        ]
        if len(eligible_cells) < int(candidate_count):
            continue
        rng.shuffle(eligible_cells)
        for correct_cell in eligible_cells:
            wrong_orientations = list(_wrong_tile_orientations(required_map[correct_cell]))
            rng.shuffle(wrong_orientations)
            for wrong_openings in wrong_orientations:
                current_map: dict[Cell, Openings] = {
                    tuple(cell): normalize_openings(openings)
                    for cell, openings in required_map.items()
                }
                current_map[tuple(correct_cell)] = normalize_openings(wrong_openings)
                if connected_to_destination(
                    current_map,
                    rows=int(rows),
                    cols=int(cols),
                    start_cell=path[0],
                    destination_cell=path[-1],
                ):
                    continue
                candidates = _build_misrotated_candidates(
                    labels=labels,
                    answer_label=str(answer_label),
                    path=tuple(path),
                    required_map=required_map,
                    current_map=current_map,
                    correct_cell=tuple(correct_cell),
                    rows=int(rows),
                    cols=int(cols),
                    rng=rng,
                )
                if candidates is None:
                    continue
                path_set = set(path)
                labels_by_cell = {
                    (int(candidate.row), int(candidate.col)): str(candidate.label)
                    for candidate in candidates
                }
                tiles = tuple(
                    TileSpec(
                        tile_id=f"tile_r{cell[0]}c{cell[1]}",
                        row=int(cell[0]),
                        col=int(cell[1]),
                        required_openings=tuple(required_map.get(cell, tuple())),
                        current_openings=tuple(current_map[cell]),
                        is_path=bool(cell in path_set),
                        is_branch=False,
                        label=str(labels_by_cell.get(tuple(cell), "")),
                    )
                    for cell in sorted(current_map)
                )
                correct_candidate = next(
                    candidate for candidate in candidates if candidate.is_correct
                )
                return PipeFlowMisrotatedDataset(
                    rows=int(rows),
                    cols=int(cols),
                    grid_size_variant=str(grid_size_variant),
                    scene_variant=str(scene_variant),
                    path_cells=tuple(path),
                    branch_cells=tuple(),
                    branch_terminal_cells=tuple(),
                    start_cell=tuple(path[0]),
                    destination_cell=tuple(path[-1]),
                    answer_label=str(answer_label),
                    candidate_count=int(candidate_count),
                    misrotated_tile_id=str(correct_candidate.tile_id),
                    tiles=tiles,
                    candidates=tuple(candidates),
                )
    raise RuntimeError("failed to construct unique pipe-flow misrotated-tile puzzle")
