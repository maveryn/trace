"""Shared dots-and-boxes board rules for games-domain tasks."""

from __future__ import annotations

from typing import Dict, List, Mapping, MutableMapping, Sequence, Tuple

from .state import (
    DotsAndBoxesBoardState,
    DotsAndBoxesBoxInstance,
    DotsAndBoxesEdgeInstance,
    DotsAndBoxesSimulationResult,
)


_BoxCoord = Tuple[int, int]


def _box_id(row_index: int, column_index: int) -> str:
    return f"box_{int(row_index)}_{int(column_index)}"


def _horizontal_edge_id(dot_row: int, dot_col: int) -> str:
    return f"h_{int(dot_row)}_{int(dot_col)}"


def _vertical_edge_id(dot_row: int, dot_col: int) -> str:
    return f"v_{int(dot_row)}_{int(dot_col)}"


def _build_geometry(
    box_rows: int,
    box_cols: int,
) -> Tuple[
    Dict[str, Tuple[str, Tuple[int, int], Tuple[int, int]]],
    Dict[str, Tuple[str, str, str, str]],
    Dict[str, Tuple[str, ...]],
]:
    """Return edge specs, per-box edge ids, and adjacent boxes per edge."""

    edge_specs: Dict[str, Tuple[str, Tuple[int, int], Tuple[int, int]]] = {}
    edge_boxes: Dict[str, List[str]] = {}
    box_edges: Dict[str, Tuple[str, str, str, str]] = {}

    for dot_row in range(int(box_rows) + 1):
        for dot_col in range(int(box_cols)):
            edge_id = _horizontal_edge_id(dot_row, dot_col)
            edge_specs[edge_id] = ("h", (dot_row, dot_col), (dot_row, dot_col + 1))
            edge_boxes[edge_id] = []
    for dot_row in range(int(box_rows)):
        for dot_col in range(int(box_cols) + 1):
            edge_id = _vertical_edge_id(dot_row, dot_col)
            edge_specs[edge_id] = ("v", (dot_row, dot_col), (dot_row + 1, dot_col))
            edge_boxes[edge_id] = []

    for row_index in range(int(box_rows)):
        for column_index in range(int(box_cols)):
            box_id = _box_id(row_index, column_index)
            top = _horizontal_edge_id(row_index, column_index)
            bottom = _horizontal_edge_id(row_index + 1, column_index)
            left = _vertical_edge_id(row_index, column_index)
            right = _vertical_edge_id(row_index, column_index + 1)
            edges = (top, right, bottom, left)
            box_edges[box_id] = edges
            for edge_id in edges:
                edge_boxes[edge_id].append(box_id)

    return edge_specs, box_edges, {edge_id: tuple(box_ids) for edge_id, box_ids in edge_boxes.items()}


def _box_coord_from_id(box_id: str) -> _BoxCoord:
    _, row_str, col_str = str(box_id).split("_")
    return int(row_str), int(col_str)


def _adjacent_boxes(box: _BoxCoord, box_rows: int, box_cols: int) -> List[_BoxCoord]:
    """Return orthogonally adjacent box coordinates inside the board."""

    row_index, column_index = box
    neighbors: List[_BoxCoord] = []
    if row_index > 0:
        neighbors.append((row_index - 1, column_index))
    if row_index + 1 < int(box_rows):
        neighbors.append((row_index + 1, column_index))
    if column_index > 0:
        neighbors.append((row_index, column_index - 1))
    if column_index + 1 < int(box_cols):
        neighbors.append((row_index, column_index + 1))
    return neighbors


def _shared_edge_id(box_a: _BoxCoord, box_b: _BoxCoord, box_edges: Mapping[str, Tuple[str, str, str, str]]) -> str:
    """Return the shared edge id for two adjacent boxes."""

    edges_a = set(box_edges[_box_id(*box_a)])
    edges_b = set(box_edges[_box_id(*box_b)])
    shared = tuple(sorted(edges_a & edges_b))
    if len(shared) != 1:
        raise ValueError(f"boxes {box_a} and {box_b} do not share exactly one edge")
    return str(shared[0])


def _is_boundary_edge(edge_id: str, edge_boxes: Mapping[str, Tuple[str, ...]]) -> bool:
    return len(edge_boxes[str(edge_id)]) == 1


def _extend_induced_path(
    *,
    path: List[_BoxCoord],
    length: int,
    box_rows: int,
    box_cols: int,
    rng,
) -> bool:
    """Extend one self-avoiding induced path in the box grid."""

    if len(path) >= int(length):
        return True
    current = path[-1]
    candidates = [neighbor for neighbor in _adjacent_boxes(current, box_rows, box_cols) if neighbor not in path]
    rng.shuffle(candidates)
    for candidate in candidates:
        if any(abs(candidate[0] - prev[0]) + abs(candidate[1] - prev[1]) == 1 for prev in path[:-1]):
            continue
        path.append(candidate)
        if _extend_induced_path(path=path, length=int(length), box_rows=int(box_rows), box_cols=int(box_cols), rng=rng):
            return True
        path.pop()
    return False


def _sample_induced_box_path(rng, *, box_rows: int, box_cols: int, length: int) -> List[_BoxCoord]:
    """Sample one induced box path of the requested length."""

    cells = [(row_index, column_index) for row_index in range(int(box_rows)) for column_index in range(int(box_cols))]
    for _ in range(256):
        start = cells[int(rng.randrange(len(cells)))]
        path = [start]
        if int(length) == 1:
            return path
        if _extend_induced_path(path=path, length=int(length), box_rows=int(box_rows), box_cols=int(box_cols), rng=rng):
            return list(path)
    raise RuntimeError(f"failed to sample an induced dots-and-boxes path of length {length}")


def _boundary_end_candidates(
    path: Sequence[_BoxCoord],
    *,
    box_edges: Mapping[str, Tuple[str, str, str, str]],
    edge_boxes: Mapping[str, Tuple[str, ...]],
) -> Tuple[str, ...]:
    """Return boundary edges usable as the final capture edge for one path."""

    last_box = path[-1]
    last_edges = list(box_edges[_box_id(*last_box)])
    blocked: set[str] = set()
    if len(path) >= 2:
        blocked.add(_shared_edge_id(path[-2], path[-1], box_edges))
    candidates = [edge_id for edge_id in last_edges if edge_id not in blocked and _is_boundary_edge(edge_id, edge_boxes)]
    return tuple(str(edge_id) for edge_id in candidates)


def _single_box_start_candidates(
    box: _BoxCoord,
    *,
    box_edges: Mapping[str, Tuple[str, str, str, str]],
    edge_boxes: Mapping[str, Tuple[str, ...]],
) -> Tuple[str, ...]:
    """Return boundary edges usable as the highlighted starting edge for a single-box chain."""

    return tuple(
        str(edge_id)
        for edge_id in box_edges[_box_id(*box)]
        if _is_boundary_edge(edge_id, edge_boxes)
    )


def _completed_box_ids(
    *,
    drawn_edge_ids: Sequence[str],
    box_edges: Mapping[str, Tuple[str, str, str, str]],
    captured_box_ids: Sequence[str] = (),
) -> Tuple[str, ...]:
    """Return uncaptured boxes whose four edges are all currently drawn."""

    drawn = set(str(edge_id) for edge_id in drawn_edge_ids)
    captured = set(str(box_id) for box_id in captured_box_ids)
    completed: List[str] = []
    for box_id in sorted(box_edges):
        if box_id in captured:
            continue
        if all(str(edge_id) in drawn for edge_id in box_edges[box_id]):
            completed.append(str(box_id))
    return tuple(completed)


def box_drawn_side_counts(
    *,
    drawn_edge_ids: Sequence[str],
    box_edges: Mapping[str, Tuple[str, str, str, str]],
) -> Dict[str, int]:
    """Return the visible drawn-side count for every box."""

    drawn = set(str(edge_id) for edge_id in drawn_edge_ids)
    return {
        str(box_id): int(sum(1 for edge_id in edges if str(edge_id) in drawn))
        for box_id, edges in sorted(box_edges.items())
    }


def immediate_capture_edge_ids(
    *,
    drawn_edge_ids: Sequence[str],
    box_edges: Mapping[str, Tuple[str, str, str, str]],
) -> Tuple[str, ...]:
    """Return undrawn edges that would complete at least one box immediately."""

    drawn = set(str(edge_id) for edge_id in drawn_edge_ids)
    all_edge_ids = sorted({str(edge_id) for edges in box_edges.values() for edge_id in edges})
    capture_edges: List[str] = []
    for edge_id in all_edge_ids:
        if str(edge_id) in drawn:
            continue
        completed_if_drawn = _completed_box_ids(
            drawn_edge_ids=tuple(sorted(drawn | {str(edge_id)})),
            box_edges=box_edges,
        )
        if completed_if_drawn:
            capture_edges.append(str(edge_id))
    return tuple(capture_edges)


def simulate_forced_capture_turn(
    *,
    box_edges: Mapping[str, Tuple[str, str, str, str]],
    drawn_edge_ids: Sequence[str],
    start_edge_id: str,
) -> DotsAndBoxesSimulationResult:
    """Simulate one dots-and-boxes turn from a highlighted starting edge."""

    initial_completed = _completed_box_ids(drawn_edge_ids=drawn_edge_ids, box_edges=box_edges)
    if initial_completed:
        return DotsAndBoxesSimulationResult(
            is_forced=False,
            capture_count=0,
            captured_box_ids=(),
            move_edge_sequence=(),
            branching_edge_ids=(),
            initial_completed_box_ids=tuple(initial_completed),
        )

    drawn = set(str(edge_id) for edge_id in drawn_edge_ids)
    if str(start_edge_id) in drawn:
        return DotsAndBoxesSimulationResult(
            is_forced=False,
            capture_count=0,
            captured_box_ids=(),
            move_edge_sequence=(),
            branching_edge_ids=(str(start_edge_id),),
            initial_completed_box_ids=(),
        )

    move_edge_sequence: List[str] = []
    captured_box_ids: List[str] = []
    branching_edge_ids: List[str] = []
    current_edge_id = str(start_edge_id)

    while True:
        drawn.add(str(current_edge_id))
        move_edge_sequence.append(str(current_edge_id))
        newly_completed = list(
            _completed_box_ids(
                drawn_edge_ids=tuple(sorted(drawn)),
                box_edges=box_edges,
                captured_box_ids=tuple(captured_box_ids),
            )
        )
        if not newly_completed:
            break
        captured_box_ids.extend(str(box_id) for box_id in newly_completed)

        capture_edges: List[str] = []
        for candidate_edge_id in sorted(set(edge_id for edges in box_edges.values() for edge_id in edges) - drawn):
            completed_if_drawn = _completed_box_ids(
                drawn_edge_ids=tuple(sorted(drawn | {str(candidate_edge_id)})),
                box_edges=box_edges,
                captured_box_ids=tuple(captured_box_ids),
            )
            if completed_if_drawn:
                capture_edges.append(str(candidate_edge_id))
        if not capture_edges:
            break
        if len(capture_edges) != 1:
            branching_edge_ids = list(capture_edges)
            return DotsAndBoxesSimulationResult(
                is_forced=False,
                capture_count=len(captured_box_ids),
                captured_box_ids=tuple(captured_box_ids),
                move_edge_sequence=tuple(move_edge_sequence),
                branching_edge_ids=tuple(branching_edge_ids),
                initial_completed_box_ids=(),
            )
        current_edge_id = str(capture_edges[0])

    return DotsAndBoxesSimulationResult(
        is_forced=bool(captured_box_ids),
        capture_count=len(captured_box_ids),
        captured_box_ids=tuple(captured_box_ids),
        move_edge_sequence=tuple(move_edge_sequence),
        branching_edge_ids=tuple(branching_edge_ids),
        initial_completed_box_ids=(),
    )


def _path_turn_count(path: Sequence[_BoxCoord]) -> int:
    """Count direction changes along one box path."""

    if len(path) < 3:
        return 0
    turns = 0
    for index in range(2, len(path)):
        prev_delta = (path[index - 1][0] - path[index - 2][0], path[index - 1][1] - path[index - 2][1])
        next_delta = (path[index][0] - path[index - 1][0], path[index][1] - path[index - 1][1])
        if prev_delta != next_delta:
            turns += 1
    return int(turns)


def build_dots_and_boxes_board_state(
    *,
    rng,
    target_answer: int,
    box_rows: int,
    box_cols: int,
) -> DotsAndBoxesBoardState:
    """Build one dots-and-boxes board with a forced capture chain of the requested length."""

    edge_specs, box_edges, _edge_boxes = _build_geometry(int(box_rows), int(box_cols))
    target = int(target_answer)
    if target < 1:
        raise ValueError("dots-and-boxes capture task requires target_answer >= 1")

    for _ in range(512):
        path = _sample_induced_box_path(rng, box_rows=int(box_rows), box_cols=int(box_cols), length=int(target))
        if len(path) == 1:
            single_candidates = _single_box_start_candidates(
                path[0],
                box_edges=box_edges,
                edge_boxes=edge_boxes,
            )
            if not single_candidates:
                continue
            start_edge_id = str(single_candidates[int(rng.randrange(len(single_candidates)))])
            end_edge_id = str(start_edge_id)
        else:
            end_candidates = _boundary_end_candidates(path, box_edges=box_edges, edge_boxes=edge_boxes)
            if not end_candidates:
                continue
            start_edge_id = _shared_edge_id(path[0], path[1], box_edges)
            end_edge_id = str(end_candidates[int(rng.randrange(len(end_candidates)))])

        missing_edges_by_box: Dict[str, set[str]] = {}
        for index, box in enumerate(path):
            box_id = _box_id(*box)
            missing_edges: set[str] = set()
            if len(path) == 1:
                missing_edges.add(str(start_edge_id))
            else:
                if index == 0:
                    missing_edges.add(str(start_edge_id))
                elif index == len(path) - 1:
                    missing_edges.add(_shared_edge_id(path[index - 1], path[index], box_edges))
                    missing_edges.add(str(end_edge_id))
                else:
                    missing_edges.add(_shared_edge_id(path[index - 1], path[index], box_edges))
                    missing_edges.add(_shared_edge_id(path[index], path[index + 1], box_edges))
            missing_edges_by_box[box_id] = set(str(edge_id) for edge_id in missing_edges)

        drawn_edge_ids: set[str] = set()
        for box in path:
            box_id = _box_id(*box)
            for edge_id in box_edges[box_id]:
                if str(edge_id) not in missing_edges_by_box[box_id]:
                    drawn_edge_ids.add(str(edge_id))

        simulation = simulate_forced_capture_turn(
            box_edges=box_edges,
            drawn_edge_ids=tuple(sorted(drawn_edge_ids)),
            start_edge_id=str(start_edge_id),
        )
        if not simulation.is_forced:
            continue
        if tuple(simulation.captured_box_ids) != tuple(_box_id(*box) for box in path):
            continue
        if int(simulation.capture_count) != int(target):
            continue

        edges: List[DotsAndBoxesEdgeInstance] = []
        for edge_id in sorted(edge_specs):
            orientation, dot_start, dot_end = edge_specs[edge_id]
            edges.append(
                DotsAndBoxesEdgeInstance(
                    edge_id=str(edge_id),
                    orientation=str(orientation),
                    dot_start=tuple(int(value) for value in dot_start),
                    dot_end=tuple(int(value) for value in dot_end),
                    is_drawn=bool(edge_id in drawn_edge_ids),
                    is_highlighted=bool(str(edge_id) == str(start_edge_id)),
                )
            )

        boxes: List[DotsAndBoxesBoxInstance] = []
        for box_id in sorted(box_edges):
            row_index, column_index = _box_coord_from_id(box_id)
            boxes.append(
                DotsAndBoxesBoxInstance(
                    box_id=str(box_id),
                    row_index=int(row_index),
                    column_index=int(column_index),
                    edge_ids=tuple(str(edge_id) for edge_id in box_edges[box_id]),
                )
            )

        return DotsAndBoxesBoardState(
            box_rows=int(box_rows),
            box_cols=int(box_cols),
            edges=tuple(edges),
            boxes=tuple(boxes),
            highlighted_edge_id=str(start_edge_id),
            drawn_edge_ids=tuple(sorted(drawn_edge_ids)),
            captured_box_ids=tuple(str(box_id) for box_id in simulation.captured_box_ids),
            move_edge_sequence=tuple(str(edge_id) for edge_id in simulation.move_edge_sequence),
            branching_edge_ids=tuple(str(edge_id) for edge_id in simulation.branching_edge_ids),
            path_box_ids=tuple(_box_id(*box) for box in path),
            path_turn_count=int(_path_turn_count(path)),
            target_answer=int(target),
            highlighted_edge_ids=(str(start_edge_id),),
            counted_box_ids=tuple(str(box_id) for box_id in simulation.captured_box_ids),
            counted_edge_ids=(),
            candidate_edge_ids=(),
        )

    raise RuntimeError(f"failed to build a dots-and-boxes board for target {target}")


def _all_edge_ids(box_edges: Mapping[str, Tuple[str, str, str, str]]) -> Tuple[str, ...]:
    """Return all edge ids in stable order."""

    return tuple(sorted({str(edge_id) for edges in box_edges.values() for edge_id in edges}))


def _remove_completed_boxes(
    *,
    rng,
    drawn_edge_ids: set[str],
    box_edges: Mapping[str, Tuple[str, str, str, str]],
) -> None:
    """Remove random edges until no already-completed boxes remain."""

    for _ in range(64):
        completed = _completed_box_ids(drawn_edge_ids=tuple(sorted(drawn_edge_ids)), box_edges=box_edges)
        if not completed:
            return
        for box_id in completed:
            removable = [str(edge_id) for edge_id in box_edges[str(box_id)] if str(edge_id) in drawn_edge_ids]
            if removable:
                drawn_edge_ids.remove(str(removable[int(rng.randrange(len(removable)))]))


def _sample_open_drawn_edges(
    *,
    rng,
    all_edge_ids: Sequence[str],
    box_edges: Mapping[str, Tuple[str, str, str, str]],
) -> set[str]:
    """Sample a visible board state with no already-completed boxes."""

    density = float(rng.uniform(0.18, 0.82))
    drawn_edge_ids = {
        str(edge_id)
        for edge_id in all_edge_ids
        if float(rng.random()) < float(density)
    }
    _remove_completed_boxes(rng=rng, drawn_edge_ids=drawn_edge_ids, box_edges=box_edges)
    return set(str(edge_id) for edge_id in drawn_edge_ids)


def _make_board_state_from_drawn_edges(
    *,
    box_rows: int,
    box_cols: int,
    edge_specs: Mapping[str, Tuple[str, Tuple[int, int], Tuple[int, int]]],
    box_edges: Mapping[str, Tuple[str, str, str, str]],
    drawn_edge_ids: Sequence[str],
    highlighted_edge_ids: Sequence[str],
    counted_box_ids: Sequence[str],
    counted_edge_ids: Sequence[str],
    candidate_edge_ids: Sequence[str],
    target_answer: int,
    box_owner_by_id: Mapping[str, str] | None = None,
    option_label_by_box_id: Sequence[Tuple[str, str]] = (),
    answer_box_id: str = "",
    answer_label: str = "",
) -> DotsAndBoxesBoardState:
    """Build a board state from a sampled static edge set."""

    drawn = set(str(edge_id) for edge_id in drawn_edge_ids)
    highlighted = tuple(str(edge_id) for edge_id in highlighted_edge_ids)
    highlighted_set = set(highlighted)
    edges: List[DotsAndBoxesEdgeInstance] = []
    for edge_id in sorted(edge_specs):
        orientation, dot_start, dot_end = edge_specs[edge_id]
        edges.append(
            DotsAndBoxesEdgeInstance(
                edge_id=str(edge_id),
                orientation=str(orientation),
                dot_start=tuple(int(value) for value in dot_start),
                dot_end=tuple(int(value) for value in dot_end),
                is_drawn=bool(str(edge_id) in drawn),
                is_highlighted=bool(str(edge_id) in highlighted_set),
            )
        )

    boxes: List[DotsAndBoxesBoxInstance] = []
    owners = {str(key): str(value) for key, value in dict(box_owner_by_id or {}).items()}
    for box_id in sorted(box_edges):
        row_index, column_index = _box_coord_from_id(box_id)
        boxes.append(
            DotsAndBoxesBoxInstance(
                box_id=str(box_id),
                row_index=int(row_index),
                column_index=int(column_index),
                edge_ids=tuple(str(edge_id) for edge_id in box_edges[box_id]),
                owner=str(owners.get(str(box_id), "")),
            )
        )

    return DotsAndBoxesBoardState(
        box_rows=int(box_rows),
        box_cols=int(box_cols),
        edges=tuple(edges),
        boxes=tuple(boxes),
        highlighted_edge_id=str(highlighted[0]) if highlighted else "",
        drawn_edge_ids=tuple(sorted(drawn)),
        captured_box_ids=(),
        move_edge_sequence=(),
        branching_edge_ids=(),
        path_box_ids=(),
        path_turn_count=0,
        target_answer=int(target_answer),
        highlighted_edge_ids=tuple(str(edge_id) for edge_id in highlighted),
        counted_box_ids=tuple(str(box_id) for box_id in counted_box_ids),
        counted_edge_ids=tuple(str(edge_id) for edge_id in counted_edge_ids),
        candidate_edge_ids=tuple(str(edge_id) for edge_id in candidate_edge_ids),
        option_label_by_box_id=tuple((str(box_id), str(label)) for box_id, label in option_label_by_box_id),
        answer_box_id=str(answer_box_id),
        answer_label=str(answer_label),
    )


def build_dots_and_boxes_count_board_state(
    *,
    rng,
    count_mode: str,
    owner: str = "",
    target_answer: int,
    box_rows: int,
    box_cols: int,
    candidate_edge_count: int,
) -> DotsAndBoxesBoardState:
    """Build one static dots-and-boxes board for an exact count query."""

    edge_specs, box_edges, _edge_boxes = _build_geometry(int(box_rows), int(box_cols))
    all_edge_ids = _all_edge_ids(box_edges)
    target = int(target_answer)
    query = str(count_mode)
    if target < 0:
        raise ValueError("dots-and-boxes count targets must be non-negative")

    if query == "owned_box":
        target_owner = str(owner).upper()
        if target_owner not in {"A", "B"}:
            raise ValueError("owned_box count mode requires owner A or B")
        other_owner = "B" if target_owner == "A" else "A"
        box_ids = tuple(sorted(box_edges))
        if target > len(box_ids):
            raise ValueError("target_answer cannot exceed the number of boxes")
        for _ in range(8192):
            shuffled = list(box_ids)
            rng.shuffle(shuffled)
            target_box_ids = tuple(sorted(shuffled[:target]))
            remaining_box_ids = tuple(str(box_id) for box_id in shuffled[target:])
            if remaining_box_ids:
                if target == 0:
                    other_count = 1 + int(rng.randrange(min(4, len(remaining_box_ids))))
                else:
                    other_count = int(rng.randrange(min(4, len(remaining_box_ids)) + 1))
            else:
                other_count = 0
            other_box_ids = tuple(sorted(remaining_box_ids[:other_count]))
            owner_by_id = {str(box_id): str(target_owner) for box_id in target_box_ids}
            owner_by_id.update({str(box_id): str(other_owner) for box_id in other_box_ids})
            drawn_edge_ids: set[str] = set()
            for box_id in sorted(owner_by_id):
                drawn_edge_ids.update(str(edge_id) for edge_id in box_edges[str(box_id)])
            completed = set(
                _completed_box_ids(
                    drawn_edge_ids=tuple(sorted(drawn_edge_ids)),
                    box_edges=box_edges,
                )
            )
            if not completed.issubset(set(owner_by_id)):
                continue

            extra_edges = [str(edge_id) for edge_id in all_edge_ids if str(edge_id) not in drawn_edge_ids]
            rng.shuffle(extra_edges)
            for edge_id in extra_edges:
                if float(rng.random()) > 0.35:
                    continue
                candidate = set(drawn_edge_ids)
                candidate.add(str(edge_id))
                completed_if_added = set(
                    _completed_box_ids(
                        drawn_edge_ids=tuple(sorted(candidate)),
                        box_edges=box_edges,
                    )
                )
                if completed_if_added.issubset(set(owner_by_id)):
                    drawn_edge_ids = candidate

            counted_box_ids = tuple(
                str(box_id)
                for box_id, owner in sorted(owner_by_id.items())
                if str(owner) == str(target_owner)
            )
            if len(counted_box_ids) != target:
                continue
            return _make_board_state_from_drawn_edges(
                box_rows=int(box_rows),
                box_cols=int(box_cols),
                edge_specs=edge_specs,
                box_edges=box_edges,
                drawn_edge_ids=tuple(sorted(drawn_edge_ids)),
                highlighted_edge_ids=(),
                counted_box_ids=counted_box_ids,
                counted_edge_ids=(),
                candidate_edge_ids=(),
                target_answer=int(target),
                box_owner_by_id=owner_by_id,
            )
        raise RuntimeError(f"failed to build a dots-and-boxes owned-box board for owner {target_owner} target {target}")

    for _ in range(8192):
        drawn_edge_ids = _sample_open_drawn_edges(rng=rng, all_edge_ids=all_edge_ids, box_edges=box_edges)
        side_counts = box_drawn_side_counts(drawn_edge_ids=tuple(sorted(drawn_edge_ids)), box_edges=box_edges)
        if any(int(count) >= 4 for count in side_counts.values()):
            continue

        if query == "three_sided_box":
            counted_box_ids = tuple(str(box_id) for box_id, count in sorted(side_counts.items()) if int(count) == 3)
            if len(counted_box_ids) != target:
                continue
            return _make_board_state_from_drawn_edges(
                box_rows=int(box_rows),
                box_cols=int(box_cols),
                edge_specs=edge_specs,
                box_edges=box_edges,
                drawn_edge_ids=tuple(sorted(drawn_edge_ids)),
                highlighted_edge_ids=(),
                counted_box_ids=counted_box_ids,
                counted_edge_ids=(),
                candidate_edge_ids=(),
                target_answer=int(target),
            )

        capture_edges = immediate_capture_edge_ids(drawn_edge_ids=tuple(sorted(drawn_edge_ids)), box_edges=box_edges)
        if query == "capture_move":
            if len(capture_edges) != target:
                continue
            return _make_board_state_from_drawn_edges(
                box_rows=int(box_rows),
                box_cols=int(box_cols),
                edge_specs=edge_specs,
                box_edges=box_edges,
                drawn_edge_ids=tuple(sorted(drawn_edge_ids)),
                highlighted_edge_ids=(),
                counted_box_ids=(),
                counted_edge_ids=capture_edges,
                candidate_edge_ids=(),
                target_answer=int(target),
            )

        if query == "highlighted_candidate_capture":
            candidate_count = max(1, int(candidate_edge_count))
            if target > candidate_count:
                raise ValueError("target_answer cannot exceed candidate_edge_count")
            missing_edges = tuple(str(edge_id) for edge_id in all_edge_ids if str(edge_id) not in drawn_edge_ids)
            capture_set = set(str(edge_id) for edge_id in capture_edges)
            non_capture_edges = tuple(str(edge_id) for edge_id in missing_edges if str(edge_id) not in capture_set)
            if len(capture_edges) < target or len(non_capture_edges) < candidate_count - target:
                continue
            capture_pool = list(capture_edges)
            non_capture_pool = list(non_capture_edges)
            rng.shuffle(capture_pool)
            rng.shuffle(non_capture_pool)
            counted_edge_ids = tuple(sorted(str(edge_id) for edge_id in capture_pool[:target]))
            highlighted_edge_ids = tuple(
                sorted(
                    [str(edge_id) for edge_id in capture_pool[:target]]
                    + [str(edge_id) for edge_id in non_capture_pool[: candidate_count - target]]
                )
            )
            return _make_board_state_from_drawn_edges(
                box_rows=int(box_rows),
                box_cols=int(box_cols),
                edge_specs=edge_specs,
                box_edges=box_edges,
                drawn_edge_ids=tuple(sorted(drawn_edge_ids)),
                highlighted_edge_ids=highlighted_edge_ids,
                counted_box_ids=(),
                counted_edge_ids=counted_edge_ids,
                candidate_edge_ids=highlighted_edge_ids,
                target_answer=int(target),
            )

        raise ValueError(f"unsupported dots-and-boxes count mode: {query}")

    raise RuntimeError(f"failed to build a dots-and-boxes board for {query} target {target}")


def build_dots_and_boxes_completable_box_option_board_state(
    *,
    rng,
    answer_label: str,
    box_rows: int,
    box_cols: int,
    option_count: int = 6,
) -> DotsAndBoxesBoardState:
    """Build one board with exactly one labeled box completable in one move."""

    edge_specs, box_edges, edge_boxes = _build_geometry(int(box_rows), int(box_cols))
    all_edge_ids = _all_edge_ids(box_edges)
    labels = tuple(chr(ord("A") + index) for index in range(int(option_count)))
    if str(answer_label) not in labels:
        raise ValueError("answer_label must be one of the visible option labels")

    for _ in range(8192):
        drawn_edge_ids = _sample_open_drawn_edges(rng=rng, all_edge_ids=all_edge_ids, box_edges=box_edges)
        side_counts = box_drawn_side_counts(drawn_edge_ids=tuple(sorted(drawn_edge_ids)), box_edges=box_edges)
        if any(int(count) >= 4 for count in side_counts.values()):
            continue
        completable_box_ids = [str(box_id) for box_id, count in sorted(side_counts.items()) if int(count) == 3]
        if len(completable_box_ids) != 1:
            continue

        correct_box_id = str(completable_box_ids[0])
        answer_index = labels.index(str(answer_label))
        before_count = int(answer_index)
        after_count = int(option_count) - int(answer_index) - 1
        correct_coord = _box_coord_from_id(correct_box_id)
        distractor_box_ids = [str(box_id) for box_id, count in sorted(side_counts.items()) if int(count) < 3]
        before_options = [
            str(box_id)
            for box_id in distractor_box_ids
            if _box_coord_from_id(str(box_id)) < correct_coord
        ]
        after_options = [
            str(box_id)
            for box_id in distractor_box_ids
            if _box_coord_from_id(str(box_id)) > correct_coord
        ]
        if len(before_options) < before_count or len(after_options) < after_count:
            continue
        rng.shuffle(before_options)
        rng.shuffle(after_options)
        selected_box_ids = sorted(
            tuple(before_options[:before_count])
            + (str(correct_box_id),)
            + tuple(after_options[:after_count]),
            key=_box_coord_from_id,
        )
        option_label_by_box_id = tuple(
            (str(box_id), str(label))
            for label, box_id in zip(labels, selected_box_ids)
        )
        return _make_board_state_from_drawn_edges(
            box_rows=int(box_rows),
            box_cols=int(box_cols),
            edge_specs=edge_specs,
            box_edges=box_edges,
            drawn_edge_ids=tuple(sorted(drawn_edge_ids)),
            highlighted_edge_ids=(),
            counted_box_ids=(str(correct_box_id),),
            counted_edge_ids=(),
            candidate_edge_ids=(),
            target_answer=0,
            option_label_by_box_id=option_label_by_box_id,
            answer_box_id=str(correct_box_id),
            answer_label=str(answer_label),
        )

    raise RuntimeError("failed to build a dots-and-boxes completable-box option board")


def annotation_box_ids(board_state: DotsAndBoxesBoardState) -> Tuple[str, ...]:
    """Return the prompt-facing annotation box ids for one dots-and-boxes task."""

    return tuple(str(box_id) for box_id in board_state.captured_box_ids)


__all__ = [
    "DotsAndBoxesBoardState",
    "DotsAndBoxesBoxInstance",
    "DotsAndBoxesEdgeInstance",
    "DotsAndBoxesSimulationResult",
    "box_drawn_side_counts",
    "build_dots_and_boxes_board_state",
    "build_dots_and_boxes_completable_box_option_board_state",
    "build_dots_and_boxes_count_board_state",
    "annotation_box_ids",
    "immediate_capture_edge_ids",
    "simulate_forced_capture_turn",
]
