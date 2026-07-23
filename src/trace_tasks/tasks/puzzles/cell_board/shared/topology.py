"""Grid topology helpers for rectangular cell-board puzzles."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, List, Mapping, Sequence, Set, Tuple

Coord = Tuple[int, int]


def cell_id(coord: Coord) -> str:
    """Return the stable entity id for one board cell."""

    row, col = int(coord[0]), int(coord[1])
    return f"cell_r{row}_c{col}"


def sort_coords(coords: Iterable[Coord]) -> List[Coord]:
    """Return coordinates in row-major order without duplicates."""

    return sorted(
        {(int(row), int(col)) for row, col in coords},
        key=lambda item: (int(item[0]), int(item[1])),
    )


def four_neighbors(coord: Coord, *, rows: int, cols: int) -> List[Coord]:
    """Return in-bounds orthogonal neighbors for one coordinate."""

    row, col = int(coord[0]), int(coord[1])
    out: List[Coord] = []
    for dr, dc in ((-1, 0), (0, 1), (1, 0), (0, -1)):
        rr, cc = row + dr, col + dc
        if 0 <= rr < int(rows) and 0 <= cc < int(cols):
            out.append((int(rr), int(cc)))
    return out


def coords_with_neighbors(coords: Iterable[Coord], *, rows: int, cols: int) -> Set[Coord]:
    """Return coordinates plus all orthogonal neighbors around them."""

    out: Set[Coord] = {(int(row), int(col)) for row, col in coords}
    for coord in list(out):
        out.update(four_neighbors(coord, rows=int(rows), cols=int(cols)))
    return out


def connected_components(coords: Iterable[Coord], *, rows: int, cols: int) -> List[List[Coord]]:
    """Return 4-neighbor components over the supplied active cells."""

    active: Set[Coord] = {(int(row), int(col)) for row, col in coords}
    components: List[List[Coord]] = []
    while active:
        start = min(active)
        active.remove(start)
        queue = deque([start])
        component = [start]
        while queue:
            current = queue.popleft()
            for neighbor in four_neighbors(current, rows=int(rows), cols=int(cols)):
                if neighbor not in active:
                    continue
                active.remove(neighbor)
                queue.append(neighbor)
                component.append(neighbor)
        components.append(sort_coords(component))
    components.sort(key=lambda comp: (comp[0][0], comp[0][1], len(comp)))
    return components


def bfs_distances(
    *,
    start: Coord,
    passable: Iterable[Coord],
    rows: int,
    cols: int,
) -> Dict[Coord, int]:
    """Return shortest orthogonal distances from start through passable cells."""

    passable_set: Set[Coord] = {(int(row), int(col)) for row, col in passable}
    start_coord = (int(start[0]), int(start[1]))
    if start_coord not in passable_set:
        return {}
    distances: Dict[Coord, int] = {start_coord: 0}
    queue = deque([start_coord])
    while queue:
        current = queue.popleft()
        for neighbor in four_neighbors(current, rows=int(rows), cols=int(cols)):
            if neighbor not in passable_set or neighbor in distances:
                continue
            distances[neighbor] = int(distances[current]) + 1
            queue.append(neighbor)
    return distances


def reconstruct_shortest_path(
    *,
    start: Coord,
    goal: Coord,
    passable: Iterable[Coord],
    rows: int,
    cols: int,
) -> List[Coord]:
    """Return one deterministic shortest path through passable cells."""

    passable_set: Set[Coord] = {(int(row), int(col)) for row, col in passable}
    start_coord = (int(start[0]), int(start[1]))
    goal_coord = (int(goal[0]), int(goal[1]))
    queue = deque([start_coord])
    parents: Dict[Coord, Coord | None] = {start_coord: None}
    while queue:
        current = queue.popleft()
        if current == goal_coord:
            break
        for neighbor in four_neighbors(current, rows=int(rows), cols=int(cols)):
            if neighbor not in passable_set or neighbor in parents:
                continue
            parents[neighbor] = current
            queue.append(neighbor)
    if goal_coord not in parents:
        raise ValueError("goal is not reachable from start")
    path: List[Coord] = []
    current: Coord | None = goal_coord
    while current is not None:
        path.append(current)
        current = parents[current]
    return list(reversed(path))


def manhattan_path_between(a: Coord, b: Coord) -> List[Coord]:
    """Return a deterministic orthogonal path between two aligned coordinates."""

    row, col = int(a[0]), int(a[1])
    target_row, target_col = int(b[0]), int(b[1])
    path: List[Coord] = [(row, col)]
    while col != target_col:
        col += 1 if target_col > col else -1
        path.append((row, col))
    while row != target_row:
        row += 1 if target_row > row else -1
        path.append((row, col))
    return path


def coord_distance(a: Coord, b: Coord) -> int:
    """Return Manhattan grid distance between two coordinates."""

    return abs(int(a[0]) - int(b[0])) + abs(int(a[1]) - int(b[1]))


__all__ = [
    "Coord",
    "bfs_distances",
    "cell_id",
    "connected_components",
    "coords_with_neighbors",
    "coord_distance",
    "four_neighbors",
    "manhattan_path_between",
    "reconstruct_shortest_path",
    "sort_coords",
]
