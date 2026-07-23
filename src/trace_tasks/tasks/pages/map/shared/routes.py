"""Route helpers for printed map scene packages."""

from __future__ import annotations

from collections import deque
from typing import Dict, Iterable, Mapping, Sequence

from .state import Cell, MapSceneCase


def neighbors(cell: Cell, *, grid_cols: int, grid_rows: int) -> Iterable[Cell]:
    """Yield cardinal grid neighbors inside the printed map grid."""

    x, y = int(cell[0]), int(cell[1])
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nx = int(x + dx)
        ny = int(y + dy)
        if 0 <= nx < int(grid_cols) and 0 <= ny < int(grid_rows):
            yield (nx, ny)


def build_adjacency(cells: Sequence[Cell], *, grid_cols: int, grid_rows: int) -> Dict[Cell, tuple[Cell, ...]]:
    """Return cardinal adjacency among selected landmark cells."""

    selected = {tuple(cell) for cell in cells}
    out: Dict[Cell, tuple[Cell, ...]] = {}
    for cell in selected:
        values = [
            neighbor
            for neighbor in neighbors(cell, grid_cols=int(grid_cols), grid_rows=int(grid_rows))
            if neighbor in selected
        ]
        out[cell] = tuple(sorted(values, key=lambda item: (item[1], item[0])))
    return out


def route_between(adjacency: Mapping[Cell, Sequence[Cell]], start: Cell, end: Cell) -> list[Cell]:
    """Find one shortest route between two connected landmark cells."""

    queue: deque[Cell] = deque([start])
    parent: Dict[Cell, Cell | None] = {start: None}
    while queue:
        current = queue.popleft()
        if current == end:
            break
        for neighbor in adjacency[current]:
            if neighbor in parent:
                continue
            parent[neighbor] = current
            queue.append(neighbor)
    if end not in parent:
        return [start]
    path: list[Cell] = []
    current: Cell | None = end
    while current is not None:
        path.append(current)
        current = parent[current]
    return list(reversed(path))


def sample_route(
    *,
    rng,
    cells: Sequence[Cell],
    adjacency: Mapping[Cell, Sequence[Cell]],
    min_edges: int,
    max_edges: int,
) -> list[Cell]:
    """Sample a simple connected route with bounded edge count."""

    cell_list = list(cells)
    for _ in range(300):
        start = cell_list[int(rng.randrange(len(cell_list)))]
        path = [start]
        while len(path) - 1 < int(max_edges):
            choices = [cell for cell in adjacency[path[-1]] if cell not in path]
            if not choices:
                break
            path.append(choices[int(rng.randrange(len(choices)))])
            if len(path) - 1 >= int(min_edges) and rng.random() < 0.36:
                break
        if int(min_edges) <= len(path) - 1 <= int(max_edges):
            return list(path)
    best_path: list[Cell] = [cell_list[0]]
    for start in cell_list:
        for end in cell_list:
            if start == end:
                continue
            path = route_between(adjacency, start, end)
            if len(best_path) < len(path) <= int(max_edges) + 1:
                best_path = path
    if len(best_path) - 1 < int(min_edges):
        raise ValueError("could not sample a feasible map route")
    return list(best_path[: int(max_edges) + 1])


def direction_between(source: Cell, target: Cell) -> str:
    """Return a compass direction for one adjacent route step."""

    dx = int(target[0] - source[0])
    dy = int(target[1] - source[1])
    if dx == 1 and dy == 0:
        return "east"
    if dx == -1 and dy == 0:
        return "west"
    if dx == 0 and dy == -1:
        return "north"
    if dx == 0 and dy == 1:
        return "south"
    raise ValueError(f"route cells are not adjacent: {source} -> {target}")


def format_direction_steps(directions: Sequence[str]) -> str:
    """Return concise direction text while preserving step order."""

    formatted: list[str] = []
    index = 0
    while index < len(directions):
        direction = str(directions[index])
        count = 1
        while index + count < len(directions) and str(directions[index + count]) == direction:
            count += 1
        if count == 1:
            formatted.append(direction)
        else:
            formatted.append(f"{direction} {count} times")
        index += count
    return ", then ".join(formatted)


def ordinal_label(value: int) -> str:
    """Return a human-readable ordinal label for route-step prompts."""

    if 10 <= int(value) % 100 <= 20:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(int(value) % 10, "th")
    return f"{int(value)}{suffix}"


def landmark_ids_for_route(case: MapSceneCase, route: Sequence[Cell]) -> list[str]:
    """Map route cells to rendered landmark ids."""

    return [str(case.cell_to_landmark_id[tuple(cell)]) for cell in route]


def landmark_lookup(case: MapSceneCase) -> Dict[str, Mapping[str, object]]:
    """Return landmark specs keyed by generated landmark id."""

    return {str(spec["landmark_id"]): dict(spec) for spec in case.landmark_specs}


def landmark_labels_for_ids(case: MapSceneCase, landmark_ids: Sequence[str]) -> list[str]:
    """Return visible labels for route landmark ids."""

    lookup = landmark_lookup(case)
    return [str(lookup[str(landmark_id)]["landmark_label"]) for landmark_id in landmark_ids]


def landmark_bbox_ids_for_ids(case: MapSceneCase, landmark_ids: Sequence[str]) -> list[str]:
    """Return landmark bbox ids for ordered route landmark ids."""

    lookup = landmark_lookup(case)
    return [
        str(lookup[str(landmark_id)]["landmark_bbox_id"])
        for landmark_id in landmark_ids
    ]


__all__ = [
    "build_adjacency",
    "direction_between",
    "format_direction_steps",
    "landmark_ids_for_route",
    "landmark_bbox_ids_for_ids",
    "landmark_labels_for_ids",
    "landmark_lookup",
    "neighbors",
    "ordinal_label",
    "route_between",
    "sample_route",
]
