"""Topology primitives for orthogonal maze construction and solving."""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

from .state import Cell, CellEdge

def edge_key(a: Cell, b: Cell) -> CellEdge:
    return tuple(sorted((tuple(a), tuple(b))))
def maze_cell_id(cell: Cell) -> str:
    return f"cell_c{int(cell[0])}_r{int(cell[1])}"
def neighbors(cell: Cell, *, rows: int, cols: int) -> List[Cell]:
    col, row = int(cell[0]), int(cell[1])
    result: List[Cell] = []
    if col > 0:
        result.append((col - 1, row))
    if col + 1 < int(cols):
        result.append((col + 1, row))
    if row > 0:
        result.append((col, row - 1))
    if row + 1 < int(rows):
        result.append((col, row + 1))
    return result
def generate_spanning_tree(
    *,
    rows: int,
    cols: int,
    start: Cell,
    rng,
    allowed_cells: set[Cell] | None = None,
) -> Tuple[CellEdge, ...]:
    """Generate a connected orthogonal tree over the allowed maze cells."""

    if allowed_cells is None:
        allowed_cells = {
            (int(col), int(row))
            for row in range(int(rows))
            for col in range(int(cols))
        }
    if tuple(start) not in allowed_cells:
        raise ValueError("maze start cell must be inside the allowed cell set")
    visited = {tuple(start)}
    stack = [tuple(start)]
    edges: List[CellEdge] = []
    while stack:
        current = stack[-1]
        candidates = [
            cell
            for cell in neighbors(current, rows=int(rows), cols=int(cols))
            if tuple(cell) in allowed_cells and tuple(cell) not in visited
        ]
        if not candidates:
            stack.pop()
            continue
        rng.shuffle(candidates)
        nxt = tuple(candidates[0])
        visited.add(nxt)
        stack.append(nxt)
        edges.append(edge_key(current, nxt))
    if len(visited) != len(allowed_cells):
        raise ValueError("maze spanning tree failed to visit all cells")
    return tuple(edges)
def boundary_sides(cell: Cell, *, rows: int, cols: int) -> Tuple[str, ...]:
    col, row = int(cell[0]), int(cell[1])
    sides: List[str] = []
    if row == 0:
        sides.append("top")
    if col == int(cols) - 1:
        sides.append("right")
    if row == int(rows) - 1:
        sides.append("bottom")
    if col == 0:
        sides.append("left")
    return tuple(sides)
def reachable_cells_from_start(*, start: Cell, rows: int, cols: int, edges: Sequence[CellEdge]) -> Tuple[Cell, ...]:
    """Solve the maze graph reachability from the start cell."""

    edge_set = {edge_key(a, b) for a, b in edges}
    frontier = [tuple(start)]
    visited = {tuple(start)}
    while frontier:
        current = frontier.pop(0)
        for nxt in neighbors(current, rows=int(rows), cols=int(cols)):
            if tuple(nxt) in visited:
                continue
            if edge_key(current, tuple(nxt)) not in edge_set:
                continue
            visited.add(tuple(nxt))
            frontier.append(tuple(nxt))
    return tuple(sorted(visited, key=lambda item: (item[1], item[0])))


def shortest_path_between(
    *,
    start: Cell,
    goal: Cell,
    rows: int,
    cols: int,
    edges: Sequence[CellEdge],
) -> Tuple[Cell, ...]:
    """Return one shortest open-corridor path from start to goal."""

    start_cell = tuple(int(value) for value in start)
    goal_cell = tuple(int(value) for value in goal)
    edge_set = {edge_key(a, b) for a, b in edges}
    frontier: List[Cell] = [start_cell]
    parent: dict[Cell, Cell | None] = {start_cell: None}
    while frontier:
        current = frontier.pop(0)
        if tuple(current) == goal_cell:
            break
        for nxt in neighbors(current, rows=int(rows), cols=int(cols)):
            nxt_cell = tuple(nxt)
            if nxt_cell in parent:
                continue
            if edge_key(current, nxt_cell) not in edge_set:
                continue
            parent[nxt_cell] = tuple(current)
            frontier.append(nxt_cell)
    if goal_cell not in parent:
        raise ValueError("goal cell is not reachable from start")

    path: List[Cell] = []
    cursor: Cell | None = goal_cell
    while cursor is not None:
        path.append(tuple(cursor))
        cursor = parent[tuple(cursor)]
    return tuple(reversed(path))


def exit_clockwise_sort_key(exit_spec: Mapping[str, Any], *, rows: int, cols: int) -> Tuple[int, int]:
    col, row = (int(value) for value in exit_spec["cell"])
    side = str(exit_spec["side"])
    if side == "top":
        return (0, col)
    if side == "right":
        return (1, row)
    if side == "bottom":
        return (2, int(cols) - 1 - col)
    return (3, int(rows) - 1 - row)


__all__ = [
    "boundary_sides",
    "edge_key",
    "exit_clockwise_sort_key",
    "generate_spanning_tree",
    "maze_cell_id",
    "neighbors",
    "reachable_cells_from_start",
    "shortest_path_between",
]
