"""Sokoban grid mechanics and serialization helpers."""

from __future__ import annotations

from collections import deque
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from .state import Cell, DIRECTIONS, DIRECTION_NAMES


def cell_id(cell: Cell) -> str:
    """Return a stable render-map id for a board cell."""

    return f"cell_r{int(cell[0])}_c{int(cell[1])}"


def box_id(label: str) -> str:
    """Return a stable entity id for one labeled box."""

    return f"box_{label}"


def target_id(label: str) -> str:
    """Return a stable entity id for one labeled target."""

    return f"target_{label}"


def option_id(label: str) -> str:
    """Return a stable entity id for one image-visible option."""

    return f"option_{label}"


def add_cells(a: Cell, b: Cell) -> Cell:
    """Add two grid coordinates."""

    return int(a[0] + b[0]), int(a[1] + b[1])


def inside_board(rows: int, cols: int, cell: Cell) -> bool:
    """Return whether a cell is inside the row/column bounds."""

    return 0 <= int(cell[0]) < int(rows) and 0 <= int(cell[1]) < int(cols)


def neighbors(rows: int, cols: int, cell: Cell) -> Iterable[Cell]:
    """Yield four-neighbor cells inside the board."""

    for delta in DIRECTIONS.values():
        nxt = add_cells(cell, delta)
        if inside_board(rows, cols, nxt):
            yield nxt


def connected_components(rows: int, cols: int, walls: set[Cell]) -> List[List[Cell]]:
    """Return connected open-cell components for a wall set."""

    seen: set[Cell] = set()
    components: List[List[Cell]] = []
    for row in range(int(rows)):
        for col in range(int(cols)):
            cell = (row, col)
            if cell in walls or cell in seen:
                continue
            queue: deque[Cell] = deque([cell])
            seen.add(cell)
            comp: List[Cell] = []
            while queue:
                cur = queue.popleft()
                comp.append(cur)
                for nxt in neighbors(rows, cols, cur):
                    if nxt not in walls and nxt not in seen:
                        seen.add(nxt)
                        queue.append(nxt)
            components.append(comp)
    return components


def largest_component(rows: int, cols: int, walls: set[Cell]) -> List[Cell]:
    """Return the largest connected open-cell component."""

    components = connected_components(rows, cols, walls)
    return max(components, key=len) if components else []


def shortest_path(passable: set[Cell], start: Cell, goal: Cell) -> List[Cell] | None:
    """Return the shortest four-neighbor path, treating omitted cells as blocked."""

    if start not in passable or goal not in passable:
        return None
    queue: deque[Cell] = deque([start])
    parent: Dict[Cell, Cell | None] = {start: None}
    while queue:
        cur = queue.popleft()
        if cur == goal:
            path: List[Cell] = []
            walk: Cell | None = cur
            while walk is not None:
                path.append(walk)
                walk = parent[walk]
            return list(reversed(path))
        for delta in DIRECTIONS.values():
            nxt = add_cells(cur, delta)
            if nxt in passable and nxt not in parent:
                parent[nxt] = cur
                queue.append(nxt)
    return None


def moves_from_path(path: Sequence[Cell]) -> List[str]:
    """Convert a coordinate path into U/D/L/R moves."""

    moves: List[str] = []
    reverse = {delta: key for key, delta in DIRECTIONS.items()}
    for prev, nxt in zip(path, path[1:]):
        delta = (int(nxt[0] - prev[0]), int(nxt[1] - prev[1]))
        moves.append(str(reverse[delta]))
    return moves


def sequence_text(moves: Sequence[str]) -> str:
    """Return compact option text for a move sequence."""

    return " ".join(str(move) for move in moves)


def sequence_description(moves: Sequence[str]) -> str:
    """Return human-readable direction names for trace metadata."""

    return ", ".join(DIRECTION_NAMES.get(str(move), str(move)) for move in moves)


def simulate_grid_path(passable: set[Cell], start: Cell, moves: Sequence[str]) -> Dict[str, Any]:
    """Simulate a path sequence until it finishes or hits a blocker."""

    cur = tuple(start)
    path = [cur]
    blocked_at = None
    for idx, move in enumerate([str(item) for item in moves], start=1):
        nxt = add_cells(cur, DIRECTIONS[str(move)])
        if nxt not in passable:
            blocked_at = int(idx)
            path.append(nxt)
            break
        cur = nxt
        path.append(cur)
    return {"end": cur, "path": path, "blocked_at_step": blocked_at}


def manhattan(a: Cell, b: Cell) -> int:
    """Return Manhattan distance between two cells."""

    return abs(int(a[0] - b[0])) + abs(int(a[1] - b[1]))


def json_safe(value: Any) -> Any:
    """Convert tuple-rich internal state into JSON-safe trace values."""

    if isinstance(value, tuple):
        return [json_safe(item) for item in value]
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    return value


def entity_description(dataset: Mapping[str, Any]) -> str:
    """Return prompt text for the marked entity in relation objectives."""

    entity_type = str(dataset.get("query_entity_type", "player"))
    entity_label = str(dataset.get("query_entity_label", "player"))
    if entity_type == "box":
        return f"the marked box {entity_label}"
    return "the player"


def query_slot_values(dataset: Mapping[str, Any]) -> Dict[str, str]:
    """Return prompt slots derived from task-owned sampled data."""

    support = dataset.get("relation_support", {})
    if not isinstance(support, Mapping):
        support = {}
    return {
        "query_entity_description": entity_description(dataset),
        "rank_word": str(support.get("rank_word", "requested")),
    }


__all__ = [
    "add_cells",
    "box_id",
    "cell_id",
    "connected_components",
    "entity_description",
    "inside_board",
    "json_safe",
    "largest_component",
    "manhattan",
    "moves_from_path",
    "neighbors",
    "option_id",
    "query_slot_values",
    "sequence_description",
    "sequence_text",
    "shortest_path",
    "simulate_grid_path",
    "target_id",
]
