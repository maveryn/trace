"""Pure graph algorithms for adjacency-scene tasks."""

from __future__ import annotations

from typing import List, Mapping, Sequence, Tuple


def bfs_visit_order(adjacency: Mapping[str, Sequence[str]], source: str) -> Tuple[str, ...]:
    """Return BFS order using the neighbor order shown in the adjacency list."""

    start = str(source)
    visited = {start}
    queue = [start]
    order: List[str] = []
    while queue:
        node = queue.pop(0)
        order.append(str(node))
        for neighbor in adjacency.get(str(node), ()):
            label = str(neighbor)
            if label in visited:
                continue
            visited.add(label)
            queue.append(label)
    return tuple(order)


def dfs_visit_order(adjacency: Mapping[str, Sequence[str]], source: str) -> Tuple[str, ...]:
    """Return recursive DFS preorder using the shown neighbor order."""

    visited: set[str] = set()
    order: List[str] = []

    def _visit(node: str) -> None:
        visited.add(str(node))
        order.append(str(node))
        for neighbor in adjacency.get(str(node), ()):
            label = str(neighbor)
            if label not in visited:
                _visit(label)

    _visit(str(source))
    return tuple(order)


__all__ = [
    "bfs_visit_order",
    "dfs_visit_order",
]
