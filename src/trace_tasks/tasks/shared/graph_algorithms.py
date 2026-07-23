"""Cross-domain graph algorithm helpers.

This module hosts representation-agnostic graph algorithms that operate on
adjacency mappings. Representation-specific adapters (for example blocked
grids or explicit node/edge scene graphs) should stay in separate modules.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, Hashable, List, Mapping, Sequence, TypeVar


NodeT = TypeVar("NodeT", bound=Hashable)


def bfs_dist_count_by_adjacency(
    adjacency: Mapping[NodeT, Sequence[NodeT]],
    *,
    start: NodeT,
) -> tuple[Dict[NodeT, int], Dict[NodeT, int]]:
    """Compute BFS distance and shortest-path count from one start node."""
    if start not in adjacency:
        return {}, {}

    dist: Dict[NodeT, int] = {start: 0}
    count: Dict[NodeT, int] = {start: 1}
    queue: deque[NodeT] = deque([start])

    while queue:
        node = queue.popleft()
        node_dist = int(dist[node])
        node_count = int(count[node])
        for neighbor in adjacency.get(node, ()):
            next_dist = int(node_dist + 1)
            if neighbor not in dist:
                dist[neighbor] = next_dist
                count[neighbor] = node_count
                queue.append(neighbor)
            elif int(dist[neighbor]) == next_dist:
                count[neighbor] = int(count.get(neighbor, 0) + node_count)

    return dist, count


def reconstruct_unique_shortest_path_by_adjacency(
    adjacency: Mapping[NodeT, Sequence[NodeT]],
    *,
    start: NodeT,
    goal: NodeT,
    dist_start: Mapping[NodeT, int],
    dist_goal: Mapping[NodeT, int],
) -> List[NodeT] | None:
    """Reconstruct a shortest path when exactly one witness path exists."""
    if start not in dist_start or goal not in dist_start:
        return None
    shortest = int(dist_start[goal])
    if shortest < 0:
        return None

    path: List[NodeT] = [start]
    current = start
    while current != goal:
        if current not in dist_start:
            return None
        current_dist = int(dist_start[current])
        candidates: List[NodeT] = []
        for neighbor in adjacency.get(current, ()):
            if neighbor not in dist_start or neighbor not in dist_goal:
                continue
            if int(dist_start[neighbor]) != int(current_dist + 1):
                continue
            if int(dist_start[neighbor]) + int(dist_goal[neighbor]) != shortest:
                continue
            candidates.append(neighbor)
        if len(candidates) != 1:
            return None
        current = candidates[0]
        path.append(current)
    return path


def connected_components_by_adjacency(
    adjacency: Mapping[NodeT, Sequence[NodeT]],
    *,
    node_order: Sequence[NodeT] | None = None,
) -> List[List[NodeT]]:
    """Return deterministic connected components for one undirected adjacency map."""
    ordered_nodes = list(node_order) if node_order is not None else list(adjacency.keys())
    visited: set[NodeT] = set()
    components: List[List[NodeT]] = []

    for node in ordered_nodes:
        if node in visited or node not in adjacency:
            continue
        queue: deque[NodeT] = deque([node])
        visited.add(node)
        component: List[NodeT] = []
        while queue:
            current = queue.popleft()
            component.append(current)
            for neighbor in adjacency.get(current, ()):
                if neighbor in visited:
                    continue
                visited.add(neighbor)
                queue.append(neighbor)
        components.append(component)

    return components


def unique_topological_order_by_adjacency(
    successors: Mapping[NodeT, Sequence[NodeT]],
    *,
    node_order: Sequence[NodeT] | None = None,
) -> List[NodeT] | None:
    """Return the unique topological order when exactly one order exists.

    The input is a directed successor adjacency mapping. The function returns
    ``None`` when the graph contains a cycle or when more than one valid
    topological ordering exists.
    """

    ordered_nodes = list(node_order) if node_order is not None else list(successors.keys())
    indegree: Dict[NodeT, int] = {node: 0 for node in ordered_nodes}
    for node in ordered_nodes:
        for neighbor in successors.get(node, ()):
            indegree[neighbor] = int(indegree.get(neighbor, 0) + 1)

    zero_indegree: List[NodeT] = [node for node in ordered_nodes if int(indegree.get(node, 0)) == 0]
    ordering: List[NodeT] = []
    while zero_indegree:
        if len(zero_indegree) != 1:
            return None
        node = zero_indegree.pop()
        ordering.append(node)
        for neighbor in successors.get(node, ()):
            indegree[neighbor] = int(indegree.get(neighbor, 0) - 1)
            if int(indegree[neighbor]) == 0:
                zero_indegree.append(neighbor)
    if len(ordering) != len(indegree):
        return None
    return ordering


__all__ = [
    "bfs_dist_count_by_adjacency",
    "connected_components_by_adjacency",
    "reconstruct_unique_shortest_path_by_adjacency",
    "unique_topological_order_by_adjacency",
]
