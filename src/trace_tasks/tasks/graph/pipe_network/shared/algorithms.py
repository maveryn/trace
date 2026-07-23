"""Grid and graph algorithms for pipe-network scene sampling."""

from __future__ import annotations

import random
from typing import Dict, Mapping, Sequence, Tuple

import networkx as nx

from ...shared.graph_sample_types import canonicalize_graph_edge_label
from .state import GridCell, LabelEdge, NodeEdge


def parse_pipe_grid_shape(grid_shape_variant: str) -> Tuple[int, int]:
    """Parse one ``RxC`` pipe-board grid shape."""

    text = str(grid_shape_variant).strip().lower()
    if "x" not in text:
        raise ValueError(f"unsupported pipe grid shape: {grid_shape_variant}")
    left, right = text.split("x", 1)
    rows = int(left)
    cols = int(right)
    if rows <= 1 or cols <= 1:
        raise ValueError(f"unsupported pipe grid shape: {grid_shape_variant}")
    return int(rows), int(cols)


def feasible_pipe_node_counts(*, node_count_min: int, node_count_max: int, grid_shape_variant: str) -> Tuple[int, ...]:
    """Return node counts that fit in the requested pipe grid shape."""

    rows, cols = parse_pipe_grid_shape(str(grid_shape_variant))
    maximum = min(int(node_count_max), int(rows * cols))
    minimum = int(node_count_min)
    if int(minimum) > int(maximum):
        return ()
    return tuple(range(int(minimum), int(maximum) + 1))


def canonical_node_edge(left: int, right: int) -> NodeEdge:
    """Return one canonical integer node edge."""

    a = int(left)
    b = int(right)
    return (a, b) if a < b else (b, a)


def sort_node_edges(edges: Sequence[NodeEdge]) -> Tuple[NodeEdge, ...]:
    """Sort canonical integer node edges."""

    return tuple(sorted((canonical_node_edge(left, right) for left, right in edges), key=lambda edge: (edge[0], edge[1])))


def grid_cells(rows: int, cols: int) -> Tuple[GridCell, ...]:
    """Return row-major grid cells."""

    return tuple((int(row), int(col)) for row in range(int(rows)) for col in range(int(cols)))


def sample_connected_cells(
    rng: random.Random,
    *,
    rows: int,
    cols: int,
    node_count: int,
) -> Tuple[GridCell, ...]:
    """Sample one connected subset of grid cells."""

    all_cells = set(grid_cells(int(rows), int(cols)))
    if int(node_count) > len(all_cells):
        raise ValueError("node_count exceeds pipe grid capacity")

    start = rng.choice(tuple(sorted(all_cells)))
    selected = {start}
    while len(selected) < int(node_count):
        frontier: set[GridCell] = set()
        for row, col in selected:
            for candidate in ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1)):
                if candidate in all_cells and candidate not in selected:
                    frontier.add(candidate)
        if not frontier:
            raise ValueError("failed to grow connected pipe-cell subset")
        selected.add(rng.choice(tuple(sorted(frontier))))
    return tuple(sorted(selected))


def candidate_edges_for_cells(cells_by_node: Mapping[int, GridCell]) -> Tuple[NodeEdge, ...]:
    """Return all four-neighbor candidate pipe edges among selected cells."""

    items = list(cells_by_node.items())
    edges: list[NodeEdge] = []
    for index, (left_node, left_cell) in enumerate(items):
        for right_node, right_cell in items[index + 1 :]:
            distance = abs(int(left_cell[0]) - int(right_cell[0])) + abs(int(left_cell[1]) - int(right_cell[1]))
            if int(distance) == 1:
                edges.append(canonical_node_edge(int(left_node), int(right_node)))
    return sort_node_edges(edges)


def candidate_adjacency(nodes: Sequence[int], candidate_edges: Sequence[NodeEdge]) -> Dict[int, Tuple[int, ...]]:
    """Build adjacency from candidate node edges."""

    adjacency: Dict[int, list[int]] = {int(node): [] for node in nodes}
    for left, right in candidate_edges:
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    return {int(node): tuple(sorted(values)) for node, values in adjacency.items()}


def random_spanning_tree_edges(
    rng: random.Random,
    *,
    nodes: Sequence[int],
    candidate_edges: Sequence[NodeEdge],
) -> Tuple[NodeEdge, ...]:
    """Sample one spanning tree over a connected candidate graph."""

    ordered_nodes = tuple(sorted(int(node) for node in nodes))
    if not ordered_nodes:
        return ()
    edge_pool = sort_node_edges(candidate_edges)
    visited = {int(rng.choice(ordered_nodes))}
    tree_edges: list[NodeEdge] = []
    while len(visited) < len(ordered_nodes):
        frontier = [
            edge
            for edge in edge_pool
            if (edge[0] in visited and edge[1] not in visited) or (edge[1] in visited and edge[0] not in visited)
        ]
        if not frontier:
            raise ValueError("candidate graph is not connected")
        edge = rng.choice(frontier)
        tree_edges.append(edge)
        visited.add(int(edge[0]))
        visited.add(int(edge[1]))
    return sort_node_edges(tree_edges)


def connected_node_subset(
    rng: random.Random,
    *,
    nodes: Sequence[int],
    candidate_edges: Sequence[NodeEdge],
    subset_size: int,
) -> Tuple[int, ...]:
    """Sample one connected subset of nodes from the candidate graph."""

    ordered_nodes = tuple(sorted(int(node) for node in nodes))
    if int(subset_size) > len(ordered_nodes):
        raise ValueError("subset_size exceeds node count")
    adjacency = candidate_adjacency(ordered_nodes, candidate_edges)
    start = int(rng.choice(ordered_nodes))
    selected = {int(start)}
    while len(selected) < int(subset_size):
        frontier = sorted({int(neighbor) for node in selected for neighbor in adjacency.get(int(node), ()) if int(neighbor) not in selected})
        if not frontier:
            raise ValueError("failed to sample connected node subset")
        selected.add(int(rng.choice(frontier)))
    return tuple(sorted(selected))


def induced_edges(candidate_edges: Sequence[NodeEdge], nodes: Sequence[int]) -> Tuple[NodeEdge, ...]:
    """Return candidate edges induced by a node subset."""

    node_set = {int(node) for node in nodes}
    return sort_node_edges((left, right) for left, right in candidate_edges if int(left) in node_set and int(right) in node_set)


def add_random_open_edges(
    graph: nx.Graph,
    rng: random.Random,
    *,
    candidate_edges: Sequence[NodeEdge],
    max_extra_edges: int,
) -> int:
    """Add random candidate non-tree edges and return how many were added."""

    available = [edge for edge in sort_node_edges(candidate_edges) if not graph.has_edge(int(edge[0]), int(edge[1]))]
    rng.shuffle(available)
    added = 0
    for left, right in available[: max(0, int(max_extra_edges))]:
        graph.add_edge(int(left), int(right))
        added += 1
    return int(added)


def sample_blocked_edges(
    rng: random.Random,
    *,
    candidate_edges: Sequence[NodeEdge],
    open_edges: Sequence[NodeEdge],
    min_count: int = 1,
    max_count: int = 5,
) -> Tuple[NodeEdge, ...]:
    """Sample visible blocked pipe edges from non-open candidate edges."""

    open_set = set(sort_node_edges(open_edges))
    available = [edge for edge in sort_node_edges(candidate_edges) if edge not in open_set]
    if not available:
        return ()
    upper = min(len(available), max(0, int(max_count)))
    lower = min(upper, max(0, int(min_count)))
    count = int(rng.randint(int(lower), int(upper))) if int(upper) >= int(lower) else 0
    rng.shuffle(available)
    return sort_node_edges(available[:count])


def open_adjacency_by_node(graph: nx.Graph) -> Dict[int, Tuple[int, ...]]:
    """Return deterministic open-pipe adjacency by node id."""

    return {int(node): tuple(sorted(int(value) for value in graph.neighbors(int(node)))) for node in sorted(graph.nodes())}


def label_edge(label_by_node: Mapping[int, str], edge: NodeEdge) -> LabelEdge:
    """Convert one integer edge to a canonical label edge."""

    return canonicalize_graph_edge_label(str(label_by_node[int(edge[0])]), str(label_by_node[int(edge[1])]), directed=False)


def new_node_graph(node_count: int) -> nx.Graph:
    """Return an empty graph over node ids."""

    graph = nx.Graph()
    graph.add_nodes_from(range(int(node_count)))
    return graph


__all__ = [
    "add_random_open_edges",
    "candidate_adjacency",
    "candidate_edges_for_cells",
    "canonical_node_edge",
    "connected_node_subset",
    "feasible_pipe_node_counts",
    "grid_cells",
    "induced_edges",
    "label_edge",
    "new_node_graph",
    "open_adjacency_by_node",
    "parse_pipe_grid_shape",
    "random_spanning_tree_edges",
    "sample_blocked_edges",
    "sample_connected_cells",
    "sort_node_edges",
]
