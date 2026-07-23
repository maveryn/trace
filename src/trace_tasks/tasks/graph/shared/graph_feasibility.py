"""Pure feasible-support helpers for graph-domain samplers."""

from __future__ import annotations

from typing import Tuple


def feasible_node_counts_for_component_query(
    *,
    target_component_size: int,
    component_count: int,
    node_count_min: int,
    node_count_max: int,
) -> Tuple[int, ...]:
    """Return node counts that can realize one disconnected component query."""

    minimum = max(int(node_count_min), int(target_component_size) + int(component_count) - 1)
    maximum = int(node_count_max)
    if int(minimum) > int(maximum):
        return ()
    return tuple(range(int(minimum), int(maximum) + 1))


def feasible_node_counts_for_component_size_after_edge_edit(
    *,
    edit_operation: str,
    target_component_size: int,
    node_count_min: int,
    node_count_max: int,
) -> Tuple[int, ...]:
    """Return node counts that can realize one undirected edge-edit component query."""

    operation = str(edit_operation)
    target_size = int(target_component_size)
    if operation == "edge_removal":
        if int(target_size) < 1:
            return ()
        minimum = max(int(node_count_min), int(target_size) + 1)
    elif operation == "edge_addition":
        if int(target_size) < 2:
            return ()
        minimum = max(int(node_count_min), int(target_size))
    else:
        return ()
    maximum = int(node_count_max)
    if int(minimum) > int(maximum):
        return ()
    return tuple(range(int(minimum), int(maximum) + 1))


def feasible_node_counts_for_reachable_count(
    *,
    target_reachable_count: int,
    node_count_min: int,
    node_count_max: int,
) -> Tuple[int, ...]:
    """Return node counts that can realize one directed reachable-count query.

    The queried source node is included in the count, and at least one node
    remains unreachable so the scene never collapses to a fully reachable graph.
    """

    target_count = int(target_reachable_count)
    if int(target_count) < 1:
        return ()
    minimum = max(int(node_count_min), 5, int(target_count) + 1)
    maximum = int(node_count_max)
    if int(minimum) > int(maximum):
        return ()
    return tuple(range(int(minimum), int(maximum) + 1))


def feasible_node_counts_for_reachable_count_after_edge_edit(
    *,
    edit_operation: str,
    target_reachable_count: int,
    node_count_min: int,
    node_count_max: int,
) -> Tuple[int, ...]:
    """Return node counts that can realize one directed edge-edit reachability query."""

    operation = str(edit_operation)
    target_count = int(target_reachable_count)
    if operation == "edge_removal":
        if int(target_count) < 1:
            return ()
        minimum = max(int(node_count_min), 5, int(target_count) + 1)
    elif operation == "edge_addition":
        if int(target_count) < 2:
            return ()
        minimum = max(int(node_count_min), 5, int(target_count))
    else:
        return ()
    maximum = int(node_count_max)
    if int(minimum) > int(maximum):
        return ()
    return tuple(range(int(minimum), int(maximum) + 1))


def feasible_node_counts_for_unique_largest_component(
    *,
    target_largest_component_size: int,
    component_count: int,
    node_count_min: int,
    node_count_max: int,
) -> Tuple[int, ...]:
    """Return node counts that can realize one unique-largest-component query."""

    target_size = int(target_largest_component_size)
    components = int(component_count)
    if int(target_size) <= 1 or int(components) <= 1:
        return ()
    minimum = max(int(node_count_min), int(target_size) + int(components) - 1)
    maximum = min(int(node_count_max), int(target_size) + ((int(components) - 1) * (int(target_size) - 1)))
    if int(minimum) > int(maximum):
        return ()
    return tuple(range(int(minimum), int(maximum) + 1))


def feasible_node_counts_for_unique_cycle_size(
    *,
    target_cycle_size: int,
    node_count_min: int,
    node_count_max: int,
) -> Tuple[int, ...]:
    """Return node counts that can realize one unique-cycle-size query."""

    target_size = int(target_cycle_size)
    minimum = max(int(node_count_min), 4, int(target_size) + 1)
    maximum = int(node_count_max)
    if int(minimum) > int(maximum):
        return ()
    return tuple(range(int(minimum), int(maximum) + 1))


def feasible_node_counts_for_largest_chordless_cycle_size(
    *,
    target_cycle_size: int,
    node_count_min: int,
    node_count_max: int,
) -> Tuple[int, ...]:
    """Return node counts that can realize one largest-chordless-cycle query."""

    target_size = int(target_cycle_size)
    if int(target_size) < 3:
        return ()
    secondary_cycle_extra_nodes = 0 if int(target_size) == 3 else 1
    minimum = max(int(node_count_min), int(target_size) + int(secondary_cycle_extra_nodes))
    maximum = int(node_count_max)
    if int(minimum) > int(maximum):
        return ()
    return tuple(range(int(minimum), int(maximum) + 1))


def feasible_node_counts_for_hamiltonian_cycle_neighbor(
    *,
    node_count_min: int,
    node_count_max: int,
) -> Tuple[int, ...]:
    """Return node counts that can realize a small Hamiltonian-cycle neighbor query."""

    minimum = max(int(node_count_min), 4)
    maximum = int(node_count_max)
    if int(minimum) > int(maximum):
        return ()
    return tuple(range(int(minimum), int(maximum) + 1))


def feasible_node_counts_for_shortest_path_length(
    *,
    target_shortest_path_length: int,
    node_count_min: int,
    node_count_max: int,
) -> Tuple[int, ...]:
    """Return node counts that can realize one unique shortest-path query."""

    target_length = int(target_shortest_path_length)
    minimum = max(int(node_count_min), 5, int(target_length) + 2)
    maximum = int(node_count_max)
    if int(target_length) < 1 or int(minimum) > int(maximum):
        return ()
    return tuple(range(int(minimum), int(maximum) + 1))


def feasible_node_counts_for_longest_path_length(
    *,
    target_longest_path_length: int,
    node_count_min: int,
    node_count_max: int,
) -> Tuple[int, ...]:
    """Return node counts that can realize one unique longest-path DAG query."""

    target_length = int(target_longest_path_length)
    minimum = max(int(node_count_min), 4, int(target_length) + 2)
    maximum = int(node_count_max)
    if int(target_length) < 1 or int(minimum) > int(maximum):
        return ()
    return tuple(range(int(minimum), int(maximum) + 1))


def feasible_node_counts_for_topological_position(
    *,
    target_position: int,
    node_count_min: int,
    node_count_max: int,
) -> Tuple[int, ...]:
    """Return node counts that can realize a queried topological position."""

    target_position_int = int(target_position)
    minimum = max(int(node_count_min), 3, int(target_position_int))
    maximum = int(node_count_max)
    if int(target_position_int) < 1 or int(minimum) > int(maximum):
        return ()
    return tuple(range(int(minimum), int(maximum) + 1))


def feasible_node_counts_for_articulation_point_count(
    *,
    target_count: int,
    node_count_min: int,
    node_count_max: int,
) -> Tuple[int, ...]:
    """Return node counts that can realize one articulation-point count query."""

    target_int = int(target_count)
    if int(target_int) < 0:
        return ()
    minimum = int(node_count_min)
    if int(target_int) > 0:
        minimum = max(int(minimum), int(target_int) + 2)
    maximum = int(node_count_max)
    if int(minimum) > int(maximum):
        return ()
    return tuple(range(int(minimum), int(maximum) + 1))


def feasible_node_counts_for_bridge_count(
    *,
    target_count: int,
    node_count_min: int,
    node_count_max: int,
) -> Tuple[int, ...]:
    """Return node counts that can realize one bridge-edge count query.

    The connected construction uses a tree of bridgeless blocks. This makes
    `target_count + 2` nodes infeasible because it would force exactly one
    extra node beyond the bridge skeleton, which cannot form a bridgeless block.
    """

    target_int = int(target_count)
    if int(target_int) < 0:
        return ()
    feasible: list[int] = []
    for node_count in range(int(node_count_min), int(node_count_max) + 1):
        node_count_int = int(node_count)
        if int(target_int) == 0:
            if int(node_count_int) >= 3:
                feasible.append(int(node_count_int))
            continue
        if int(node_count_int) < int(target_int) + 1:
            continue
        if int(node_count_int) == int(target_int) + 2:
            continue
        feasible.append(int(node_count_int))
    return tuple(int(value) for value in feasible)


def feasible_extra_edge_counts_for_minimum_spanning_tree(
    *,
    node_count: int,
    extra_edge_count_min: int,
    extra_edge_count_max: int,
    edge_weight_min: int,
    edge_weight_max: int,
) -> Tuple[int, ...]:
    """Return feasible non-tree edge counts for one weighted MST query.

    The MST construction uses a connected spanning tree plus a small number of
    additional non-tree edges. We require all rendered edge weights to be
    distinct integers in ``[edge_weight_min, edge_weight_max]``, so the total
    edge count must not exceed the available weight support.
    """

    node_count_int = int(node_count)
    tree_edge_count = max(0, int(node_count_int) - 1)
    max_available_edges = int(edge_weight_max) - int(edge_weight_min) + 1
    if int(tree_edge_count) <= 0 or int(max_available_edges) <= int(tree_edge_count):
        return ()

    max_non_edges = (int(node_count_int) * int(node_count_int - 1) // 2) - int(tree_edge_count)
    feasible_max = min(
        int(extra_edge_count_max),
        int(max_non_edges),
        int(max_available_edges - tree_edge_count),
    )
    feasible_min = max(1, int(extra_edge_count_min))
    if int(feasible_min) > int(feasible_max):
        return ()
    return tuple(range(int(feasible_min), int(feasible_max) + 1))


__all__ = [
    "feasible_extra_edge_counts_for_minimum_spanning_tree",
    "feasible_node_counts_for_articulation_point_count",
    "feasible_node_counts_for_bridge_count",
    "feasible_node_counts_for_component_query",
    "feasible_node_counts_for_component_size_after_edge_edit",
    "feasible_node_counts_for_hamiltonian_cycle_neighbor",
    "feasible_node_counts_for_largest_chordless_cycle_size",
    "feasible_node_counts_for_longest_path_length",
    "feasible_node_counts_for_reachable_count",
    "feasible_node_counts_for_reachable_count_after_edge_edit",
    "feasible_node_counts_for_shortest_path_length",
    "feasible_node_counts_for_topological_position",
    "feasible_node_counts_for_unique_cycle_size",
    "feasible_node_counts_for_unique_largest_component",
]
