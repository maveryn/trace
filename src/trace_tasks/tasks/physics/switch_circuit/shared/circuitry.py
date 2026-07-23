"""Circuit graph primitives for switch-circuit diagrams."""

from __future__ import annotations

import itertools
from collections import deque
from typing import Dict, Iterable, List, Mapping, Sequence, Tuple

from .state import BULB_LABELS, NEG_NODE, POS_NODE, SWITCH_LABELS, CircuitEdge


def make_edges(switch_states: Mapping[str, bool]) -> Tuple[CircuitEdge, ...]:
    """Return the mixed-branch circuit edges for one switch-state assignment."""

    return (
        CircuitEdge("S1", "switch", POS_NODE, "T0", "S1", bool(switch_states["S1"])),
        CircuitEdge("B1", "bulb", "T0", "T1", "B1", True),
        CircuitEdge("B2", "bulb", "T1", NEG_NODE, "B2", True),
        CircuitEdge("S2", "switch", POS_NODE, "M0", "S2", bool(switch_states["S2"])),
        CircuitEdge("B3", "bulb", "M0", "M1", "B3", True),
        CircuitEdge("S3", "switch", "M1", "MJ", "S3", bool(switch_states["S3"])),
        CircuitEdge("S4", "switch", "M0", "M2", "S4", bool(switch_states["S4"])),
        CircuitEdge("B4", "bulb", "M2", "MJ", "B4", True),
        CircuitEdge("W1", "wire", "MJ", NEG_NODE, "", True),
        CircuitEdge("S5", "switch", POS_NODE, "D0", "S5", bool(switch_states["S5"])),
        CircuitEdge("B5", "bulb", "D0", NEG_NODE, "B5", True),
    )


def adjacency(
    edges: Iterable[CircuitEdge],
    *,
    exclude_edge_id: str | None = None,
) -> Dict[str, List[str]]:
    """Return conductive adjacency, optionally removing one edge."""

    result: Dict[str, List[str]] = {}
    for edge in edges:
        if str(edge.edge_id) == str(exclude_edge_id):
            continue
        if not bool(edge.conductive):
            continue
        result.setdefault(str(edge.node_a), []).append(str(edge.node_b))
        result.setdefault(str(edge.node_b), []).append(str(edge.node_a))
    return result


def reachable_nodes(start: str, graph: Mapping[str, Sequence[str]]) -> set[str]:
    """Return all nodes reachable from start in the conductive graph."""

    seen = {str(start)}
    queue: deque[str] = deque([str(start)])
    while queue:
        node = queue.popleft()
        for neighbor in graph.get(node, ()):
            neighbor = str(neighbor)
            if neighbor in seen:
                continue
            seen.add(neighbor)
            queue.append(neighbor)
    return seen


def lit_bulbs_from_edges(edges: Sequence[CircuitEdge]) -> Tuple[str, ...]:
    """Return bulb labels that lie on a simple conductive battery path."""

    conductive_edges = [edge for edge in edges if bool(edge.conductive)]
    edge_lookup: Dict[Tuple[str, str], List[CircuitEdge]] = {}
    for edge in conductive_edges:
        node_a = str(edge.node_a)
        node_b = str(edge.node_b)
        edge_lookup.setdefault((node_a, node_b), []).append(edge)
        edge_lookup.setdefault((node_b, node_a), []).append(edge)

    graph = adjacency(conductive_edges)
    lit: set[str] = set()
    stack: List[Tuple[str, Tuple[str, ...]]] = [(POS_NODE, (POS_NODE,))]
    while stack:
        node, path = stack.pop()
        if node == NEG_NODE:
            for left, right in zip(path, path[1:]):
                for edge in edge_lookup.get((str(left), str(right)), ()):
                    if edge.kind == "bulb":
                        lit.add(str(edge.label))
            continue
        for neighbor in graph.get(str(node), ()):
            neighbor = str(neighbor)
            if neighbor in path:
                continue
            stack.append((neighbor, (*path, neighbor)))
    return tuple(label for label in BULB_LABELS if label in lit)


def enumerate_switch_state_solutions(target_answer: int) -> List[Dict[str, bool]]:
    """Enumerate switch-state assignments realizing a target lit-bulb count."""

    solutions: List[Dict[str, bool]] = []
    for values in itertools.product((False, True), repeat=len(SWITCH_LABELS)):
        states = {label: bool(values[index]) for index, label in enumerate(SWITCH_LABELS)}
        if len(lit_bulbs_from_edges(make_edges(states))) == int(target_answer):
            solutions.append(dict(states))
    return solutions


__all__ = [
    "adjacency",
    "enumerate_switch_state_solutions",
    "lit_bulbs_from_edges",
    "make_edges",
    "reachable_nodes",
]
