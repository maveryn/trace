"""Label, color, and edge-attribute graph samplers."""

from __future__ import annotations

import random
from itertools import product
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import networkx as nx

from .graph_edge_sampling import (
    _add_directed_edge_without_reciprocal,
    _add_random_directed_distractor_edges,
    _add_random_undirected_distractor_edges,
    _profile_extra_edge_budget,
)
from .graph_profile_sampling import _sample_profile_tree_graph
from .graph_sample_types import (
    SUPPORTED_CROSS_COLOR_EDGE_COUNT_DIRECTIONS,
    SUPPORTED_EDGE_ATTRIBUTE_LABEL_DIRECTIONS,
    SUPPORTED_EDGE_COLOR_COUNT_DIRECTIONS,
    SUPPORTED_NODE_COLOR_COUNT_DIRECTIONS,
    SUPPORTED_UNIQUE_NODE_LABEL_RELATION_MODES,
    GraphCrossColorEdgeCountSample,
    GraphEdgeAttributeLabelSample,
    GraphEdgeColorCountSample,
    GraphEdgeTextLabelCountSample,
    GraphNodeColorCountSample,
    GraphUniqueNodeLabelRelationSample,
    canonicalize_graph_edge_label,
    graph_label_sort_key,
    sort_graph_edge_labels,
)
from .graph_topology_helpers import _build_labeled_graph_topology_sample, _has_reciprocal_edges


EdgeLabelSupportResolver = Callable[[random.Random, Sequence[str]], Tuple[Sequence[str], Mapping[str, Any]]]


def _resolve_visible_edge_label_support(
    rng: random.Random,
    *,
    node_labels: Sequence[str],
    target_edge_label: str,
    edge_label_support: Sequence[str],
    target_edge_label_index: int,
    edge_label_support_resolver: EdgeLabelSupportResolver | None,
) -> Tuple[Tuple[str, ...], str, Dict[str, Any]]:
    """Resolve per-instance visible edge-text labels after node labels exist."""

    metadata: Dict[str, Any] = {}
    if edge_label_support_resolver is not None:
        resolved_support, resolved_metadata = edge_label_support_resolver(
            rng,
            tuple(str(label) for label in node_labels),
        )
        edge_labels_supported = tuple(
            str(label).strip().lower()
            for label in resolved_support
            if str(label).strip()
        )
        metadata = dict(resolved_metadata)
        explicit_target = str(target_edge_label).strip().lower()
        if explicit_target:
            target_label = explicit_target
        elif edge_labels_supported:
            target_index = int(target_edge_label_index)
            if not 0 <= target_index < len(edge_labels_supported):
                raise ValueError("target_edge_label_index is outside edge-label support")
            target_label = str(edge_labels_supported[target_index])
        else:
            target_label = ""
    else:
        edge_labels_supported = tuple(str(label).strip().lower() for label in edge_label_support if str(label).strip())
        target_label = str(target_edge_label).strip().lower()
    if len(set(edge_labels_supported)) != len(edge_labels_supported) or len(edge_labels_supported) < 2:
        raise ValueError("edge_label_support must contain at least two unique labels")
    node_label_set = {str(label).strip().lower() for label in node_labels if str(label).strip()}
    if any(str(label) in node_label_set for label in edge_labels_supported):
        raise ValueError("edge labels must not overlap node labels")
    if target_label not in set(edge_labels_supported):
        raise ValueError("target_edge_label is outside edge_label_support")
    return tuple(str(label) for label in edge_labels_supported), str(target_label), dict(metadata)


def _sample_unique_node_label_undirected_graph(
    rng: random.Random,
    *,
    node_count: int,
    max_degree: int,
    topology_profile: str,
) -> Tuple[nx.Graph, int, int]:
    """Construct an undirected graph where one query node has one neighbor."""

    node_count_int = int(node_count)
    if int(node_count_int) < 3:
        raise ValueError("unique-neighbor lookup requires at least three nodes")
    max_degree_int = max(1, int(max_degree))

    nodes = list(range(int(node_count_int)))
    query_node = int(rng.choice(nodes))
    answer_node = int(rng.choice([node for node in nodes if int(node) != int(query_node)]))
    graph = nx.Graph()
    graph.add_nodes_from(nodes)
    graph.add_edge(int(query_node), int(answer_node))

    non_query_nodes = [int(node) for node in nodes if int(node) != int(query_node)]
    shuffled = list(non_query_nodes)
    rng.shuffle(shuffled)
    for left, right in zip(shuffled, shuffled[1:]):
        if int(graph.degree(int(left))) < int(max_degree_int) and int(graph.degree(int(right))) < int(max_degree_int):
            graph.add_edge(int(left), int(right))

    _add_random_undirected_distractor_edges(
        graph,
        rng,
        nodes=tuple(non_query_nodes),
        extra_edges=int(
            _profile_extra_edge_budget(
                node_count=int(node_count_int),
                topology_profile=str(topology_profile),
                directed=False,
            )
        ),
        max_degree=int(max_degree_int),
    )
    if int(graph.degree(int(query_node))) != 1:
        raise ValueError("unique-neighbor sampler failed to preserve one query neighbor")
    return graph, int(query_node), int(answer_node)


def _sample_unique_node_label_directed_graph(
    rng: random.Random,
    *,
    relation_mode: str,
    node_count: int,
    max_degree: int,
    topology_profile: str,
) -> Tuple[nx.DiGraph, int, int]:
    """Construct a directed graph with one successor or predecessor for the query node."""

    mode = str(relation_mode)
    if mode not in {"directed_unique_successor", "directed_unique_predecessor"}:
        raise ValueError(f"unsupported directed unique-node relation mode: {relation_mode}")
    node_count_int = int(node_count)
    if int(node_count_int) < 3:
        raise ValueError("unique directed-node lookup requires at least three nodes")
    max_degree_int = max(1, int(max_degree))

    nodes = list(range(int(node_count_int)))
    query_node = int(rng.choice(nodes))
    answer_node = int(rng.choice([node for node in nodes if int(node) != int(query_node)]))
    graph = nx.DiGraph()
    graph.add_nodes_from(nodes)
    if str(mode) == "directed_unique_successor":
        graph.add_edge(int(query_node), int(answer_node))
    else:
        graph.add_edge(int(answer_node), int(query_node))

    non_query_nodes = [int(node) for node in nodes if int(node) != int(query_node)]
    shuffled = list(non_query_nodes)
    rng.shuffle(shuffled)
    for left, right in zip(shuffled, shuffled[1:]):
        if rng.random() < 0.5:
            source, target = int(left), int(right)
        else:
            source, target = int(right), int(left)
        _add_directed_edge_without_reciprocal(
            graph,
            source=int(source),
            target=int(target),
            max_degree=int(max_degree_int),
        )

    candidates = [
        (int(left), int(right))
        for left in nodes
        for right in nodes
        if int(left) != int(right)
    ]
    rng.shuffle(candidates)
    extra_edges = int(
        _profile_extra_edge_budget(
            node_count=int(node_count_int),
            topology_profile=str(topology_profile),
            directed=True,
        )
    )
    added = 0
    for source, target in candidates:
        if int(added) >= int(extra_edges):
            break
        if str(mode) == "directed_unique_successor" and int(source) == int(query_node):
            continue
        if str(mode) == "directed_unique_predecessor" and int(target) == int(query_node):
            continue
        if _add_directed_edge_without_reciprocal(
            graph,
            source=int(source),
            target=int(target),
            max_degree=int(max_degree_int),
        ):
            added += 1

    if _has_reciprocal_edges(graph):
        raise ValueError("unique directed-node sampler produced reciprocal edges")
    if str(mode) == "directed_unique_successor":
        observed = tuple(int(node) for node in graph.successors(int(query_node)))
        if observed != (int(answer_node),):
            raise ValueError("unique-successor sampler failed to preserve one query successor")
    else:
        observed = tuple(int(node) for node in graph.predecessors(int(query_node)))
        if observed != (int(answer_node),):
            raise ValueError("unique-predecessor sampler failed to preserve one query predecessor")
    return graph, int(query_node), int(answer_node)


def sample_unique_node_label_relation_graph(
    rng: random.Random,
    *,
    relation_mode: str,
    node_count: int,
    max_degree: int,
    topology_profile: str,
    label_variant: str,
) -> GraphUniqueNodeLabelRelationSample:
    """Construct one graph for a unique neighbor/successor/predecessor label query."""

    mode = str(relation_mode)
    if mode not in SUPPORTED_UNIQUE_NODE_LABEL_RELATION_MODES:
        raise ValueError(f"unsupported unique-node relation mode: {relation_mode}")
    if str(mode) == "undirected_unique_neighbor":
        graph, query_node, answer_node = _sample_unique_node_label_undirected_graph(
            rng,
            node_count=int(node_count),
            max_degree=int(max_degree),
            topology_profile=str(topology_profile),
        )
        directed = False
    else:
        graph, query_node, answer_node = _sample_unique_node_label_directed_graph(
            rng,
            relation_mode=str(mode),
            node_count=int(node_count),
            max_degree=int(max_degree),
            topology_profile=str(topology_profile),
        )
        directed = True

    topology_sample, label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=bool(directed),
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    query_label = str(label_by_node[int(query_node)])
    answer_label = str(label_by_node[int(answer_node)])
    if str(mode) == "directed_unique_predecessor":
        supporting_edge = canonicalize_graph_edge_label(str(answer_label), str(query_label), directed=True)
    else:
        supporting_edge = canonicalize_graph_edge_label(str(query_label), str(answer_label), directed=bool(directed))

    if str(mode) == "undirected_unique_neighbor":
        target_labels = tuple(str(label) for label in topology_sample.adjacency_by_label[str(query_label)])
    elif str(mode) == "directed_unique_successor":
        target_labels = tuple(str(label) for label in topology_sample.successors_by_label[str(query_label)])
    else:
        target_labels = tuple(str(label) for label in topology_sample.predecessors_by_label[str(query_label)])
    if tuple(target_labels) != (str(answer_label),):
        raise ValueError("unique-node relation target labels are inconsistent with topology metadata")

    return GraphUniqueNodeLabelRelationSample(
        graph=topology_sample.graph,
        directed=bool(directed),
        node_labels=tuple(str(label) for label in topology_sample.node_labels),
        edge_labels=tuple((str(left), str(right)) for left, right in topology_sample.edge_labels),
        degrees_by_label={str(key): int(value) for key, value in topology_sample.degrees_by_label.items()},
        in_degrees_by_label={str(key): int(value) for key, value in topology_sample.in_degrees_by_label.items()},
        out_degrees_by_label={str(key): int(value) for key, value in topology_sample.out_degrees_by_label.items()},
        adjacency_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.adjacency_by_label.items()},
        successors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.successors_by_label.items()},
        predecessors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.predecessors_by_label.items()},
        edge_count=int(topology_sample.edge_count),
        topology_profile=str(topology_sample.topology_profile),
        label_variant=str(topology_sample.label_variant),
        label_source_kind=str(topology_sample.label_source_kind),
        label_bucket=str(topology_sample.label_bucket),
        label_manifest=str(topology_sample.label_manifest),
        label_filter=dict(topology_sample.label_filter),
        label_bucket_probabilities=dict(topology_sample.label_bucket_probabilities),
        query_label=str(query_label),
        answer_label=str(answer_label),
        target_labels=(str(answer_label),),
        relation_mode=str(mode),
        graph_directionality="directed" if bool(directed) else "undirected",
        supporting_edge=(str(supporting_edge[0]), str(supporting_edge[1])),
    )


def _sample_node_color_count_base_graph(
    rng: random.Random,
    *,
    node_count: int,
    graph_directionality: str,
    topology_profile: str,
    max_degree: int,
    max_edge_count: int | None = None,
) -> nx.Graph | nx.DiGraph:
    """Sample one simple node-link graph for semantic color-count queries."""

    base_graph = _sample_profile_tree_graph(
        rng,
        node_count=int(node_count),
        topology_profile=str(topology_profile),
    )
    extra_edges = _profile_extra_edge_budget(
        node_count=int(node_count),
        topology_profile=str(topology_profile),
        directed=bool(str(graph_directionality) == "directed"),
    )
    if str(graph_directionality) == "directed":
        graph = nx.DiGraph()
        graph.add_nodes_from(int(node) for node in base_graph.nodes())
        for left, right in base_graph.edges():
            if rng.random() < 0.5:
                graph.add_edge(int(left), int(right))
            else:
                graph.add_edge(int(right), int(left))
        if max_edge_count is not None:
            extra_edges = min(
                int(extra_edges),
                max(0, int(max_edge_count) - int(graph.number_of_edges())),
            )
        _add_random_directed_distractor_edges(
            graph,
            rng,
            nodes=tuple(int(node) for node in graph.nodes()),
            extra_edges=int(extra_edges),
            max_degree=max(1, int(max_degree)),
        )
        if _has_reciprocal_edges(graph):
            raise ValueError("node-color directed sampler produced reciprocal edges")
        return graph

    graph = base_graph.copy()
    if max_edge_count is not None:
        extra_edges = min(
            int(extra_edges),
            max(0, int(max_edge_count) - int(graph.number_of_edges())),
        )
    _add_random_undirected_distractor_edges(
        graph,
        rng,
        nodes=tuple(int(node) for node in graph.nodes()),
        extra_edges=int(extra_edges),
        max_degree=max(1, int(max_degree)),
    )
    return graph


def sample_node_color_count_graph(
    rng: random.Random,
    *,
    graph_directionality: str,
    node_count: int,
    target_count: int,
    target_color_name: str,
    color_support: Sequence[str],
    topology_profile: str,
    label_variant: str,
    max_degree: int,
) -> GraphNodeColorCountSample:
    """Construct one labeled graph with exactly ``target_count`` nodes in one color."""

    directionality = str(graph_directionality)
    if directionality not in SUPPORTED_NODE_COLOR_COUNT_DIRECTIONS:
        raise ValueError(f"unsupported graph_directionality: {graph_directionality}")
    node_count_int = int(node_count)
    target_count_int = int(target_count)
    if int(target_count_int) < 0 or int(target_count_int) > int(node_count_int):
        raise ValueError("target_count must be between zero and node_count")

    colors = tuple(str(color).strip().lower() for color in color_support if str(color).strip())
    if len(set(colors)) != len(colors) or len(colors) < 2:
        raise ValueError("color_support must contain at least two unique color names")
    target_color = str(target_color_name).strip().lower()
    if target_color not in set(colors):
        raise ValueError("target_color_name is outside color_support")
    non_target_colors = tuple(str(color) for color in colors if str(color) != str(target_color))
    if int(target_count_int) < int(node_count_int) and not non_target_colors:
        raise ValueError("non-target nodes require at least one non-target color")

    graph = _sample_node_color_count_base_graph(
        rng,
        node_count=int(node_count_int),
        graph_directionality=str(directionality),
        topology_profile=str(topology_profile),
        max_degree=int(max_degree),
    )
    topology_sample, _label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=bool(directionality == "directed"),
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    labels = tuple(str(label) for label in topology_sample.node_labels)
    target_label_set = set(str(label) for label in rng.sample(labels, int(target_count_int)))
    node_color_names_by_label: Dict[str, str] = {}
    for label in labels:
        if str(label) in target_label_set:
            node_color_names_by_label[str(label)] = str(target_color)
        else:
            node_color_names_by_label[str(label)] = str(rng.choice(non_target_colors))
    color_counts_by_name = {
        str(color): sum(1 for label in labels if str(node_color_names_by_label[str(label)]) == str(color))
        for color in colors
    }
    target_labels = tuple(sorted((str(label) for label in target_label_set), key=graph_label_sort_key))
    if int(color_counts_by_name[str(target_color)]) != int(target_count_int):
        raise ValueError("node-color sampler produced inconsistent target metadata")

    return GraphNodeColorCountSample(
        graph=topology_sample.graph,
        directed=bool(topology_sample.directed),
        node_labels=tuple(str(label) for label in topology_sample.node_labels),
        edge_labels=tuple((str(left), str(right)) for left, right in topology_sample.edge_labels),
        degrees_by_label={str(key): int(value) for key, value in topology_sample.degrees_by_label.items()},
        in_degrees_by_label={str(key): int(value) for key, value in topology_sample.in_degrees_by_label.items()},
        out_degrees_by_label={str(key): int(value) for key, value in topology_sample.out_degrees_by_label.items()},
        adjacency_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.adjacency_by_label.items()},
        successors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.successors_by_label.items()},
        predecessors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.predecessors_by_label.items()},
        edge_count=int(topology_sample.edge_count),
        topology_profile=str(topology_sample.topology_profile),
        label_variant=str(topology_sample.label_variant),
        target_labels=tuple(str(label) for label in target_labels),
        target_count=int(target_count_int),
        target_color_name=str(target_color),
        node_color_names_by_label={str(key): str(value) for key, value in node_color_names_by_label.items()},
        color_counts_by_name={str(key): int(value) for key, value in color_counts_by_name.items()},
        graph_directionality=str(directionality),
    )


def _sample_edge_color_count_base_graph(
    rng: random.Random,
    *,
    node_count: int,
    graph_directionality: str,
    topology_profile: str,
    max_degree: int,
    max_edge_count: int | None = None,
) -> nx.Graph | nx.DiGraph:
    """Sample one simple node-link graph for semantic edge-color queries."""

    return _sample_node_color_count_base_graph(
        rng,
        node_count=int(node_count),
        graph_directionality=str(graph_directionality),
        topology_profile=str(topology_profile),
        max_degree=int(max_degree),
        max_edge_count=max_edge_count,
    )


def sample_edge_color_count_graph(
    rng: random.Random,
    *,
    graph_directionality: str,
    node_count: int,
    target_count: int,
    target_color_name: str,
    color_support: Sequence[str],
    topology_profile: str,
    label_variant: str,
    max_degree: int,
) -> GraphEdgeColorCountSample:
    """Construct one labeled graph with exactly ``target_count`` edges in one color."""

    directionality = str(graph_directionality)
    if directionality not in SUPPORTED_EDGE_COLOR_COUNT_DIRECTIONS:
        raise ValueError(f"unsupported graph_directionality: {graph_directionality}")
    node_count_int = int(node_count)
    target_count_int = int(target_count)
    if int(target_count_int) < 0:
        raise ValueError("target_count cannot be negative")

    colors = tuple(str(color).strip().lower() for color in color_support if str(color).strip())
    if len(set(colors)) != len(colors) or len(colors) < 2:
        raise ValueError("color_support must contain at least two unique color names")
    target_color = str(target_color_name).strip().lower()
    if target_color not in set(colors):
        raise ValueError("target_color_name is outside color_support")
    non_target_colors = tuple(str(color) for color in colors if str(color) != str(target_color))

    graph = _sample_edge_color_count_base_graph(
        rng,
        node_count=int(node_count_int),
        graph_directionality=str(directionality),
        topology_profile=str(topology_profile),
        max_degree=int(max_degree),
    )
    topology_sample, _label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=bool(directionality == "directed"),
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    edge_labels = tuple((str(left), str(right)) for left, right in topology_sample.edge_labels)
    if int(target_count_int) > len(edge_labels):
        raise ValueError("target_count must be no larger than the sampled edge count")
    if int(target_count_int) < len(edge_labels) and not non_target_colors:
        raise ValueError("non-target edges require at least one non-target color")

    target_edge_set = set(tuple(edge) for edge in rng.sample(edge_labels, int(target_count_int)))
    edge_color_names_by_label: Dict[Tuple[str, str], str] = {}
    for edge in edge_labels:
        canonical_edge = (str(edge[0]), str(edge[1]))
        if canonical_edge in target_edge_set:
            edge_color_names_by_label[canonical_edge] = str(target_color)
        else:
            edge_color_names_by_label[canonical_edge] = str(rng.choice(non_target_colors))
    color_counts_by_name = {
        str(color): sum(1 for edge in edge_labels if str(edge_color_names_by_label[(str(edge[0]), str(edge[1]))]) == str(color))
        for color in colors
    }
    target_edges = sort_graph_edge_labels(
        tuple((str(left), str(right)) for left, right in target_edge_set),
        directed=bool(directionality == "directed"),
    )
    if int(color_counts_by_name[str(target_color)]) != int(target_count_int):
        raise ValueError("edge-color sampler produced inconsistent target metadata")

    return GraphEdgeColorCountSample(
        graph=topology_sample.graph,
        directed=bool(topology_sample.directed),
        node_labels=tuple(str(label) for label in topology_sample.node_labels),
        edge_labels=tuple((str(left), str(right)) for left, right in topology_sample.edge_labels),
        degrees_by_label={str(key): int(value) for key, value in topology_sample.degrees_by_label.items()},
        in_degrees_by_label={str(key): int(value) for key, value in topology_sample.in_degrees_by_label.items()},
        out_degrees_by_label={str(key): int(value) for key, value in topology_sample.out_degrees_by_label.items()},
        adjacency_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.adjacency_by_label.items()},
        successors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.successors_by_label.items()},
        predecessors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.predecessors_by_label.items()},
        edge_count=int(topology_sample.edge_count),
        topology_profile=str(topology_sample.topology_profile),
        label_variant=str(topology_sample.label_variant),
        target_edges=tuple((str(left), str(right)) for left, right in target_edges),
        target_count=int(target_count_int),
        target_color_name=str(target_color),
        edge_color_names_by_label={
            (str(left), str(right)): str(color_name)
            for (left, right), color_name in edge_color_names_by_label.items()
        },
        color_counts_by_name={str(key): int(value) for key, value in color_counts_by_name.items()},
        graph_directionality=str(directionality),
    )


def sample_edge_text_label_count_graph(
    rng: random.Random,
    *,
    graph_directionality: str,
    node_count: int,
    target_count: int,
    target_edge_label: str,
    edge_label_support: Sequence[str],
    topology_profile: str,
    label_variant: str,
    max_degree: int,
    target_edge_label_index: int = 0,
    edge_label_support_resolver: EdgeLabelSupportResolver | None = None,
    max_labeled_edge_count: int | None = None,
) -> GraphEdgeTextLabelCountSample:
    """Construct one labeled graph with exactly ``target_count`` visible edge-text labels."""

    directionality = str(graph_directionality)
    if directionality not in SUPPORTED_EDGE_ATTRIBUTE_LABEL_DIRECTIONS:
        raise ValueError(f"unsupported graph_directionality: {graph_directionality}")
    node_count_int = int(node_count)
    target_count_int = int(target_count)
    if int(target_count_int) < 0:
        raise ValueError("target_count cannot be negative")

    graph = _sample_edge_color_count_base_graph(
        rng,
        node_count=int(node_count_int),
        graph_directionality=str(directionality),
        topology_profile=str(topology_profile),
        max_degree=int(max_degree),
        max_edge_count=max_labeled_edge_count,
    )
    topology_sample, _label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=bool(directionality == "directed"),
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    edge_labels = tuple((str(left), str(right)) for left, right in topology_sample.edge_labels)
    if max_labeled_edge_count is not None and len(edge_labels) > int(max_labeled_edge_count):
        raise ValueError("sampled graph exceeds max_labeled_edge_count")
    edge_labels_supported, target_label, edge_label_metadata = _resolve_visible_edge_label_support(
        rng,
        node_labels=tuple(str(label) for label in topology_sample.node_labels),
        target_edge_label=str(target_edge_label),
        edge_label_support=edge_label_support,
        target_edge_label_index=int(target_edge_label_index),
        edge_label_support_resolver=edge_label_support_resolver,
    )
    non_target_labels = tuple(str(label) for label in edge_labels_supported if str(label) != str(target_label))
    if not non_target_labels:
        raise ValueError("edge_label_support must include a non-target label")
    if int(target_count_int) > len(edge_labels):
        raise ValueError("target_count must be no larger than the sampled edge count")

    target_edge_set = set(tuple(edge) for edge in rng.sample(edge_labels, int(target_count_int)))
    edge_attribute_labels_by_label: Dict[Tuple[str, str], str] = {}
    for edge in edge_labels:
        canonical_edge = (str(edge[0]), str(edge[1]))
        if canonical_edge in target_edge_set:
            edge_attribute_labels_by_label[canonical_edge] = str(target_label)
        else:
            edge_attribute_labels_by_label[canonical_edge] = str(rng.choice(non_target_labels))
    edge_label_counts_by_value = {
        str(label): sum(
            1
            for edge in edge_labels
            if str(edge_attribute_labels_by_label[(str(edge[0]), str(edge[1]))]) == str(label)
        )
        for label in edge_labels_supported
    }
    target_edges = sort_graph_edge_labels(
        tuple((str(left), str(right)) for left, right in target_edge_set),
        directed=bool(directionality == "directed"),
    )
    if int(edge_label_counts_by_value[str(target_label)]) != int(target_count_int):
        raise ValueError("edge-text-label sampler produced inconsistent target metadata")

    return GraphEdgeTextLabelCountSample(
        graph=topology_sample.graph,
        directed=bool(topology_sample.directed),
        node_labels=tuple(str(label) for label in topology_sample.node_labels),
        edge_labels=tuple((str(left), str(right)) for left, right in topology_sample.edge_labels),
        degrees_by_label={str(key): int(value) for key, value in topology_sample.degrees_by_label.items()},
        in_degrees_by_label={str(key): int(value) for key, value in topology_sample.in_degrees_by_label.items()},
        out_degrees_by_label={str(key): int(value) for key, value in topology_sample.out_degrees_by_label.items()},
        adjacency_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.adjacency_by_label.items()},
        successors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.successors_by_label.items()},
        predecessors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.predecessors_by_label.items()},
        edge_count=int(topology_sample.edge_count),
        topology_profile=str(topology_sample.topology_profile),
        label_variant=str(topology_sample.label_variant),
        label_source_kind=str(topology_sample.label_source_kind),
        label_bucket=str(topology_sample.label_bucket),
        label_manifest=str(topology_sample.label_manifest),
        label_filter=dict(topology_sample.label_filter),
        label_bucket_probabilities=dict(topology_sample.label_bucket_probabilities),
        target_edges=tuple((str(left), str(right)) for left, right in target_edges),
        target_count=int(target_count_int),
        target_edge_label=str(target_label),
        edge_label_support=tuple(str(label) for label in edge_labels_supported),
        edge_attribute_labels_by_label={
            (str(left), str(right)): str(label)
            for (left, right), label in edge_attribute_labels_by_label.items()
        },
        edge_label_counts_by_value={str(key): int(value) for key, value in edge_label_counts_by_value.items()},
        graph_directionality=str(directionality),
        edge_label_source_kind=str(edge_label_metadata.get("edge_label_source_kind", "")),
        edge_label_bucket=str(edge_label_metadata.get("edge_label_bucket", "")),
        edge_label_manifest=str(edge_label_metadata.get("edge_label_manifest", "")),
        edge_label_filter=dict(edge_label_metadata.get("edge_label_filter", {})),
        edge_label_bucket_probabilities={
            str(key): float(value)
            for key, value in dict(edge_label_metadata.get("edge_label_bucket_probabilities", {})).items()
        },
    )


def sample_edge_attribute_label_graph(
    rng: random.Random,
    *,
    graph_directionality: str,
    node_count: int,
    target_edge_label: str,
    edge_label_support: Sequence[str],
    topology_profile: str,
    label_variant: str,
    max_degree: int,
    target_edge_label_index: int = 0,
    edge_label_support_resolver: EdgeLabelSupportResolver | None = None,
    max_labeled_edge_count: int | None = None,
) -> GraphEdgeAttributeLabelSample:
    """Construct one labeled graph with visible text labels on every edge."""

    directionality = str(graph_directionality)
    if directionality not in SUPPORTED_EDGE_ATTRIBUTE_LABEL_DIRECTIONS:
        raise ValueError(f"unsupported graph_directionality: {graph_directionality}")
    node_count_int = int(node_count)
    if int(node_count_int) < 2:
        raise ValueError("edge-attribute label graphs require at least two nodes")

    graph = _sample_edge_color_count_base_graph(
        rng,
        node_count=int(node_count_int),
        graph_directionality=str(directionality),
        topology_profile=str(topology_profile),
        max_degree=int(max_degree),
        max_edge_count=max_labeled_edge_count,
    )
    topology_sample, _label_by_node = _build_labeled_graph_topology_sample(
        rng,
        graph=graph,
        directed=bool(directionality == "directed"),
        topology_profile=str(topology_profile),
        label_variant=str(label_variant),
    )
    edge_labels = tuple((str(left), str(right)) for left, right in topology_sample.edge_labels)
    if not edge_labels:
        raise ValueError("edge-attribute label graph has no edges")
    if max_labeled_edge_count is not None and len(edge_labels) > int(max_labeled_edge_count):
        raise ValueError("sampled graph exceeds max_labeled_edge_count")
    edge_labels_supported, target_label, edge_label_metadata = _resolve_visible_edge_label_support(
        rng,
        node_labels=tuple(str(label) for label in topology_sample.node_labels),
        target_edge_label=str(target_edge_label),
        edge_label_support=edge_label_support,
        target_edge_label_index=int(target_edge_label_index),
        edge_label_support_resolver=edge_label_support_resolver,
    )

    query_edge = tuple(str(value) for value in rng.choice(edge_labels))
    non_target_labels = tuple(str(label) for label in edge_labels_supported if str(label) != str(target_label))
    edge_attribute_labels_by_label: Dict[Tuple[str, str], str] = {}
    for edge in edge_labels:
        canonical_edge = (str(edge[0]), str(edge[1]))
        if canonical_edge == tuple(query_edge):
            edge_attribute_labels_by_label[canonical_edge] = str(target_label)
        else:
            # Use all labels, including the target label, as distractors. The queried
            # edge endpoints make the answer unique, so repeated labels are valid.
            edge_attribute_labels_by_label[canonical_edge] = str(rng.choice(edge_labels_supported))
    if not non_target_labels:
        raise ValueError("edge_label_support must include a non-target label")
    edge_label_counts_by_value = {
        str(label): sum(
            1
            for edge in edge_labels
            if str(edge_attribute_labels_by_label[(str(edge[0]), str(edge[1]))]) == str(label)
        )
        for label in edge_labels_supported
    }

    return GraphEdgeAttributeLabelSample(
        graph=topology_sample.graph,
        directed=bool(topology_sample.directed),
        node_labels=tuple(str(label) for label in topology_sample.node_labels),
        edge_labels=tuple((str(left), str(right)) for left, right in topology_sample.edge_labels),
        degrees_by_label={str(key): int(value) for key, value in topology_sample.degrees_by_label.items()},
        in_degrees_by_label={str(key): int(value) for key, value in topology_sample.in_degrees_by_label.items()},
        out_degrees_by_label={str(key): int(value) for key, value in topology_sample.out_degrees_by_label.items()},
        adjacency_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.adjacency_by_label.items()},
        successors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.successors_by_label.items()},
        predecessors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.predecessors_by_label.items()},
        edge_count=int(topology_sample.edge_count),
        topology_profile=str(topology_sample.topology_profile),
        label_variant=str(topology_sample.label_variant),
        query_edge=(str(query_edge[0]), str(query_edge[1])),
        target_edge_label=str(target_label),
        edge_label_support=tuple(str(label) for label in edge_labels_supported),
        edge_attribute_labels_by_label={
            (str(left), str(right)): str(label)
            for (left, right), label in edge_attribute_labels_by_label.items()
        },
        edge_label_counts_by_value={str(key): int(value) for key, value in edge_label_counts_by_value.items()},
        graph_directionality=str(directionality),
        edge_label_source_kind=str(edge_label_metadata.get("edge_label_source_kind", "")),
        edge_label_bucket=str(edge_label_metadata.get("edge_label_bucket", "")),
        edge_label_manifest=str(edge_label_metadata.get("edge_label_manifest", "")),
        edge_label_filter=dict(edge_label_metadata.get("edge_label_filter", {})),
        edge_label_bucket_probabilities={
            str(key): float(value)
            for key, value in dict(edge_label_metadata.get("edge_label_bucket_probabilities", {})).items()
        },
    )


def _cross_color_edges_for_assignment(
    *,
    edge_labels: Sequence[Tuple[str, str]],
    node_color_names_by_label: Mapping[str, str],
    directed: bool,
    source_color_name: str,
    target_color_name: str,
) -> Tuple[Tuple[str, str], ...]:
    """Return edges matching one cross-color rule under a node-color assignment."""

    source_color = str(source_color_name)
    target_color = str(target_color_name)
    matches = []
    for left, right in edge_labels:
        left_label = str(left)
        right_label = str(right)
        left_color = str(node_color_names_by_label[left_label])
        right_color = str(node_color_names_by_label[right_label])
        if bool(directed):
            if left_color == source_color and right_color == target_color:
                matches.append((left_label, right_label))
        else:
            if {left_color, right_color} == {source_color, target_color}:
                matches.append((left_label, right_label))
    return sort_graph_edge_labels(tuple(matches), directed=bool(directed))


def _find_cross_color_node_assignment(
    rng: random.Random,
    *,
    labels: Sequence[str],
    edge_labels: Sequence[Tuple[str, str]],
    directed: bool,
    source_color_name: str,
    target_color_name: str,
    color_support: Sequence[str],
    target_count: int,
) -> Tuple[Dict[str, str], Tuple[Tuple[str, str], ...]]:
    """Find node colors that realize exactly ``target_count`` cross-color edges."""

    labels_tuple = tuple(str(label) for label in labels)
    colors = tuple(str(color).strip().lower() for color in color_support if str(color).strip())
    source_color = str(source_color_name).strip().lower()
    target_color = str(target_color_name).strip().lower()
    if source_color == target_color:
        raise ValueError("source_color_name and target_color_name must be distinct")
    if source_color not in set(colors) or target_color not in set(colors):
        raise ValueError("queried colors must be in color_support")
    neutral_colors = tuple(
        str(color) for color in colors if str(color) not in {str(source_color), str(target_color)}
    )
    if not neutral_colors:
        raise ValueError("cross-color edge counts require at least one non-query color")

    target_count_int = int(target_count)

    def build_assignment(states_by_label: Mapping[str, int]) -> Tuple[Dict[str, str], Tuple[Tuple[str, str], ...]] | None:
        state_values = set(int(value) for value in states_by_label.values())
        if 0 not in state_values or 1 not in state_values:
            return None
        color_by_label: Dict[str, str] = {}
        for label in labels_tuple:
            state = int(states_by_label[str(label)])
            if state == 0:
                color_by_label[str(label)] = str(source_color)
            elif state == 1:
                color_by_label[str(label)] = str(target_color)
            else:
                color_by_label[str(label)] = str(rng.choice(neutral_colors))
        matching_edges = _cross_color_edges_for_assignment(
            edge_labels=edge_labels,
            node_color_names_by_label=color_by_label,
            directed=bool(directed),
            source_color_name=str(source_color),
            target_color_name=str(target_color),
        )
        if int(len(matching_edges)) != int(target_count_int):
            return None
        return color_by_label, matching_edges

    for _attempt in range(4096):
        states = {str(label): int(rng.choice((0, 1, 2))) for label in labels_tuple}
        result = build_assignment(states)
        if result is not None:
            return result

    label_order = list(labels_tuple)
    rng.shuffle(label_order)
    state_values = [0, 1, 2]
    rng.shuffle(state_values)
    for state_tuple in product(tuple(state_values), repeat=len(label_order)):
        states = {
            str(label): int(state)
            for label, state in zip(label_order, state_tuple)
        }
        result = build_assignment(states)
        if result is not None:
            return result

    raise ValueError("no node-color assignment realizes the requested cross-color edge count")


def sample_cross_color_edge_count_graph(
    rng: random.Random,
    *,
    graph_directionality: str,
    node_count: int,
    target_count: int,
    source_color_name: str,
    target_color_name: str,
    color_support: Sequence[str],
    topology_profile: str,
    label_variant: str,
    max_degree: int,
) -> GraphCrossColorEdgeCountSample:
    """Construct a labeled graph with exactly ``target_count`` queried cross-color edges."""

    directionality = str(graph_directionality)
    if directionality not in SUPPORTED_CROSS_COLOR_EDGE_COUNT_DIRECTIONS:
        raise ValueError(f"unsupported graph_directionality: {graph_directionality}")
    node_count_int = int(node_count)
    target_count_int = int(target_count)
    if int(target_count_int) < 0:
        raise ValueError("target_count cannot be negative")

    colors = tuple(str(color).strip().lower() for color in color_support if str(color).strip())
    if len(set(colors)) != len(colors) or len(colors) < 3:
        raise ValueError("color_support must contain at least three unique color names")
    source_color = str(source_color_name).strip().lower()
    target_color = str(target_color_name).strip().lower()
    if source_color == target_color:
        raise ValueError("source_color_name and target_color_name must be distinct")
    if source_color not in set(colors) or target_color not in set(colors):
        raise ValueError("queried colors must be in color_support")

    last_error: Exception | None = None
    for _attempt in range(100):
        try:
            graph = _sample_node_color_count_base_graph(
                rng,
                node_count=int(node_count_int),
                graph_directionality=str(directionality),
                topology_profile=str(topology_profile),
                max_degree=int(max_degree),
            )
            topology_sample, _label_by_node = _build_labeled_graph_topology_sample(
                rng,
                graph=graph,
                directed=bool(directionality == "directed"),
                topology_profile=str(topology_profile),
                label_variant=str(label_variant),
            )
            edge_labels = tuple((str(left), str(right)) for left, right in topology_sample.edge_labels)
            if int(target_count_int) > int(len(edge_labels)):
                raise ValueError("target_count must be no larger than the sampled edge count")
            node_color_names_by_label, target_edges = _find_cross_color_node_assignment(
                rng,
                labels=topology_sample.node_labels,
                edge_labels=edge_labels,
                directed=bool(directionality == "directed"),
                source_color_name=str(source_color),
                target_color_name=str(target_color),
                color_support=colors,
                target_count=int(target_count_int),
            )
            color_counts_by_name = {
                str(color): sum(
                    1
                    for label in topology_sample.node_labels
                    if str(node_color_names_by_label[str(label)]) == str(color)
                )
                for color in colors
            }
            if int(len(target_edges)) != int(target_count_int):
                raise ValueError("cross-color sampler produced inconsistent target metadata")

            return GraphCrossColorEdgeCountSample(
                graph=topology_sample.graph,
                directed=bool(topology_sample.directed),
                node_labels=tuple(str(label) for label in topology_sample.node_labels),
                edge_labels=tuple((str(left), str(right)) for left, right in topology_sample.edge_labels),
                degrees_by_label={str(key): int(value) for key, value in topology_sample.degrees_by_label.items()},
                in_degrees_by_label={str(key): int(value) for key, value in topology_sample.in_degrees_by_label.items()},
                out_degrees_by_label={str(key): int(value) for key, value in topology_sample.out_degrees_by_label.items()},
                adjacency_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.adjacency_by_label.items()},
                successors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.successors_by_label.items()},
                predecessors_by_label={str(key): tuple(str(value) for value in values) for key, values in topology_sample.predecessors_by_label.items()},
                edge_count=int(topology_sample.edge_count),
                topology_profile=str(topology_sample.topology_profile),
                label_variant=str(topology_sample.label_variant),
                target_edges=tuple((str(left), str(right)) for left, right in target_edges),
                target_count=int(target_count_int),
                source_color_name=str(source_color),
                target_color_name=str(target_color),
                node_color_names_by_label={str(key): str(value) for key, value in node_color_names_by_label.items()},
                color_counts_by_name={str(key): int(value) for key, value in color_counts_by_name.items()},
                graph_directionality=str(directionality),
            )
        except Exception as exc:
            last_error = exc
            continue
    raise ValueError("failed to sample graph for cross-color edge count") from last_error


__all__ = [
    "_cross_color_edges_for_assignment",
    "_find_cross_color_node_assignment",
    "_sample_edge_color_count_base_graph",
    "_sample_node_color_count_base_graph",
    "_sample_unique_node_label_directed_graph",
    "_sample_unique_node_label_undirected_graph",
    "sample_cross_color_edge_count_graph",
    "sample_edge_attribute_label_graph",
    "sample_edge_color_count_graph",
    "sample_edge_text_label_count_graph",
    "sample_node_color_count_graph",
    "sample_unique_node_label_relation_graph",
]
