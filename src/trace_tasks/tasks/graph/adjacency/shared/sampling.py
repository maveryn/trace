"""Neutral sampling helpers for adjacency-scene graph representations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.sampling import uniform_choice
from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default
from ....shared.deterministic_sampling import uniform_probability_map
from ...shared.graph_sample_types import SUPPORTED_NODE_LINK_LABEL_VARIANTS
from ...shared.label_assets import default_graph_label_bucket_weights, resolve_graph_node_labels
from ...shared.task_support import graph_int_support, resolve_graph_named_variant
from .state import (
    AdjacencyGraphSample,
    AdjacencyLabelSet,
    SUPPORTED_ADJACENCY_REPRESENTATION_VARIANTS,
    canonical_undirected_edge,
)


@dataclass(frozen=True)
class ResolvedIntAxis:
    """Resolved integer axis value and its support probabilities."""

    value: int
    probabilities: Dict[str, float]
    support: tuple[int, ...]


@dataclass(frozen=True)
class ResolvedLabelVariantAxis:
    """Resolved node-label variant and its support probabilities."""

    value: str
    probabilities: Dict[str, float]


@dataclass(frozen=True)
class ResolvedLabelNodeAxes:
    """Common label and node-count axes for adjacency tasks."""

    label: ResolvedLabelVariantAxis
    node: ResolvedIntAxis


@dataclass(frozen=True)
class ComponentCountDefaults:
    """Scene-local fallback defaults shared by component count objectives."""

    node_count_min: int = 6
    node_count_max: int = 9
    component_count_min: int = 2
    component_count_max: int = 6
    extra_edge_count_min: int = 1
    extra_edge_count_max: int = 3
    label_max_chars: int = 5
    label_variant: str = "letters"
    scene_variant: str = "adjacency_list_panel"
    canvas_width: int = 900
    canvas_height: int = 640
    label_font_size_px: int = 19


@dataclass(frozen=True)
class ComponentCountAxes:
    """Resolved scene axes for one component-count instance."""

    scene_variant: str
    node_count: int
    component_count: int
    extra_edge_count: int
    label_variant: str
    scene_variant_probabilities: Dict[str, float]
    node_count_probabilities: Dict[str, float]
    component_count_probabilities: Dict[str, float]
    extra_edge_count_probabilities: Dict[str, float]
    label_variant_probabilities: Dict[str, float]


def resolve_adjacency_int_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    axis_name: str,
    default_min: int,
    default_max: int,
    rng_namespace: str,
    support: Sequence[int] | None = None,
    max_value: int | None = None,
) -> ResolvedIntAxis:
    """Resolve one integer support axis for an adjacency task."""

    configured = (
        tuple(int(value) for value in support)
        if support is not None
        else graph_int_support(params, gen_defaults, axis_name, int(default_min), int(default_max))
    )
    feasible = tuple(int(value) for value in configured if max_value is None or int(value) <= int(max_value))
    if not feasible:
        raise ValueError(f"{axis_name} support is empty")
    explicit = params.get(str(axis_name))
    if explicit is not None:
        value = int(explicit)
        if int(value) not in set(feasible):
            raise ValueError(f"{axis_name} is outside configured support")
    else:
        value = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{str(rng_namespace)}:{str(axis_name)}"),
                feasible,
            )
        )
    return ResolvedIntAxis(
        value=int(value),
        probabilities=uniform_probability_map(feasible, selected=int(value) if explicit is not None else None),
        support=tuple(feasible),
    )


def resolve_adjacency_label_variant_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    rng_namespace: str,
) -> ResolvedLabelVariantAxis:
    """Resolve the node-label variant axis for an adjacency task."""

    label_variant, label_probs = resolve_graph_named_variant(
        spawn_rng(int(instance_seed), f"{str(rng_namespace)}.label_variant"),
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="label_variant",
        weights_key="label_variant_weights",
        balance_flag_key="balanced_label_variant_sampling",
        supported=SUPPORTED_NODE_LINK_LABEL_VARIANTS,
        instance_seed=int(instance_seed),
        task_id=str(rng_namespace),
        namespace="label_variant",
    )
    return ResolvedLabelVariantAxis(value=str(label_variant), probabilities=dict(label_probs))


def configured_axis_max(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    axis_name: str,
    default_max: int,
) -> int:
    """Return the configured max value for a min/max integer support axis."""

    return int(params.get(f"{str(axis_name)}_max", group_default(gen_defaults, f"{str(axis_name)}_max", int(default_max))))


def configured_axis_min(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    axis_name: str,
    default_min: int,
) -> int:
    """Return the configured min value for a min/max integer support axis."""

    return int(params.get(f"{str(axis_name)}_min", group_default(gen_defaults, f"{str(axis_name)}_min", int(default_min))))


def resolve_label_node_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    node_count_min: int,
    node_count_max: int,
    rng_namespace: str,
) -> ResolvedLabelNodeAxes:
    """Resolve repeated label and node-count axes for adjacency tasks."""

    label_axis = resolve_adjacency_label_variant_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        rng_namespace=str(rng_namespace),
    )
    node_axis = resolve_adjacency_int_axis(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        axis_name="node_count",
        default_min=int(node_count_min),
        default_max=int(node_count_max),
        rng_namespace=str(rng_namespace),
    )
    return ResolvedLabelNodeAxes(label=label_axis, node=node_axis)


def resolve_component_count_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    defaults: ComponentCountDefaults,
    rng_namespace: str,
) -> ComponentCountAxes:
    """Resolve component-count supports without knowing the public objective."""

    scene_variant, scene_probs = resolve_graph_named_variant(
        spawn_rng(int(instance_seed), f"{str(rng_namespace)}.scene_variant"),
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        supported=SUPPORTED_ADJACENCY_REPRESENTATION_VARIANTS,
        instance_seed=int(instance_seed),
        task_id=str(rng_namespace),
        namespace="scene_variant",
    )
    label_variant, label_probs = resolve_graph_named_variant(
        spawn_rng(int(instance_seed), f"{str(rng_namespace)}.label_variant"),
        params=params,
        gen_defaults=gen_defaults,
        explicit_key="label_variant",
        weights_key="label_variant_weights",
        balance_flag_key="balanced_label_variant_sampling",
        supported=SUPPORTED_NODE_LINK_LABEL_VARIANTS,
        instance_seed=int(instance_seed),
        task_id=str(rng_namespace),
        namespace="label_variant",
    )

    node_support = graph_int_support(params, gen_defaults, "node_count", defaults.node_count_min, defaults.node_count_max)
    explicit_node = params.get("node_count")
    if explicit_node is not None:
        node_count = int(explicit_node)
        if int(node_count) not in set(node_support):
            raise ValueError("node_count is outside configured support")
    else:
        node_count = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{str(rng_namespace)}:node_count"),
                node_support,
            )
        )

    component_max = min(
        int(node_count),
        int(params.get("component_count_max", group_default(gen_defaults, "component_count_max", defaults.component_count_max))),
    )
    component_min = int(params.get("component_count_min", group_default(gen_defaults, "component_count_min", defaults.component_count_min)))
    component_support = tuple(range(int(component_min), int(component_max) + 1))
    if not component_support:
        raise ValueError("component_count support is empty")
    explicit_component = params.get("component_count")
    if explicit_component is not None:
        component_count = int(explicit_component)
        if int(component_count) not in set(component_support):
            raise ValueError("component_count is outside configured support")
    else:
        component_count = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{str(rng_namespace)}:component_count"),
                component_support,
            )
        )

    extra_support = graph_int_support(params, gen_defaults, "extra_edge_count", defaults.extra_edge_count_min, defaults.extra_edge_count_max)
    explicit_extra = params.get("extra_edge_count")
    if explicit_extra is not None:
        extra_edge_count = int(explicit_extra)
        if int(extra_edge_count) not in set(extra_support):
            raise ValueError("extra_edge_count is outside configured support")
    else:
        extra_edge_count = int(
            uniform_choice(
                spawn_rng(int(instance_seed), f"{str(rng_namespace)}:extra_edge_count"),
                extra_support,
            )
        )

    return ComponentCountAxes(
        scene_variant=str(scene_variant),
        node_count=int(node_count),
        component_count=int(component_count),
        extra_edge_count=int(extra_edge_count),
        label_variant=str(label_variant),
        scene_variant_probabilities=dict(scene_probs),
        node_count_probabilities=uniform_probability_map(node_support, selected=int(node_count) if explicit_node is not None else None),
        component_count_probabilities=uniform_probability_map(
            component_support,
            selected=int(component_count) if explicit_component is not None else None,
        ),
        extra_edge_count_probabilities=uniform_probability_map(
            extra_support,
            selected=int(extra_edge_count) if explicit_extra is not None else None,
        ),
        label_variant_probabilities=dict(label_probs),
    )


def resolve_adjacency_labels(
    *,
    instance_seed: int,
    rng_namespace: str,
    label_variant: str,
    node_count: int,
    max_chars: int,
    min_chars: int | None = None,
    bucket_weights: Mapping[str, float] | None = None,
) -> AdjacencyLabelSet:
    """Resolve compact node labels for adjacency panels."""

    resolved = resolve_graph_node_labels(
        spawn_rng(int(instance_seed), f"{str(rng_namespace)}.labels"),
        label_variant=str(label_variant),
        object_count=int(node_count),
        max_chars=int(max_chars),
        min_chars=min_chars,
        bucket_weights=bucket_weights or default_graph_label_bucket_weights(),
        sequential_numbers=True,
    )
    return AdjacencyLabelSet(
        labels=tuple(str(label) for label in resolved.labels),
        label_variant=str(resolved.label_variant),
        label_source_kind=str(resolved.label_source_kind),
        label_bucket=str(resolved.label_bucket),
        label_manifest=str(resolved.label_manifest),
        label_filter=dict(resolved.label_filter),
        label_bucket_probabilities=dict(resolved.label_bucket_probabilities),
    )


def edge_set_from_adjacency(
    adjacency: Mapping[str, Sequence[str]],
    *,
    directed: bool,
) -> Tuple[Tuple[str, str], ...]:
    """Return stable edge ids for one adjacency mapping."""

    edges: set[Tuple[str, str]] = set()
    for source, targets in adjacency.items():
        for target in targets:
            if bool(directed):
                edges.add((str(source), str(target)))
            else:
                edges.add(canonical_undirected_edge(str(source), str(target)))
    return tuple(sorted(edges))


def sample_reachable_directed_adjacency(
    *,
    instance_seed: int,
    rng_namespace: str,
    labels: Sequence[str],
    source_label: str,
    extra_edge_count: int,
) -> AdjacencyGraphSample:
    """Sample a directed graph where every node is reachable from source."""

    rng = spawn_rng(int(instance_seed), f"{str(rng_namespace)}.reachable_digraph")
    label_order = tuple(str(label) for label in labels)
    source = str(source_label)
    remaining = [label for label in label_order if label != source]
    rng.shuffle(remaining)
    reached = [source]
    edges: set[Tuple[str, str]] = set()
    for label in remaining:
        parent = str(rng.choice(reached))
        edges.add((parent, str(label)))
        reached.append(str(label))
    all_pairs = [(left, right) for left in label_order for right in label_order if left != right]
    rng.shuffle(all_pairs)
    for edge in all_pairs:
        if len(edges) >= (len(label_order) - 1) + int(extra_edge_count):
            break
        edges.add((str(edge[0]), str(edge[1])))
    adjacency_lists: Dict[str, List[str]] = {label: [] for label in label_order}
    for left, right in sorted(edges):
        adjacency_lists[str(left)].append(str(right))
    for targets in adjacency_lists.values():
        rng.shuffle(targets)
    adjacency = {label: tuple(targets) for label, targets in adjacency_lists.items()}
    return AdjacencyGraphSample(
        labels=label_order,
        directed=True,
        adjacency=adjacency,
        edges=edge_set_from_adjacency(adjacency, directed=True),
        weights={},
    )


def partition_labels(labels: Sequence[str], component_count: int, rng: Any) -> Tuple[Tuple[str, ...], ...]:
    """Partition labels into non-empty component groups."""

    shuffled = [str(label) for label in labels]
    rng.shuffle(shuffled)
    count = max(1, min(int(component_count), len(shuffled)))
    groups: List[List[str]] = [[] for _ in range(count)]
    for index, label in enumerate(shuffled):
        groups[int(index % count)].append(str(label))
    return tuple(tuple(group) for group in groups if group)


def sample_component_adjacency(
    *,
    instance_seed: int,
    rng_namespace: str,
    labels: Sequence[str],
    component_count: int,
    directed: bool,
    extra_edge_count: int,
) -> AdjacencyGraphSample:
    """Sample an undirected component graph or directed SCC graph."""

    rng = spawn_rng(int(instance_seed), f"{str(rng_namespace)}.components.{int(component_count)}.{bool(directed)}")
    label_order = tuple(str(label) for label in labels)
    components = partition_labels(label_order, int(component_count), rng)
    adjacency_lists: Dict[str, List[str]] = {label: [] for label in label_order}
    edges: set[Tuple[str, str]] = set()

    for component in components:
        nodes = list(component)
        if len(nodes) <= 1:
            continue
        if bool(directed):
            for index, source in enumerate(nodes):
                target = nodes[(index + 1) % len(nodes)]
                edges.add((str(source), str(target)))
            if len(nodes) > 2:
                edges.add((str(nodes[0]), str(nodes[2])))
        else:
            for left, right in zip(nodes, nodes[1:]):
                edges.add(canonical_undirected_edge(str(left), str(right)))
            possible = [canonical_undirected_edge(a, b) for idx, a in enumerate(nodes) for b in nodes[idx + 1 :]]
            rng.shuffle(possible)
            for edge in possible[: max(0, int(extra_edge_count))]:
                edges.add(edge)

    if bool(directed) and len(components) >= 2:
        for left_component, right_component in zip(components, components[1:]):
            edges.add((str(left_component[0]), str(right_component[0])))

    for left, right in sorted(edges):
        adjacency_lists[str(left)].append(str(right))
        if not bool(directed):
            adjacency_lists[str(right)].append(str(left))
    for targets in adjacency_lists.values():
        rng.shuffle(targets)

    return AdjacencyGraphSample(
        labels=label_order,
        directed=bool(directed),
        adjacency={label: tuple(targets) for label, targets in adjacency_lists.items()},
        edges=edge_set_from_adjacency(adjacency_lists, directed=bool(directed)),
        weights={},
        components=tuple(tuple(node for node in component) for component in components),
    )


def sample_weighted_matrix_mst_adjacency(
    *,
    instance_seed: int,
    rng_namespace: str,
    labels: Sequence[str],
    extra_edge_count: int,
    edge_weight_min: int,
    edge_weight_max: int,
) -> AdjacencyGraphSample:
    """Sample a connected weighted graph with a unique obvious MST."""

    rng = spawn_rng(int(instance_seed), f"{str(rng_namespace)}.weighted_matrix_mst")
    label_order = tuple(str(label) for label in labels)
    shuffled = list(label_order)
    rng.shuffle(shuffled)
    tree_edges: List[Tuple[str, str]] = []
    connected = [shuffled[0]]
    for label in shuffled[1:]:
        parent = str(rng.choice(connected))
        tree_edges.append(canonical_undirected_edge(parent, str(label)))
        connected.append(str(label))

    low_max = max(int(edge_weight_min), min(int(edge_weight_max), int(edge_weight_min) + 4))
    tree_weights = {edge: int(rng.randint(int(edge_weight_min), int(low_max))) for edge in tree_edges}
    max_tree_weight = max(tree_weights.values()) if tree_weights else int(edge_weight_min)
    non_tree_min = min(int(edge_weight_max), int(max_tree_weight) + 2)
    if int(non_tree_min) > int(edge_weight_max):
        non_tree_min = int(edge_weight_max)

    possible = [
        canonical_undirected_edge(left, right)
        for idx, left in enumerate(label_order)
        for right in label_order[idx + 1 :]
        if canonical_undirected_edge(left, right) not in set(tree_edges)
    ]
    rng.shuffle(possible)
    extra_edges = possible[: max(0, int(extra_edge_count))]
    weights: Dict[Tuple[str, str], int] = dict(tree_weights)
    for edge in extra_edges:
        weights[edge] = int(rng.randint(int(non_tree_min), int(edge_weight_max)))

    adjacency_lists: Dict[str, List[str]] = {label: [] for label in label_order}
    for left, right in sorted(weights):
        adjacency_lists[str(left)].append(str(right))
        adjacency_lists[str(right)].append(str(left))
    for targets in adjacency_lists.values():
        targets.sort(key=lambda label: label_order.index(str(label)))

    return AdjacencyGraphSample(
        labels=label_order,
        directed=False,
        adjacency={label: tuple(targets) for label, targets in adjacency_lists.items()},
        edges=tuple(sorted(weights)),
        weights=dict(weights),
        mst_edges=tuple(sorted(tree_edges)),
        mst_weight=int(sum(tree_weights.values())),
    )


__all__ = [
    "ComponentCountAxes",
    "ComponentCountDefaults",
    "ResolvedIntAxis",
    "ResolvedLabelVariantAxis",
    "ResolvedLabelNodeAxes",
    "configured_axis_max",
    "configured_axis_min",
    "edge_set_from_adjacency",
    "partition_labels",
    "resolve_adjacency_int_axis",
    "resolve_adjacency_label_variant_axis",
    "resolve_adjacency_labels",
    "resolve_component_count_axes",
    "resolve_label_node_axes",
    "sample_component_adjacency",
    "sample_reachable_directed_adjacency",
    "sample_weighted_matrix_mst_adjacency",
]
