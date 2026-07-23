"""Sampling primitives for graph option-panel scenes."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.sampling import integer_range_choice, uniform_choice
from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default
from ....shared.deterministic_sampling import uniform_probability_map
from ....shared.mcq import option_label_for_index
from ....shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant
from .state import (
    LABEL_POOL,
    SUPPORTED_EDGE_MODES,
    SUPPORTED_SCENE_VARIANTS,
    GraphOptionsDataset,
    GraphOptionsDefaults,
)


def edge_key(u_label: str, v_label: str, *, directed: bool = False) -> Tuple[str, str]:
    """Return a stable edge key, preserving direction only when requested."""

    u, v = str(u_label), str(v_label)
    if u == v:
        raise ValueError("self edges are not supported")
    if directed:
        return (u, v)
    return tuple(sorted((u, v)))  # type: ignore[return-value]


def is_directed(spec: Mapping[str, Any]) -> bool:
    """Return whether a structure spec uses directed edges."""

    return bool(spec.get("directed", False))


def canonical_spec(spec: Mapping[str, Any]) -> str:
    """Return a stable JSON signature for one labeled graph spec."""

    labels = tuple(sorted(str(label) for label in spec["labels"]))
    directed = is_directed(spec)
    edges = tuple(sorted(edge_key(str(edge[0]), str(edge[1]), directed=directed) for edge in spec["edges"]))
    return json.dumps({"directed": directed, "labels": labels, "edges": edges}, sort_keys=True, separators=(",", ":"))


def edge_set(spec: Mapping[str, Any]) -> set[Tuple[str, str]]:
    """Return the canonical edge set for one structure spec."""

    directed = is_directed(spec)
    return {edge_key(str(edge[0]), str(edge[1]), directed=directed) for edge in spec["edges"]}


def spec_from(labels: Sequence[str], edges: Sequence[Sequence[str]], *, directed: bool = False) -> Dict[str, Any]:
    """Build a canonical structure spec from labels and edges."""

    label_list = [str(label) for label in labels]
    edge_spec = {"labels": label_list, "edges": edges, "directed": bool(directed)}
    return {
        "labels": list(label_list),
        "edges": [list(edge) for edge in sorted(edge_set(edge_spec))],
        "directed": bool(directed),
    }


def all_label_pairs(labels: Sequence[str], *, directed: bool = False) -> List[Tuple[str, str]]:
    """Return all possible non-self edges for a label set."""

    label_list = [str(label) for label in labels]
    if directed:
        return [
            edge_key(source, target, directed=True)
            for source in label_list
            for target in label_list
            if source != target
        ]
    return [
        edge_key(label_list[i], label_list[j])
        for i in range(len(label_list))
        for j in range(i + 1, len(label_list))
    ]


def would_create_antiparallel_edge(
    pair: Sequence[str],
    existing_edges: set[Tuple[str, str]],
    *,
    directed: bool,
) -> bool:
    """Return whether adding a directed edge would overlap its reverse edge."""

    if not bool(directed):
        return False
    source, target = str(pair[0]), str(pair[1])
    return (target, source) in existing_edges


def is_connected(labels: Sequence[str], edges: Sequence[Sequence[str]]) -> bool:
    """Return whether an undirected reading of the spec is connected."""

    label_list = [str(label) for label in labels]
    if not label_list:
        return False
    adjacency = {label: set() for label in label_list}
    for u_label, v_label in edges:
        u, v = str(u_label), str(v_label)
        if u in adjacency and v in adjacency:
            adjacency[u].add(v)
            adjacency[v].add(u)
    seen = {label_list[0]}
    stack = [label_list[0]]
    while stack:
        current = stack.pop()
        for neighbor in sorted(adjacency[current]):
            if neighbor not in seen:
                seen.add(neighbor)
                stack.append(neighbor)
    return len(seen) == len(label_list)


def make_connected_spec(
    rng: Any,
    *,
    node_count: int,
    extra_edge_min: int,
    extra_edge_max: int,
    directed: bool,
) -> Dict[str, Any]:
    """Sample one connected labeled graph spec."""

    labels = list(LABEL_POOL[:])
    rng.shuffle(labels)
    labels = sorted(labels[: int(node_count)])
    path_order = list(labels)
    rng.shuffle(path_order)
    edges: set[Tuple[str, str]] = set()
    for index in range(len(path_order) - 1):
        source, target = path_order[index], path_order[index + 1]
        if bool(directed) and bool(rng.randrange(2)):
            source, target = target, source
        edges.add(edge_key(source, target, directed=bool(directed)))

    available = [
        pair
        for pair in all_label_pairs(labels, directed=bool(directed))
        if pair not in edges and not would_create_antiparallel_edge(pair, edges, directed=bool(directed))
    ]
    rng.shuffle(available)
    max_extra = min(int(extra_edge_max), len(available))
    min_extra = min(int(extra_edge_min), max_extra)
    extra_count = int(rng.randint(int(min_extra), int(max_extra))) if max_extra >= min_extra else 0
    added_extra_count = 0
    for pair in available:
        if int(added_extra_count) >= int(extra_count):
            break
        if pair in edges or would_create_antiparallel_edge(pair, edges, directed=bool(directed)):
            continue
        edges.add(pair)
        added_extra_count += 1
    return spec_from(labels, sorted(edges), directed=bool(directed))


def mutate_edges(
    rng: Any,
    spec: Mapping[str, Any],
    *,
    toggle_count: int,
    require_connected: bool = True,
) -> Dict[str, Any]:
    """Toggle edges while preserving labels and, optionally, connectivity."""

    labels = [str(label) for label in spec["labels"]]
    directed = is_directed(spec)
    base_edges = edge_set(spec)
    for _ in range(160):
        edges = set(base_edges)
        candidates = list(all_label_pairs(labels, directed=directed))
        rng.shuffle(candidates)
        toggled = 0
        for pair in candidates:
            if toggled >= int(toggle_count):
                break
            if pair in edges:
                if len(edges) <= len(labels) - 1:
                    continue
                edges.remove(pair)
                if require_connected and not is_connected(labels, sorted(edges)):
                    edges.add(pair)
                    continue
            else:
                if would_create_antiparallel_edge(pair, edges, directed=directed):
                    continue
                edges.add(pair)
            toggled += 1
        if toggled == int(toggle_count):
            candidate = spec_from(labels, sorted(edges), directed=directed)
            if canonical_spec(candidate) != canonical_spec(spec):
                return candidate
    raise ValueError("failed to mutate structure edges")


def swap_two_labels(rng: Any, spec: Mapping[str, Any]) -> Dict[str, Any]:
    """Return an isomorphic-looking distractor with two labels swapped."""

    labels = [str(label) for label in spec["labels"]]
    directed = is_directed(spec)
    if len(labels) < 2:
        raise ValueError("need at least two labels to swap")
    a_idx, b_idx = rng.sample(range(len(labels)), 2)
    swapped = list(labels)
    swapped[a_idx], swapped[b_idx] = swapped[b_idx], swapped[a_idx]
    mapping = {old: new for old, new in zip(labels, swapped)}
    edges = [edge_key(mapping[str(edge[0])], mapping[str(edge[1])], directed=directed) for edge in spec["edges"]]
    return spec_from(sorted(swapped), edges, directed=directed)


def contains_pattern(pattern: Mapping[str, Any], candidate: Mapping[str, Any]) -> bool:
    """Return whether all pattern labels and edges appear in the candidate."""

    if is_directed(pattern) != is_directed(candidate):
        return False
    pattern_labels = set(str(label) for label in pattern["labels"])
    candidate_labels = set(str(label) for label in candidate["labels"])
    if not pattern_labels.issubset(candidate_labels):
        return False
    return edge_set(pattern).issubset(edge_set(candidate))


def sample_connected_label_subset(rng: Any, spec: Mapping[str, Any], *, subset_size: int) -> List[str]:
    """Sample labels that induce a connected portion of the larger graph."""

    labels = [str(label) for label in spec["labels"]]
    adjacency = {label: set() for label in labels}
    for u_label, v_label in spec["edges"]:
        adjacency[str(u_label)].add(str(v_label))
        adjacency[str(v_label)].add(str(u_label))
    for _ in range(100):
        start = str(rng.choice(labels))
        subset = [start]
        frontier = sorted(adjacency[start])
        while len(subset) < int(subset_size) and frontier:
            rng.shuffle(frontier)
            candidate = str(frontier.pop())
            if candidate in subset:
                continue
            subset.append(candidate)
            frontier.extend(sorted(adjacency[candidate] - set(subset)))
        if len(subset) == int(subset_size):
            return sorted(subset)
    return sorted(labels[: int(subset_size)])


def subspec_from_labels(spec: Mapping[str, Any], labels: Sequence[str]) -> Dict[str, Any]:
    """Return the induced subgraph over the requested labels."""

    subset = set(str(label) for label in labels)
    edges = [
        edge_key(str(edge[0]), str(edge[1]), directed=is_directed(spec))
        for edge in spec["edges"]
        if str(edge[0]) in subset and str(edge[1]) in subset
    ]
    if not edges:
        raise ValueError("pattern must include at least one edge")
    return spec_from(sorted(subset), edges, directed=is_directed(spec))


def replacement_label(existing: Sequence[str]) -> str:
    """Return a label outside the existing label set."""

    taken = set(str(label) for label in existing)
    for label in LABEL_POOL:
        if label not in taken:
            return str(label)
    raise ValueError("no replacement labels left")


def resolve_int_range(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
    namespace: str,
    instance_seed: int,
) -> Tuple[int, Tuple[int, int], Dict[str, float]]:
    """Resolve one bounded integer generation axis."""

    lower = int(params.get(str(min_key), group_default(gen_defaults, str(min_key), int(fallback_min))))
    upper = int(params.get(str(max_key), group_default(gen_defaults, str(max_key), int(fallback_max))))
    if int(lower) > int(upper):
        raise ValueError(f"{min_key} must be <= {max_key}")
    explicit_key = str(min_key).removesuffix("_min")
    explicit = params.get(explicit_key)
    if explicit is not None:
        value = int(explicit)
        if not int(lower) <= value <= int(upper):
            raise ValueError(f"{explicit_key} must fall in [{lower}, {upper}]")
        return int(value), (int(lower), int(upper)), dict(uniform_probability_map(tuple(range(lower, upper + 1)), selected=value))
    value, probabilities = integer_range_choice(
        spawn_rng(int(instance_seed), str(namespace)),
        int(lower),
        int(upper),
    )
    return int(value), (int(lower), int(upper)), dict(probabilities)


def resolve_scene_variant(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve scene style variation without public task identity."""

    selected_variant, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), f"{namespace}.scene_variant"),
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
    )
    variant = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        selected_variant=str(selected_variant),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        balance_flag_key="balanced_scene_variant_sampling",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        sampling_namespace=f"{namespace}.scene_variant",
    )
    return str(variant), {str(key): float(value) for key, value in sorted(probabilities.items())}


def resolve_edge_mode(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve whether the displayed graph options are directed."""

    selected_variant, probabilities = resolve_variant(
        spawn_rng(int(instance_seed), f"{namespace}.edge_mode"),
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=SUPPORTED_EDGE_MODES,
        explicit_key="edge_mode",
        weights_key="edge_mode_weights",
    )
    variant = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        selected_variant=str(selected_variant),
        variant_probabilities=probabilities,
        supported_variants=SUPPORTED_EDGE_MODES,
        balance_flag_key="balanced_edge_mode_sampling",
        explicit_key="edge_mode",
        weights_key="edge_mode_weights",
        sampling_namespace=f"{namespace}.edge_mode",
    )
    return str(variant), {str(key): float(value) for key, value in sorted(probabilities.items())}


def resolve_correct_option_index(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    option_count: int,
    namespace: str,
) -> int:
    """Resolve the correct visual option slot."""

    explicit = params.get("correct_option_index")
    if explicit is not None:
        index = int(explicit)
        if not 0 <= int(index) < int(option_count):
            raise ValueError("correct_option_index must fall inside option count")
        return int(index)
    if int(option_count) <= 0:
        raise ValueError("option_count must be positive")
    return int(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{namespace}.correct_option_index"),
            tuple(range(int(option_count))),
        )
    )


def build_options(
    *,
    correct_spec: Mapping[str, Any],
    distractors: Sequence[Mapping[str, Any]],
    correct_option_index: int,
    option_count: int,
) -> List[Dict[str, Any]]:
    """Return ordered option-panel specs with one correct graph."""

    seen = {canonical_spec(correct_spec)}
    unique_distractors: List[Dict[str, Any]] = []
    for spec in distractors:
        signature = canonical_spec(spec)
        if signature in seen:
            continue
        seen.add(signature)
        unique_distractors.append(dict(spec))
    if len(unique_distractors) < int(option_count) - 1:
        raise ValueError("not enough unique structure distractors")
    ordered = unique_distractors[: int(option_count) - 1]
    ordered.insert(int(correct_option_index), dict(correct_spec))
    options: List[Dict[str, Any]] = []
    for index, spec in enumerate(ordered):
        label = str(option_label_for_index(int(index)))
        options.append(
            {
                "option_panel_id": f"option_{label}",
                "option_index": int(index),
                "option_label": str(label),
                "structure_spec": dict(spec),
                "structure_signature": canonical_spec(spec),
                "is_correct": bool(index == int(correct_option_index)),
            }
        )
    return options


def build_same_structure_dataset(
    *,
    rng: Any,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    option_count: int,
    correct_option_index: int,
    edge_mode: str,
    namespace: str,
    defaults: GraphOptionsDefaults = GraphOptionsDefaults(),
) -> GraphOptionsDataset:
    """Build a Reference/options dataset where one option exactly matches."""

    directed = str(edge_mode) == "directed"
    node_count, node_count_range, node_count_probs = resolve_int_range(
        params,
        gen_defaults,
        min_key="same_structure_node_count_min",
        max_key="same_structure_node_count_max",
        fallback_min=defaults.same_structure_node_count_min,
        fallback_max=defaults.same_structure_node_count_max,
        namespace=f"{namespace}.node_count",
        instance_seed=int(instance_seed),
    )
    base = make_connected_spec(
        rng,
        node_count=int(node_count),
        extra_edge_min=int(params.get("extra_edge_min", group_default(gen_defaults, "extra_edge_min", defaults.extra_edge_min))),
        extra_edge_max=int(params.get("extra_edge_max", group_default(gen_defaults, "extra_edge_max", defaults.extra_edge_max))),
        directed=bool(directed),
    )
    distractors: List[Dict[str, Any]] = []
    for delta in (1, 2, 1, 2, 3, 1, 2, 3, 1):
        try:
            distractors.append(mutate_edges(rng, base, toggle_count=int(delta), require_connected=True))
        except ValueError:
            continue
    while len({canonical_spec(item) for item in distractors}) < int(option_count) - 1:
        distractors.append(swap_two_labels(rng, mutate_edges(rng, base, toggle_count=1, require_connected=True)))
    options = build_options(
        correct_spec=base,
        distractors=distractors,
        correct_option_index=int(correct_option_index),
        option_count=int(option_count),
    )
    return GraphOptionsDataset(
        panel_title="Reference",
        source_structure_spec=dict(base),
        answer_structure_spec=dict(base),
        option_specs=options,
        answer_option_label=str(option_label_for_index(int(correct_option_index))),
        correct_option_index=int(correct_option_index),
        correct_option_panel_id=f"option_{option_label_for_index(int(correct_option_index))}",
        option_count=int(option_count),
        node_count=int(node_count),
        node_count_range=list(node_count_range),
        node_count_probabilities=dict(node_count_probs),
        edge_mode=str(edge_mode),
        solver_trace={"rule": "same labeled graph structure, layout ignored", "edge_mode": str(edge_mode)},
    )


def build_contained_subgraph_dataset(
    *,
    rng: Any,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    option_count: int,
    correct_option_index: int,
    edge_mode: str,
    namespace: str,
    defaults: GraphOptionsDefaults = GraphOptionsDefaults(),
) -> GraphOptionsDataset:
    """Build a Target/options dataset where one option is contained in the target."""

    directed = str(edge_mode) == "directed"
    node_count, node_count_range, node_count_probs = resolve_int_range(
        params,
        gen_defaults,
        min_key="node_count_min",
        max_key="node_count_max",
        fallback_min=defaults.node_count_min,
        fallback_max=defaults.node_count_max,
        namespace=f"{namespace}.node_count",
        instance_seed=int(instance_seed),
    )
    subgraph_count, subgraph_count_range, subgraph_count_probs = resolve_int_range(
        params,
        gen_defaults,
        min_key="subgraph_node_count_min",
        max_key="subgraph_node_count_max",
        fallback_min=defaults.subgraph_node_count_min,
        fallback_max=defaults.subgraph_node_count_max,
        namespace=f"{namespace}.subgraph_node_count",
        instance_seed=int(instance_seed),
    )
    subgraph_count = min(int(subgraph_count), int(node_count) - 1)
    base = make_connected_spec(
        rng,
        node_count=int(node_count),
        extra_edge_min=int(params.get("extra_edge_min", group_default(gen_defaults, "extra_edge_min", defaults.extra_edge_min))),
        extra_edge_max=int(params.get("extra_edge_max", group_default(gen_defaults, "extra_edge_max", defaults.extra_edge_max))),
        directed=bool(directed),
    )
    pattern_labels = sample_connected_label_subset(rng, base, subset_size=int(subgraph_count))
    pattern = subspec_from_labels(base, pattern_labels)
    distractors: List[Dict[str, Any]] = []
    for _ in range(120):
        candidate_labels = sample_connected_label_subset(rng, base, subset_size=int(subgraph_count))
        candidate = subspec_from_labels(base, candidate_labels)
        if contains_pattern(candidate, base):
            candidate_edges = set(edge_set(candidate))
            all_pairs = [
                pair
                for pair in all_label_pairs(candidate["labels"], directed=directed)
                if pair not in candidate_edges
                and not would_create_antiparallel_edge(pair, candidate_edges, directed=directed)
            ]
            rng.shuffle(all_pairs)
            if all_pairs:
                candidate_edges.add(all_pairs[0])
                candidate = spec_from(candidate["labels"], sorted(candidate_edges), directed=directed)
        if contains_pattern(candidate, base):
            labels = [str(label) for label in candidate["labels"]]
            replace_label = str(rng.choice(labels))
            replacement = replacement_label(base["labels"])
            mapped = {label: (replacement if label == replace_label else label) for label in labels}
            candidate = spec_from(
                sorted(mapped.values()),
                [edge_key(mapped[str(edge[0])], mapped[str(edge[1])], directed=directed) for edge in candidate["edges"]],
                directed=directed,
            )
        if not contains_pattern(candidate, base):
            distractors.append(candidate)
        if len({canonical_spec(item) for item in distractors}) >= int(option_count) - 1:
            break
    if len({canonical_spec(item) for item in distractors}) < int(option_count) - 1:
        for _ in range(60):
            labels = [str(label) for label in base["labels"]]
            replace_label = str(rng.choice(pattern_labels))
            replacement = replacement_label(base["labels"])
            mapped = {label: (replacement if label == replace_label else label) for label in labels}
            candidate = spec_from(
                sorted(mapped.values()),
                [edge_key(mapped[str(edge[0])], mapped[str(edge[1])], directed=directed) for edge in pattern["edges"]],
                directed=directed,
            )
            if not contains_pattern(candidate, base):
                distractors.append(candidate)
            if len({canonical_spec(item) for item in distractors}) >= int(option_count) - 1:
                break
    options = build_options(
        correct_spec=pattern,
        distractors=distractors,
        correct_option_index=int(correct_option_index),
        option_count=int(option_count),
    )
    return GraphOptionsDataset(
        panel_title="Target Graph",
        source_structure_spec=dict(base),
        answer_structure_spec=dict(pattern),
        option_specs=options,
        answer_option_label=str(option_label_for_index(int(correct_option_index))),
        correct_option_index=int(correct_option_index),
        correct_option_panel_id=f"option_{option_label_for_index(int(correct_option_index))}",
        option_count=int(option_count),
        node_count=int(node_count),
        node_count_range=list(node_count_range),
        node_count_probabilities=dict(node_count_probs),
        edge_mode=str(edge_mode),
        solver_trace={"rule": "selected option is contained in the larger graph", "edge_mode": str(edge_mode)},
        pattern_node_count=int(subgraph_count),
        pattern_node_count_range=list(subgraph_count_range),
        pattern_node_count_probability_map=dict(subgraph_count_probs),
        subgraph_node_count=int(subgraph_count),
        subgraph_node_count_range=list(subgraph_count_range),
        subgraph_node_count_probability_map=dict(subgraph_count_probs),
    )


def option_count_from_params(
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    *,
    defaults: GraphOptionsDefaults = GraphOptionsDefaults(),
) -> int:
    """Return the configured number of visual options."""

    return int(params.get("option_count", group_default(gen_defaults, "option_count", defaults.option_count)))


__all__ = [
    "all_label_pairs",
    "build_contained_subgraph_dataset",
    "build_options",
    "build_same_structure_dataset",
    "canonical_spec",
    "contains_pattern",
    "edge_key",
    "edge_set",
    "is_directed",
    "make_connected_spec",
    "mutate_edges",
    "option_count_from_params",
    "replacement_label",
    "resolve_correct_option_index",
    "resolve_edge_mode",
    "resolve_int_range",
    "resolve_scene_variant",
    "sample_connected_label_subset",
    "spec_from",
    "subspec_from_labels",
    "swap_two_labels",
    "would_create_antiparallel_edge",
]
