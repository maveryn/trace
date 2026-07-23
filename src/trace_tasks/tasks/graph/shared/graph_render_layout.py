"""Layout projection helpers for node-link graph rendering."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Mapping, Sequence, Tuple

import networkx as nx

from .graph_sample_types import GraphTopologySample
from .graph_render_types import BBox, Point
from .graph_render_geometry import _segment_intersects_bbox


def _scale_layout_to_content(
    raw_positions: Mapping[int, Sequence[float]],
    *,
    content_bbox: BBox,
    node_radius_px: int,
) -> Dict[int, Point]:
    """Scale raw layout positions into pixel centers inside one content box."""

    xs = [float(pos[0]) for pos in raw_positions.values()]
    ys = [float(pos[1]) for pos in raw_positions.values()]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    x0, y0, x1, y1 = [int(value) for value in content_bbox]
    inset = max(4, int(node_radius_px) + 4)
    target_x0 = float(x0 + inset)
    target_y0 = float(y0 + inset)
    target_x1 = float(x1 - inset)
    target_y1 = float(y1 - inset)
    source_w = float(max_x - min_x)
    source_h = float(max_y - min_y)

    scaled: Dict[int, Point] = {}
    for node, position in raw_positions.items():
        if float(source_w) <= 1e-9:
            px = 0.5 * float(target_x0 + target_x1)
        else:
            px = float(target_x0) + (
                ((float(position[0]) - float(min_x)) / float(source_w)) * float(max(1.0, target_x1 - target_x0))
            )
        if float(source_h) <= 1e-9:
            py = 0.5 * float(target_y0 + target_y1)
        else:
            py = float(target_y0) + (
                ((float(position[1]) - float(min_y)) / float(source_h)) * float(max(1.0, target_y1 - target_y0))
            )
        scaled[int(node)] = (int(round(px)), int(round(py)))
    return scaled


def _transform_raw_layout(
    raw_positions: Mapping[int, Sequence[float]],
    *,
    layout_transform_variant: str,
) -> Dict[int, Tuple[float, float]]:
    """Apply one global D4-style transform to raw layout coordinates."""

    transform = str(layout_transform_variant)
    transformed: Dict[int, Tuple[float, float]] = {}
    for node, position in raw_positions.items():
        x = float(position[0])
        y = float(position[1])
        if transform == "rotate_90":
            point = (-y, x)
        elif transform == "rotate_180":
            point = (-x, -y)
        elif transform == "rotate_270":
            point = (y, -x)
        elif transform == "mirror_left_right":
            point = (-x, y)
        elif transform == "mirror_up_down":
            point = (x, -y)
        else:
            point = (x, y)
            transform = "identity"
        transformed[int(node)] = (float(point[0]), float(point[1]))
    return transformed


def _spring_layout(graph: nx.Graph, *, seed: int) -> Dict[int, Point]:
    """Return one spring layout in abstract coordinates."""

    node_count = max(1, int(graph.number_of_nodes()))
    layout = nx.spring_layout(
        graph,
        seed=int(seed) % (2**32 - 1),
        k=1.8 / math.sqrt(float(node_count)),
        iterations=150,
        scale=1.0,
    )
    return {int(node): (float(position[0]), float(position[1])) for node, position in layout.items()}


def _circular_layout(graph: nx.Graph) -> Dict[int, Point]:
    """Return one circular layout in abstract coordinates."""

    layout = nx.circular_layout(graph, scale=1.0)
    return {int(node): (float(position[0]), float(position[1])) for node, position in layout.items()}


def _shell_layout(graph: nx.Graph) -> Dict[int, Point]:
    """Return one shell layout in abstract coordinates."""

    sorted_nodes = sorted((int(node) for node in graph.nodes()), key=lambda node: (-int(graph.degree(node)), int(node)))
    if len(sorted_nodes) <= 2:
        return _circular_layout(graph)
    inner_shell_size = max(1, min(len(sorted_nodes) - 1, int(round(len(sorted_nodes) * 0.35)))) if len(sorted_nodes) > 2 else 1
    inner_nodes = tuple(int(node) for node in sorted_nodes[:inner_shell_size])
    outer_nodes = tuple(int(node) for node in sorted_nodes[inner_shell_size:])
    positions: Dict[int, Tuple[float, float]] = {}
    if len(inner_nodes) == 1:
        positions[int(inner_nodes[0])] = (0.0, 0.0)
    else:
        inner_phase = math.pi / float(max(4, len(inner_nodes) * 2))
        for index, node in enumerate(inner_nodes):
            angle = float(inner_phase) + ((2.0 * math.pi * float(index)) / float(len(inner_nodes)))
            positions[int(node)] = (0.46 * math.cos(angle), 0.46 * math.sin(angle))
    outer_phase = math.pi / float(max(3, len(outer_nodes)))
    for index, node in enumerate(outer_nodes):
        angle = float(outer_phase) + ((2.0 * math.pi * float(index)) / float(max(1, len(outer_nodes))))
        positions[int(node)] = (math.cos(angle), math.sin(angle))
    return positions


def _sorted_graph_nodes(graph: nx.Graph) -> Tuple[int, ...]:
    """Return graph nodes in a stable numeric order."""

    return tuple(sorted((int(node) for node in graph.nodes())))


def _component_node_sets(graph: nx.Graph) -> Tuple[Tuple[int, ...], ...]:
    """Return weak/undirected components in stable largest-first order."""

    if graph.is_directed():
        components = nx.weakly_connected_components(graph)  # type: ignore[arg-type]
    else:
        components = nx.connected_components(graph)
    ordered = [tuple(sorted(int(node) for node in component)) for component in components]
    return tuple(sorted(ordered, key=lambda nodes: (-len(nodes), nodes[0] if nodes else 0)))


def _grid_jitter_layout(graph: nx.Graph, *, seed: int) -> Dict[int, Point]:
    """Return a jittered grid layout in abstract coordinates."""

    nodes = _sorted_graph_nodes(graph)
    node_count = len(nodes)
    if node_count <= 0:
        return {}
    rng = random.Random(int(seed) + 17)
    columns = max(1, int(math.ceil(math.sqrt(float(node_count)))))
    rows = max(1, int(math.ceil(float(node_count) / float(columns))))
    positions: Dict[int, Tuple[float, float]] = {}
    for index, node in enumerate(nodes):
        row = int(index // columns)
        column = int(index % columns)
        jitter_x = (rng.random() - 0.5) * 0.28
        jitter_y = (rng.random() - 0.5) * 0.28
        positions[int(node)] = (
            float(column - ((columns - 1) / 2.0) + jitter_x),
            float(row - ((rows - 1) / 2.0) + jitter_y),
        )
    return positions


def _component_subgraph(graph: nx.Graph, nodes: Sequence[int]) -> nx.Graph:
    """Return a copied subgraph for one component."""

    return graph.subgraph(tuple(int(node) for node in nodes)).copy()


def _component_clustered_layout(graph: nx.Graph, *, seed: int) -> Dict[int, Point]:
    """Return a layout that separates connected components into visual clusters."""

    components = _component_node_sets(graph)
    if not components:
        return {}
    if len(components) == 1:
        nodes = tuple(int(node) for node in components[0])
        if len(nodes) <= 2:
            return _circular_layout(graph)
        undirected = graph.to_undirected() if graph.is_directed() else graph
        seed_nodes = [sorted(nodes, key=lambda node: (-int(undirected.degree(int(node))), int(node)))[0]]
        while len(seed_nodes) < min(3, len(nodes)):
            distances_by_candidate = []
            for node in nodes:
                if int(node) in set(seed_nodes):
                    continue
                shortest = min(
                    int(nx.shortest_path_length(undirected, int(node), int(seed_node)))
                    for seed_node in seed_nodes
                )
                distances_by_candidate.append((int(shortest), -int(node), int(node)))
            if not distances_by_candidate:
                break
            seed_nodes.append(max(distances_by_candidate)[2])
        clusters: Dict[int, List[int]] = {int(index): [] for index in range(len(seed_nodes))}
        for node in nodes:
            best_index = min(
                range(len(seed_nodes)),
                key=lambda index: (
                    int(nx.shortest_path_length(undirected, int(node), int(seed_nodes[int(index)]))),
                    int(index),
                ),
            )
            clusters[int(best_index)].append(int(node))
        positions: Dict[int, Tuple[float, float]] = {}
        center_radius = 1.22
        for cluster_index, cluster_nodes in sorted(clusters.items()):
            angle = (2.0 * math.pi * float(cluster_index)) / float(max(1, len(clusters)))
            center = (float(center_radius * math.cos(angle)), float(center_radius * math.sin(angle)))
            ordered = sorted(cluster_nodes)
            if len(ordered) == 1:
                positions[int(ordered[0])] = tuple(float(value) for value in center)
                continue
            local_radius = 0.46 + (0.04 * float(len(ordered)))
            for index, node in enumerate(ordered):
                local_angle = angle + math.pi + ((2.0 * math.pi * float(index)) / float(len(ordered)))
                positions[int(node)] = (
                    float(center[0] + (local_radius * math.cos(local_angle))),
                    float(center[1] + (local_radius * math.sin(local_angle))),
                )
        return positions

    center_radius = max(1.5, 0.72 * float(len(components)))
    positions: Dict[int, Tuple[float, float]] = {}
    for component_index, component_nodes in enumerate(components):
        angle = (2.0 * math.pi * float(component_index)) / float(len(components))
        center = (float(center_radius * math.cos(angle)), float(center_radius * math.sin(angle)))
        subgraph = _component_subgraph(graph, component_nodes)
        if len(component_nodes) == 1:
            local = {int(component_nodes[0]): (0.0, 0.0)}
        elif len(component_nodes) == 2:
            local = {
                int(component_nodes[0]): (-0.35, 0.0),
                int(component_nodes[1]): (0.35, 0.0),
            }
        else:
            local = _spring_layout(subgraph, seed=int(seed) + 31 + int(component_index))
        local_scale = min(0.72, max(0.34, 0.18 * math.sqrt(float(len(component_nodes)))))
        for node, point in local.items():
            positions[int(node)] = (
                float(center[0] + (float(point[0]) * float(local_scale))),
                float(center[1] + (float(point[1]) * float(local_scale))),
            )
    return positions


def _layer_ranks_for_component(graph: nx.Graph, component_nodes: Sequence[int]) -> Dict[int, int]:
    """Return stable layer ranks for one component."""

    nodes = tuple(int(node) for node in component_nodes)
    if not nodes:
        return {}
    subgraph = graph.subgraph(nodes).copy()
    if graph.is_directed() and nx.is_directed_acyclic_graph(subgraph):
        ranks = {int(node): 0 for node in nodes}
        for node in nx.topological_sort(subgraph):
            predecessors = [int(pred) for pred in subgraph.predecessors(int(node))]
            if predecessors:
                ranks[int(node)] = max(int(ranks[int(pred)]) + 1 for pred in predecessors)
        return ranks

    undirected = subgraph.to_undirected() if subgraph.is_directed() else subgraph
    root = sorted(nodes, key=lambda node: (-int(undirected.degree(int(node))), int(node)))[0]
    distances = nx.single_source_shortest_path_length(undirected, int(root))
    return {int(node): int(distances.get(int(node), 0)) for node in nodes}


def _layered_layout(graph: nx.Graph) -> Dict[int, Point]:
    """Return a rank/layer layout in abstract coordinates."""

    components = _component_node_sets(graph)
    positions: Dict[int, Tuple[float, float]] = {}
    x_offset = 0.0
    for component_nodes in components:
        ranks = _layer_ranks_for_component(graph, component_nodes)
        layers: Dict[int, List[int]] = {}
        for node, rank in ranks.items():
            layers.setdefault(int(rank), []).append(int(node))
        if not layers:
            continue
        max_rank = max(layers)
        for rank in sorted(layers):
            layer_nodes = sorted(layers[int(rank)], key=lambda node: (-int(graph.degree(int(node))), int(node)))
            for index, node in enumerate(layer_nodes):
                x_jitter = 0.22 * (float(index) - ((float(len(layer_nodes)) - 1.0) / 2.0))
                y = float(index - ((len(layer_nodes) - 1) / 2.0))
                if len(layer_nodes) == 1 and max_rank > 1:
                    y += 0.34 * math.sin(float(rank) * 1.1)
                positions[int(node)] = (float(x_offset + float(rank) + x_jitter), float(y))
        x_offset += float(max_rank) + 2.0
    return positions


def _label_to_node_map(graph_sample: GraphTopologySample) -> Dict[str, int]:
    """Return a label-to-node map for one topology sample."""

    return {str(label): int(node) for node, label in zip(graph_sample.graph.nodes(), graph_sample.node_labels)}


def _path_spine_nodes(graph_sample: GraphTopologySample) -> Tuple[int, ...]:
    """Return a meaningful path to use as a visual spine."""

    label_to_node = _label_to_node_map(graph_sample)
    target_labels = tuple(str(label) for label in getattr(graph_sample, "target_labels", ()) or ())
    if len(target_labels) >= 2 and all(str(label) in label_to_node for label in target_labels):
        return tuple(int(label_to_node[str(label)]) for label in target_labels)

    graph = graph_sample.graph
    undirected = graph.to_undirected() if graph.is_directed() else graph
    best_path: Tuple[int, ...] = ()
    for component_nodes in _component_node_sets(undirected):
        if len(component_nodes) == 1:
            candidate = tuple(int(node) for node in component_nodes)
        else:
            start = int(component_nodes[0])
            first_distances = nx.single_source_shortest_path_length(undirected, start)
            farthest = max(component_nodes, key=lambda node: (int(first_distances.get(int(node), -1)), -int(node)))
            second_distances = nx.single_source_shortest_path_length(undirected, int(farthest))
            other = max(component_nodes, key=lambda node: (int(second_distances.get(int(node), -1)), -int(node)))
            candidate = tuple(int(node) for node in nx.shortest_path(undirected, int(farthest), int(other)))
        if len(candidate) > len(best_path):
            best_path = tuple(int(node) for node in candidate)
    return best_path or _sorted_graph_nodes(graph)


def _path_spine_layout(graph_sample: GraphTopologySample) -> Dict[int, Point]:
    """Return a layout with one important path as a horizontal spine."""

    graph = graph_sample.graph
    spine = _path_spine_nodes(graph_sample)
    if not spine:
        return _circular_layout(graph)
    spine_index = {int(node): int(index) for index, node in enumerate(spine)}
    positions: Dict[int, Tuple[float, float]] = {
        int(node): (
            float(index),
            0.0 if len(spine) <= 2 else 0.20 * math.sin(float(index) * 1.1),
        )
        for index, node in enumerate(spine)
    }
    undirected = graph.to_undirected() if graph.is_directed() else graph
    buckets: Dict[int, List[Tuple[int, int]]] = {int(index): [] for index in range(len(spine))}
    fallback_bucket = len(spine) - 1
    for node in _sorted_graph_nodes(graph):
        if int(node) in spine_index:
            continue
        best_anchor = fallback_bucket
        best_distance = 10**9
        for anchor_node, anchor_index in spine_index.items():
            try:
                distance = int(nx.shortest_path_length(undirected, int(node), int(anchor_node)))
            except nx.NetworkXNoPath:
                continue
            if (distance, anchor_index) < (best_distance, best_anchor):
                best_distance = int(distance)
                best_anchor = int(anchor_index)
        buckets.setdefault(int(best_anchor), []).append((int(best_distance if best_distance < 10**9 else 1), int(node)))

    for anchor_index, entries in sorted(buckets.items()):
        ordered_entries = sorted(entries, key=lambda item: (item[0], item[1]))
        if not ordered_entries:
            continue
        # Keep each anchor's off-spine branch on one side of the spine. Alternating
        # individual nodes can make non-spine edges cut back through the anchor and
        # visually merge with incident spine edges.
        side = 1.0 if int(anchor_index) % 2 == 0 else -1.0
        center_slot = (float(len(ordered_entries)) - 1.0) / 2.0
        fan_half_angle = min(1.22, 0.68 + (0.22 * float(max(0, len(ordered_entries) - 1))))
        for entry_index, (distance, node) in enumerate(ordered_entries):
            slot = float(entry_index) - float(center_slot)
            distance_scale = float(max(1, int(distance)))
            slot_norm = 0.0 if center_slot <= 0.0 else float(slot) / float(center_slot)
            base_angle = (math.pi / 2.0) if side > 0.0 else (-math.pi / 2.0)
            angle = float(base_angle) + (float(slot_norm) * float(fan_half_angle))
            radius = 1.48 + (0.78 * distance_scale) + (0.12 * abs(float(slot)))
            horizontal_shift = float(radius) * math.cos(float(angle))
            vertical_shift = float(radius) * math.sin(float(angle))
            positions[int(node)] = (float(anchor_index) + float(horizontal_shift), float(vertical_shift))
    return positions


def _radial_tree_layout(graph_sample: GraphTopologySample, *, seed: int) -> Dict[int, Point]:
    """Return a rooted radial layout for each component."""

    graph = graph_sample.graph
    undirected = graph.to_undirected() if graph.is_directed() else graph
    label_to_node = _label_to_node_map(graph_sample)
    source_label = str(getattr(graph_sample, "source_label", "") or "")
    preferred_root = label_to_node.get(source_label)
    components = _component_node_sets(undirected)
    if not components:
        return {}
    component_center_radius = 0.0 if len(components) == 1 else max(1.7, 0.70 * float(len(components)))
    positions: Dict[int, Tuple[float, float]] = {}
    for component_index, component_nodes in enumerate(components):
        component_set = set(int(node) for node in component_nodes)
        if preferred_root in component_set:
            root = int(preferred_root)
        else:
            root = sorted(component_nodes, key=lambda node: (-int(undirected.degree(int(node))), int(node)))[0]
        if len(components) == 1:
            center = (0.0, 0.0)
        else:
            angle = (2.0 * math.pi * float(component_index)) / float(len(components))
            center = (
                float(component_center_radius * math.cos(angle)),
                float(component_center_radius * math.sin(angle)),
            )
        distances = nx.single_source_shortest_path_length(undirected.subgraph(component_nodes), int(root))
        layers: Dict[int, List[int]] = {}
        for node in component_nodes:
            layers.setdefault(int(distances.get(int(node), 0)), []).append(int(node))
        phase = (random.Random(int(seed) + 43 + int(component_index)).random() * 2.0 * math.pi)
        layer_scale = 0.62 if len(components) > 1 else 1.0
        for distance, layer_nodes in sorted(layers.items()):
            ordered = sorted(layer_nodes, key=lambda node: (-int(undirected.degree(int(node))), int(node)))
            if int(distance) == 0:
                positions[int(root)] = tuple(float(value) for value in center)
                continue
            for index, node in enumerate(ordered):
                angle = (
                    float(phase)
                    + ((2.0 * math.pi * float(index)) / float(max(1, len(ordered))))
                    + (0.19 * float(distance))
                )
                radius = (1.16 * float(distance) * float(layer_scale)) + (0.08 * float(index))
                positions[int(node)] = (
                    float(center[0] + (radius * math.cos(angle))),
                    float(center[1] + (radius * math.sin(angle))),
                )
    return positions


def _min_node_distance(positions: Mapping[int, Point]) -> float:
    """Return the minimum pairwise node-center distance."""

    points = list(positions.values())
    if len(points) < 2:
        return float("inf")
    min_distance = float("inf")
    for index, left in enumerate(points):
        for right in points[index + 1 :]:
            distance = math.hypot(float(right[0]) - float(left[0]), float(right[1]) - float(left[1]))
            min_distance = min(min_distance, float(distance))
    return float(min_distance)


def _min_incident_edge_angle_degrees(graph: nx.Graph, positions: Mapping[int, Point]) -> float:
    """Return the minimum angle between incident straight-edge rays."""

    incident_graph = graph.to_undirected() if graph.is_directed() else graph
    min_angle = float("inf")
    for raw_node in incident_graph.nodes():
        node = int(raw_node)
        center = positions.get(int(node))
        if center is None:
            continue
        vectors: List[Tuple[float, float]] = []
        for raw_neighbor in sorted(incident_graph.neighbors(raw_node), key=lambda value: int(value)):
            neighbor = int(raw_neighbor)
            neighbor_point = positions.get(int(neighbor))
            if neighbor_point is None:
                continue
            dx = float(neighbor_point[0]) - float(center[0])
            dy = float(neighbor_point[1]) - float(center[1])
            norm = float(math.hypot(dx, dy))
            if norm <= 1e-6:
                continue
            vectors.append((float(dx / norm), float(dy / norm)))
        for index, left in enumerate(vectors):
            for right in vectors[index + 1 :]:
                dot = max(-1.0, min(1.0, float((left[0] * right[0]) + (left[1] * right[1]))))
                angle = math.degrees(math.acos(dot))
                min_angle = min(float(min_angle), float(angle))
    return float(min_angle)


def _inflated_node_bbox(*, center: Point, node_radius_px: int, clearance_px: int) -> BBox:
    """Return a node bbox inflated by a visual edge-clearance margin."""

    radius = int(node_radius_px) + int(clearance_px)
    return (
        int(center[0]) - int(radius),
        int(center[1]) - int(radius),
        int(center[0]) + int(radius),
        int(center[1]) + int(radius),
    )


def _unrelated_edge_node_conflict_count(
    graph_sample: GraphTopologySample,
    *,
    positions: Mapping[int, Point],
    node_radius_px: int,
) -> int:
    """Count edges whose straight centerline passes too close to unrelated nodes."""

    label_to_node = _label_to_node_map(graph_sample)
    clearance_px = max(6, int(round(float(node_radius_px) * 0.35)))
    conflicts = 0
    for left_label, right_label in graph_sample.edge_labels:
        left_node = label_to_node.get(str(left_label))
        right_node = label_to_node.get(str(right_label))
        if left_node is None or right_node is None:
            continue
        start = positions.get(int(left_node))
        end = positions.get(int(right_node))
        if start is None or end is None:
            continue
        edge_nodes = {int(left_node), int(right_node)}
        segment = (tuple(int(value) for value in start), tuple(int(value) for value in end))
        for node, center in positions.items():
            if int(node) in edge_nodes:
                continue
            if _segment_intersects_bbox(
                segment,
                _inflated_node_bbox(
                    center=tuple(int(value) for value in center),
                    node_radius_px=int(node_radius_px),
                    clearance_px=int(clearance_px),
                ),
            ):
                conflicts += 1
    return int(conflicts)


def _raw_layout_for_variant(
    graph_sample: GraphTopologySample,
    *,
    layout_variant: str,
    layout_seed: int,
) -> Tuple[Dict[int, Point], str]:
    """Resolve abstract coordinates for one requested layout variant."""

    graph = graph_sample.graph
    requested = str(layout_variant)
    if requested == "shell":
        return _shell_layout(graph), "shell"
    if requested == "spring":
        return _spring_layout(graph, seed=int(layout_seed)), "spring"
    if requested == "grid_jitter":
        return _grid_jitter_layout(graph, seed=int(layout_seed)), "grid_jitter"
    if requested == "layered":
        return _layered_layout(graph), "layered"
    if requested == "component_clustered":
        return _component_clustered_layout(graph, seed=int(layout_seed)), "component_clustered"
    if requested == "path_spine":
        return _path_spine_layout(graph_sample), "path_spine"
    if requested == "radial_tree":
        return _radial_tree_layout(graph_sample, seed=int(layout_seed)), "radial_tree"
    return _circular_layout(graph), "circular"


def _resolve_positions(
    graph_sample: GraphTopologySample,
    *,
    layout_variant: str,
    layout_transform_variant: str,
    content_bbox: BBox,
    node_radius_px: int,
    layout_seed: int,
    layout_fallback_variants: Sequence[str] | None = None,
) -> Tuple[Dict[int, Point], str, str]:
    """Resolve pixel node centers for one graph layout variant."""

    graph = graph_sample.graph
    actual_transform = str(layout_transform_variant)
    min_node_distance_px = float(max(18, int(node_radius_px) * 2 + 4))
    min_incident_angle_degrees = float(max(5.0, min(8.0, float(node_radius_px) * 0.25)))
    component_count = len(_component_node_sets(graph))
    if int(component_count) > 1 and str(layout_variant) not in {"component_clustered", "layered", "radial_tree"}:
        layout_variant = "component_clustered"
    fallback_variants = tuple(str(value) for value in layout_fallback_variants) if layout_fallback_variants is not None else (
        ("layered", "radial_tree", "grid_jitter", "spring", "shell", "circular")
        if int(component_count) > 1
        else ("spring", "shell", "circular")
    )
    seed_offsets = {"spring": 97, "grid_jitter": 131, "component_clustered": 179, "radial_tree": 223, "shell": 0, "circular": 0}
    stochastic_layouts = {"spring", "grid_jitter", "component_clustered", "radial_tree"}
    layout_candidates: List[Tuple[str, int]] = []
    for seed_delta in ((0, 101, 211) if str(layout_variant) in stochastic_layouts else (0,)):
        layout_candidates.append((str(layout_variant), int(layout_seed) + int(seed_delta)))
    for fallback_variant in fallback_variants:
        if str(fallback_variant) == str(layout_variant):
            continue
        for seed_delta in ((0, 101, 211) if str(fallback_variant) in stochastic_layouts else (0,)):
            layout_candidates.append((str(fallback_variant), int(layout_seed) + int(seed_offsets.get(str(fallback_variant), 0)) + int(seed_delta)))

    best_positions: Dict[int, Point] | None = None
    best_layout = "circular"
    best_score: Tuple[float, float, float, float] | None = None
    for candidate_layout, candidate_seed in layout_candidates:
        raw_positions, actual_layout = _raw_layout_for_variant(
            graph_sample,
            layout_variant=str(candidate_layout),
            layout_seed=int(candidate_seed),
        )
        transformed = _transform_raw_layout(raw_positions, layout_transform_variant=actual_transform)
        positions = _scale_layout_to_content(
            transformed,
            content_bbox=tuple(int(value) for value in content_bbox),
            node_radius_px=int(node_radius_px),
        )
        node_distance = float(_min_node_distance(positions))
        incident_angle = float(_min_incident_edge_angle_degrees(graph, positions))
        edge_node_conflicts = int(
            _unrelated_edge_node_conflict_count(
                graph_sample,
                positions=positions,
                node_radius_px=int(node_radius_px),
            )
        )
        angle_score = 180.0 if math.isinf(float(incident_angle)) else float(incident_angle)
        score = (
            1.0 if edge_node_conflicts == 0 else 0.0,
            1.0 if node_distance >= min_node_distance_px else 0.0,
            1.0 if angle_score >= min_incident_angle_degrees else 0.0,
            float(-edge_node_conflicts),
            float(angle_score),
            float(node_distance),
        )
        if best_score is None or score > best_score:
            best_score = tuple(float(value) for value in score)
            best_positions = dict(positions)
            best_layout = str(actual_layout)
        if edge_node_conflicts == 0 and node_distance >= min_node_distance_px and angle_score >= min_incident_angle_degrees:
            return positions, str(actual_layout), actual_transform
    if best_positions is None:
        fallback_positions = _scale_layout_to_content(
            _transform_raw_layout(_circular_layout(graph), layout_transform_variant=actual_transform),
            content_bbox=tuple(int(value) for value in content_bbox),
            node_radius_px=int(node_radius_px),
        )
        return fallback_positions, "circular", actual_transform
    return dict(best_positions), str(best_layout), actual_transform


__all__ = [
    "_resolve_positions",
]
