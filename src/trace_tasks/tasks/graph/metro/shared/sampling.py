"""Sampling helpers for the metro graph scene."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Sequence, Tuple

from ...shared.graph_sample_types import graph_label_sort_key
from ...shared.label_assets import resolve_graph_node_labels
from .algorithms import (
    canonical_label_edge,
    exact_distance_coords,
    matching_exact_distance_route_combos,
    matching_route_single_route_station_route_combos,
    matching_route_transfer_station_route_combos,
    matching_shortest_path_route_combos,
    matching_single_route_station_route_combos,
    matching_transfer_station_route_combos,
    route_combo_coords,
    route_single_route_station_coords,
    route_transfer_station_coords,
    single_route_coords,
    unique_shortest_coord_path,
)
from .state import GridPoint, LabelEdge, MetroRouteNetworkSample, MetroRouteTemplate


def build_labeled_metro_sample(rng: random.Random, *, routes: Sequence[MetroRouteTemplate], label_variant: str) -> MetroRouteNetworkSample:
    """Build one labeled metro sample from already-selected route templates."""

    station_coords, _, _, _ = route_combo_coords(tuple(routes))
    resolved_labels = resolve_graph_node_labels(rng, label_variant=str(label_variant), object_count=len(station_coords), max_chars=3, sequential_numbers=False)
    station_labels = tuple(str(label) for label in resolved_labels.labels)
    label_by_coord = {tuple(int(value) for value in coord): str(label) for coord, label in zip(station_coords, station_labels)}
    coord_by_label = {str(label): tuple(coord) for coord, label in label_by_coord.items()}
    route_station_labels: dict[str, Tuple[str, ...]] = {}
    station_route_ids: dict[str, list[str]] = defaultdict(list)
    adjacency_sets: dict[str, set[str]] = {str(label): set() for label in station_labels}
    edge_set: set[LabelEdge] = set()
    for route in routes:
        labels = tuple(str(label_by_coord[tuple(coord)]) for coord in route.grid_points)
        route_station_labels[str(route.route_id)] = labels
        for label in labels:
            station_route_ids[str(label)].append(str(route.route_id))
        for left, right in zip(labels, labels[1:]):
            adjacency_sets[str(left)].add(str(right))
            adjacency_sets[str(right)].add(str(left))
            edge_set.add(canonical_label_edge(str(left), str(right)))
    station_route_ids_by_label = {str(label): tuple(sorted(route_ids)) for label, route_ids in station_route_ids.items()}
    transfer_labels = tuple(sorted((label for label, route_ids in station_route_ids_by_label.items() if len(route_ids) >= 2), key=graph_label_sort_key))
    terminal_coords = {tuple(int(value) for value in route.grid_points[0]) for route in routes} | {tuple(int(value) for value in route.grid_points[-1]) for route in routes}
    terminal_labels = tuple(sorted((str(label_by_coord[coord]) for coord in terminal_coords), key=graph_label_sort_key))
    adjacency_by_label = {str(label): tuple(sorted(neighbors, key=graph_label_sort_key)) for label, neighbors in adjacency_sets.items()}
    edge_labels = tuple(sorted(edge_set, key=lambda pair: (graph_label_sort_key(pair[0]), graph_label_sort_key(pair[1]))))
    return MetroRouteNetworkSample(
        station_labels=tuple(station_labels),
        label_by_coord=dict(label_by_coord),
        coord_by_label=dict(coord_by_label),
        route_templates=tuple(routes),
        route_station_labels=dict(route_station_labels),
        station_route_ids_by_label=dict(station_route_ids_by_label),
        adjacency_by_label=dict(adjacency_by_label),
        edge_labels=tuple(edge_labels),
        transfer_labels=tuple(transfer_labels),
        target_transfer_count=int(len(transfer_labels)),
        route_count=int(len(routes)),
        station_count=int(len(station_coords)),
        label_variant=str(resolved_labels.label_variant),
        terminal_labels=tuple(terminal_labels),
        target_labels=tuple(transfer_labels),
        target_terminal_count=int(len(terminal_labels)),
        label_source_kind=str(resolved_labels.label_source_kind),
        label_bucket=str(resolved_labels.label_bucket),
        label_manifest=str(resolved_labels.label_manifest),
        label_filter=dict(resolved_labels.label_filter),
        label_bucket_probabilities=dict(resolved_labels.label_bucket_probabilities),
    )


def choose_routes(rng: random.Random, combos: Sequence[Tuple[MetroRouteTemplate, ...]]) -> Tuple[MetroRouteTemplate, ...]:
    if not combos:
        raise ValueError("no feasible metro route combo for requested target")
    return tuple(rng.choice(tuple(combos)))


def sample_transfer_station_network(rng: random.Random, *, target_count: int, route_count: int, label_variant: str) -> MetroRouteNetworkSample:
    routes = choose_routes(rng, matching_transfer_station_route_combos(target_count=int(target_count), route_count_min=int(route_count), route_count_max=int(route_count)))
    sample = build_labeled_metro_sample(rng, routes=routes, label_variant=str(label_variant))
    if int(sample.target_transfer_count) != int(target_count):
        raise ValueError("sampled metro transfer count did not match target")
    return sample


def sample_single_route_station_network(rng: random.Random, *, target_count: int, route_count: int, label_variant: str) -> MetroRouteNetworkSample:
    routes = choose_routes(rng, matching_single_route_station_route_combos(target_count=int(target_count), route_count_min=int(route_count), route_count_max=int(route_count)))
    sample = build_labeled_metro_sample(rng, routes=routes, label_variant=str(label_variant))
    labels = tuple(sorted((str(sample.label_by_coord[coord]) for coord in single_route_coords(routes)), key=graph_label_sort_key))
    return MetroRouteNetworkSample(**{**sample.__dict__, "target_labels": labels, "target_single_route_count": int(len(labels))})


def sample_route_transfer_station_network(rng: random.Random, *, target_count: int, route_count: int, label_variant: str) -> MetroRouteNetworkSample:
    routes = choose_routes(rng, matching_route_transfer_station_route_combos(target_count=int(target_count), route_count_min=int(route_count), route_count_max=int(route_count)))
    sample = build_labeled_metro_sample(rng, routes=routes, label_variant=str(label_variant))
    candidates = [
        str(route.route_id)
        for route in routes
        if len(route_transfer_station_coords(routes, route_id=str(route.route_id))) == int(target_count)
    ]
    if not candidates:
        raise ValueError("sampled metro route-transfer target has no matching route")
    route_id = str(rng.choice(candidates))
    target_coords = route_transfer_station_coords(routes, route_id=route_id)
    labels = tuple(sorted((str(sample.label_by_coord[coord]) for coord in target_coords), key=graph_label_sort_key))
    return MetroRouteNetworkSample(
        **{
            **sample.__dict__,
            "query_route_ids": (route_id,),
            "target_labels": labels,
            "target_transfer_count": int(len(labels)),
        }
    )


def sample_route_single_route_station_network(rng: random.Random, *, target_count: int, route_count: int, label_variant: str) -> MetroRouteNetworkSample:
    routes = choose_routes(rng, matching_route_single_route_station_route_combos(target_count=int(target_count), route_count_min=int(route_count), route_count_max=int(route_count)))
    sample = build_labeled_metro_sample(rng, routes=routes, label_variant=str(label_variant))
    candidates = [
        str(route.route_id)
        for route in routes
        if len(route_single_route_station_coords(routes, route_id=str(route.route_id))) == int(target_count)
    ]
    if not candidates:
        raise ValueError("sampled metro route-single-route target has no matching route")
    route_id = str(rng.choice(candidates))
    target_coords = route_single_route_station_coords(routes, route_id=route_id)
    labels = tuple(sorted((str(sample.label_by_coord[coord]) for coord in target_coords), key=graph_label_sort_key))
    return MetroRouteNetworkSample(
        **{
            **sample.__dict__,
            "query_route_ids": (route_id,),
            "target_labels": labels,
            "target_single_route_count": int(len(labels)),
        }
    )


def sample_exact_distance_station_network(rng: random.Random, *, target_count: int, route_count: int, query_distance: int, label_variant: str) -> MetroRouteNetworkSample:
    routes = choose_routes(rng, matching_exact_distance_route_combos(target_count=int(target_count), route_count_min=int(route_count), route_count_max=int(route_count), query_distance=int(query_distance)))
    sample = build_labeled_metro_sample(rng, routes=routes, label_variant=str(label_variant))
    station_coords, _, _, _ = route_combo_coords(routes)
    candidates = [coord for coord in station_coords if len(exact_distance_coords(routes, query_coord=coord, query_distance=int(query_distance))) == int(target_count)]
    if not candidates:
        raise ValueError("sampled metro exact-distance target has no matching station")
    query_coord = tuple(rng.choice(candidates))
    exact_coords = exact_distance_coords(routes, query_coord=query_coord, query_distance=int(query_distance))
    labels = tuple(sorted((str(sample.label_by_coord[coord]) for coord in exact_coords), key=graph_label_sort_key))
    return MetroRouteNetworkSample(**{**sample.__dict__, "query_label": str(sample.label_by_coord[query_coord]), "target_labels": labels, "target_exact_distance_count": int(len(labels))})


def sample_shortest_path_network(rng: random.Random, *, target_length: int, route_count: int, label_variant: str) -> MetroRouteNetworkSample:
    routes = choose_routes(rng, matching_shortest_path_route_combos(target_length=int(target_length), route_count_min=int(route_count), route_count_max=int(route_count)))
    sample = build_labeled_metro_sample(rng, routes=routes, label_variant=str(label_variant))
    station_coords, _, _, adjacency = route_combo_coords(routes)
    candidates: list[Tuple[GridPoint, GridPoint, Tuple[GridPoint, ...]]] = []
    for left_index, source in enumerate(station_coords):
        for goal in station_coords[left_index + 1 :]:
            path = unique_shortest_coord_path(adjacency, source=source, goal=goal)
            if path is not None and len(path) - 1 == int(target_length):
                candidates.append((tuple(source), tuple(goal), tuple(path)))
    if not candidates:
        raise ValueError("sampled metro shortest-path target has no matching pair")
    source_coord, goal_coord, path_coords = rng.choice(candidates)
    path_labels = tuple(str(sample.label_by_coord[coord]) for coord in path_coords)
    return MetroRouteNetworkSample(**{**sample.__dict__, "source_label": str(sample.label_by_coord[source_coord]), "goal_label": str(sample.label_by_coord[goal_coord]), "target_labels": tuple(path_labels), "target_shortest_path_length": int(len(path_labels) - 1)})


__all__ = [
    "sample_exact_distance_station_network", "sample_route_single_route_station_network",
    "sample_route_transfer_station_network", "sample_shortest_path_network",
    "sample_single_route_station_network", "sample_transfer_station_network",
]
