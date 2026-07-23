"""Metro-route topology and path algorithms."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from typing import Dict, Mapping, Sequence, Tuple

from ...shared.graph_sample_types import graph_label_sort_key
from .state import (
    METRO_ROUTE_TEMPLATES,
    GridPoint,
    LabelEdge,
    MetroRouteTemplate,
)


def station_sort_key(coord: GridPoint) -> Tuple[int, int]:
    return (int(coord[1]), int(coord[0]))


def canonical_label_edge(left: str, right: str) -> LabelEdge:
    ordered = sorted((str(left), str(right)), key=graph_label_sort_key)
    return (ordered[0], ordered[1])


def route_combo_coords(route_templates: Sequence[MetroRouteTemplate]) -> Tuple[Tuple[GridPoint, ...], Dict[GridPoint, Tuple[str, ...]], Dict[str, Tuple[GridPoint, ...]], Dict[GridPoint, Tuple[GridPoint, ...]]]:
    """Return station coords, route memberships, route coords, and station adjacency."""

    station_coords = tuple(sorted({tuple(int(value) for value in coord) for route in route_templates for coord in route.grid_points}, key=station_sort_key))
    coords_by_route = {str(route.route_id): tuple(tuple(int(value) for value in coord) for coord in route.grid_points) for route in route_templates}
    route_ids_by_coord: dict[GridPoint, list[str]] = defaultdict(list)
    adjacency: dict[GridPoint, set[GridPoint]] = {tuple(coord): set() for coord in station_coords}
    for route in route_templates:
        for coord in route.grid_points:
            route_ids_by_coord[tuple(int(value) for value in coord)].append(str(route.route_id))
        for left, right in zip(route.grid_points, route.grid_points[1:]):
            left_coord = tuple(int(value) for value in left)
            right_coord = tuple(int(value) for value in right)
            adjacency[left_coord].add(right_coord)
            adjacency[right_coord].add(left_coord)
    station_routes = {tuple(coord): tuple(sorted(route_ids)) for coord, route_ids in route_ids_by_coord.items()}
    adjacency_tuple = {tuple(coord): tuple(sorted(neighbors, key=station_sort_key)) for coord, neighbors in adjacency.items()}
    return tuple(station_coords), dict(station_routes), dict(coords_by_route), dict(adjacency_tuple)


def route_combinations_for_count(*, route_count_min: int, route_count_max: int) -> Tuple[Tuple[MetroRouteTemplate, ...], ...]:
    """Return fixed route-template combinations within a route-count range."""

    combos: list[Tuple[MetroRouteTemplate, ...]] = []
    for route_count in range(max(1, int(route_count_min)), min(len(METRO_ROUTE_TEMPLATES), int(route_count_max)) + 1):
        combos.extend(tuple(combo) for combo in combinations(METRO_ROUTE_TEMPLATES, int(route_count)))
    return tuple(combos)


def transfer_station_count(routes: Sequence[MetroRouteTemplate]) -> int:
    coord_counts = Counter(tuple(int(value) for value in coord) for route in routes for coord in route.grid_points)
    return sum(1 for count in coord_counts.values() if int(count) >= 2)


def single_route_coords(routes: Sequence[MetroRouteTemplate]) -> Tuple[GridPoint, ...]:
    station_coords, route_ids_by_coord, _, _ = route_combo_coords(routes)
    return tuple(coord for coord in station_coords if len(route_ids_by_coord[coord]) == 1)


def route_transfer_station_coords(routes: Sequence[MetroRouteTemplate], *, route_id: str) -> Tuple[GridPoint, ...]:
    """Return stations on one route that are served by at least two routes."""

    _, route_ids_by_coord, coords_by_route, _ = route_combo_coords(routes)
    route_coords = tuple(coords_by_route[str(route_id)])
    return tuple(coord for coord in route_coords if len(route_ids_by_coord[coord]) >= 2)


def route_single_route_station_coords(routes: Sequence[MetroRouteTemplate], *, route_id: str) -> Tuple[GridPoint, ...]:
    """Return stations on one route that are served only by that route."""

    _, route_ids_by_coord, coords_by_route, _ = route_combo_coords(routes)
    route_coords = tuple(coords_by_route[str(route_id)])
    return tuple(coord for coord in route_coords if len(route_ids_by_coord[coord]) == 1)


def bfs_dist_count(adjacency: Mapping[GridPoint, Sequence[GridPoint]], *, start: GridPoint) -> Tuple[Dict[GridPoint, int], Dict[GridPoint, int]]:
    """Return shortest distances and shortest-path counts from one station."""

    start_coord = tuple(int(value) for value in start)
    dist: Dict[GridPoint, int] = {start_coord: 0}
    count: Dict[GridPoint, int] = {start_coord: 1}
    queue: list[GridPoint] = [start_coord]
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for neighbor in adjacency.get(tuple(node), ()):
            neighbor_coord = tuple(int(value) for value in neighbor)
            candidate_dist = int(dist[node]) + 1
            if neighbor_coord not in dist:
                dist[neighbor_coord] = int(candidate_dist)
                count[neighbor_coord] = int(count[node])
                queue.append(neighbor_coord)
            elif int(dist[neighbor_coord]) == int(candidate_dist):
                count[neighbor_coord] = int(count[neighbor_coord]) + int(count[node])
    return dict(dist), dict(count)


def unique_shortest_coord_path(adjacency: Mapping[GridPoint, Sequence[GridPoint]], *, source: GridPoint, goal: GridPoint) -> Tuple[GridPoint, ...] | None:
    """Return the unique shortest coordinate path, or ``None`` if ambiguous/unreachable."""

    source_coord = tuple(int(value) for value in source)
    goal_coord = tuple(int(value) for value in goal)
    dist, count = bfs_dist_count(adjacency, start=source_coord)
    if goal_coord not in dist or int(count.get(goal_coord, 0)) != 1:
        return None
    path = [goal_coord]
    current = goal_coord
    while current != source_coord:
        previous = [tuple(neighbor) for neighbor in adjacency.get(tuple(current), ()) if tuple(neighbor) in dist and int(dist[tuple(neighbor)]) == int(dist[current]) - 1]
        if len(previous) != 1:
            return None
        current = tuple(previous[0])
        path.append(current)
    path.reverse()
    return tuple(path)


def exact_distance_coords(routes: Sequence[MetroRouteTemplate], *, query_coord: GridPoint, query_distance: int) -> Tuple[GridPoint, ...]:
    _, _, _, adjacency = route_combo_coords(routes)
    dist, _ = bfs_dist_count(adjacency, start=tuple(query_coord))
    return tuple(sorted((coord for coord, distance in dist.items() if int(distance) == int(query_distance)), key=station_sort_key))


def matching_transfer_station_route_combos(*, target_count: int, route_count_min: int, route_count_max: int) -> Tuple[Tuple[MetroRouteTemplate, ...], ...]:
    return tuple(routes for routes in route_combinations_for_count(route_count_min=route_count_min, route_count_max=route_count_max) if transfer_station_count(routes) == int(target_count))


def matching_single_route_station_route_combos(*, target_count: int, route_count_min: int, route_count_max: int) -> Tuple[Tuple[MetroRouteTemplate, ...], ...]:
    return tuple(routes for routes in route_combinations_for_count(route_count_min=route_count_min, route_count_max=route_count_max) if len(single_route_coords(routes)) == int(target_count))


def matching_route_transfer_station_route_combos(*, target_count: int, route_count_min: int, route_count_max: int) -> Tuple[Tuple[MetroRouteTemplate, ...], ...]:
    combos = []
    for routes in route_combinations_for_count(route_count_min=route_count_min, route_count_max=route_count_max):
        if any(len(route_transfer_station_coords(routes, route_id=str(route.route_id))) == int(target_count) for route in routes):
            combos.append(tuple(routes))
    return tuple(combos)


def matching_route_single_route_station_route_combos(*, target_count: int, route_count_min: int, route_count_max: int) -> Tuple[Tuple[MetroRouteTemplate, ...], ...]:
    combos = []
    for routes in route_combinations_for_count(route_count_min=route_count_min, route_count_max=route_count_max):
        if any(len(route_single_route_station_coords(routes, route_id=str(route.route_id))) == int(target_count) for route in routes):
            combos.append(tuple(routes))
    return tuple(combos)


def matching_exact_distance_route_combos(*, target_count: int, route_count_min: int, route_count_max: int, query_distance: int) -> Tuple[Tuple[MetroRouteTemplate, ...], ...]:
    combos = []
    for routes in route_combinations_for_count(route_count_min=route_count_min, route_count_max=route_count_max):
        station_coords, _, _, _ = route_combo_coords(routes)
        if any(len(exact_distance_coords(routes, query_coord=coord, query_distance=int(query_distance))) == int(target_count) for coord in station_coords):
            combos.append(tuple(routes))
    return tuple(combos)


def matching_shortest_path_route_combos(*, target_length: int, route_count_min: int, route_count_max: int) -> Tuple[Tuple[MetroRouteTemplate, ...], ...]:
    combos = []
    for routes in route_combinations_for_count(route_count_min=route_count_min, route_count_max=route_count_max):
        station_coords, _, _, adjacency = route_combo_coords(routes)
        if any(
            (path := unique_shortest_coord_path(adjacency, source=source, goal=goal)) is not None and len(path) - 1 == int(target_length)
            for left_index, source in enumerate(station_coords)
            for goal in station_coords[left_index + 1 :]
        ):
            combos.append(tuple(routes))
    return tuple(combos)


def feasible_transfer_station_counts(*, route_count_min: int, route_count_max: int) -> Tuple[int, ...]:
    return tuple(sorted({transfer_station_count(routes) for routes in route_combinations_for_count(route_count_min=route_count_min, route_count_max=route_count_max)}))


def feasible_single_route_station_counts(*, route_count_min: int, route_count_max: int) -> Tuple[int, ...]:
    return tuple(sorted({len(single_route_coords(routes)) for routes in route_combinations_for_count(route_count_min=route_count_min, route_count_max=route_count_max)}))


def feasible_route_transfer_station_counts(*, route_count_min: int, route_count_max: int) -> Tuple[int, ...]:
    values: set[int] = set()
    for routes in route_combinations_for_count(route_count_min=route_count_min, route_count_max=route_count_max):
        for route in routes:
            values.add(int(len(route_transfer_station_coords(routes, route_id=str(route.route_id)))))
    return tuple(sorted(values))


def feasible_route_single_route_station_counts(*, route_count_min: int, route_count_max: int) -> Tuple[int, ...]:
    values: set[int] = set()
    for routes in route_combinations_for_count(route_count_min=route_count_min, route_count_max=route_count_max):
        for route in routes:
            values.add(int(len(route_single_route_station_coords(routes, route_id=str(route.route_id)))))
    return tuple(sorted(values))


def feasible_exact_distance_counts(*, route_count_min: int, route_count_max: int, query_distance: int) -> Tuple[int, ...]:
    values: set[int] = set()
    for routes in route_combinations_for_count(route_count_min=route_count_min, route_count_max=route_count_max):
        station_coords, _, _, _ = route_combo_coords(routes)
        for coord in station_coords:
            values.add(int(len(exact_distance_coords(routes, query_coord=coord, query_distance=int(query_distance)))))
    return tuple(sorted(values))


def feasible_shortest_path_lengths(*, route_count_min: int, route_count_max: int) -> Tuple[int, ...]:
    values: set[int] = set()
    for routes in route_combinations_for_count(route_count_min=route_count_min, route_count_max=route_count_max):
        station_coords, _, _, adjacency = route_combo_coords(routes)
        for left_index, source in enumerate(station_coords):
            for goal in station_coords[left_index + 1 :]:
                path = unique_shortest_coord_path(adjacency, source=source, goal=goal)
                if path is not None:
                    values.add(int(len(path) - 1))
    return tuple(sorted(values))


__all__ = [
    "exact_distance_coords", "feasible_exact_distance_counts", "feasible_shortest_path_lengths",
    "feasible_route_single_route_station_counts", "feasible_route_transfer_station_counts",
    "feasible_single_route_station_counts", "feasible_transfer_station_counts",
    "matching_exact_distance_route_combos", "matching_route_single_route_station_route_combos",
    "matching_route_transfer_station_route_combos", "matching_shortest_path_route_combos",
    "matching_single_route_station_route_combos", "matching_transfer_station_route_combos",
    "route_combo_coords", "route_single_route_station_coords", "route_transfer_station_coords",
    "single_route_coords", "station_sort_key",
    "unique_shortest_coord_path",
]
