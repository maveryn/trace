"""Lane-crossing motion and validation helpers."""

from __future__ import annotations

from .state import (
    CrossingRouteOption,
    CrossingSample,
    CrossingVehicle,
    route_cell_entity_id,
    route_entity_id,
    start_entity_id,
)


def vehicle_col_at_tick(vehicle: CrossingVehicle, *, tick: int, lane_count: int) -> int | None:
    """Return a vehicle's lane column at a tick, or None if it has left the board."""

    col = int(vehicle.start_col) + (int(vehicle.direction) * int(tick))
    if 0 <= int(col) < int(lane_count):
        return int(col)
    return None


def vehicle_exit_tick(vehicle: CrossingVehicle, *, lane_count: int) -> int:
    """Return the first tick after which a moving object has left the board."""

    if int(vehicle.direction) < 0:
        return int(vehicle.start_col) + 1
    return int(lane_count) - int(vehicle.start_col)


def route_collision_vehicle_ids(
    route: CrossingRouteOption,
    vehicles: tuple[CrossingVehicle, ...],
    *,
    lane_count: int,
) -> tuple[str, ...]:
    """Return vehicles that intersect a route at the row-crossing tick."""

    hits: list[str] = []
    for vehicle in vehicles:
        row = int(vehicle.row)
        if row < 0 or row >= len(route.path_cols):
            continue
        tick = int(row + 1)
        vehicle_col = vehicle_col_at_tick(vehicle, tick=tick, lane_count=int(lane_count))
        if vehicle_col is not None and int(vehicle_col) == int(route.path_cols[row]):
            hits.append(str(vehicle.vehicle_id))
    return tuple(sorted(set(hits)))


def route_first_collision_tick(
    route: CrossingRouteOption,
    vehicles: tuple[CrossingVehicle, ...],
    *,
    lane_count: int,
) -> int | None:
    """Return the first tick at which a route intersects any moving object."""

    for row in range(len(route.path_cols)):
        tick = int(row + 1)
        route_col = int(route.path_cols[row])
        for vehicle in vehicles:
            if int(vehicle.row) != int(row):
                continue
            vehicle_col = vehicle_col_at_tick(vehicle, tick=tick, lane_count=int(lane_count))
            if vehicle_col is not None and int(vehicle_col) == int(route_col):
                return int(tick)
    return None


def validate_crossing_sample(sample: CrossingSample) -> None:
    """Validate scene geometry, symbolic entities, answer counts, and annotation ids."""

    lane_count = int(sample.lane_count)
    row_count = int(sample.row_count)
    if lane_count < 2:
        raise ValueError("crossing lane_count must be >= 2")
    if row_count < 1:
        raise ValueError("crossing row_count must be positive")
    if len(sample.row_directions) != row_count:
        raise ValueError("row_directions length must match row_count")
    if any(int(direction) not in {-1, 1} for direction in sample.row_directions):
        raise ValueError("row directions must be -1 or 1")
    for vehicle in sample.vehicles:
        if not (0 <= int(vehicle.row) < row_count):
            raise ValueError("vehicle row out of range")
        if not (0 <= int(vehicle.start_col) < lane_count):
            raise ValueError("vehicle start_col out of range")
        if int(vehicle.direction) not in {-1, 1}:
            raise ValueError("vehicle direction must be -1 or 1")
    for route in sample.route_options:
        if len(route.path_cols) != row_count:
            raise ValueError("route path length must match row_count")
        if any(int(col) < 0 or int(col) >= lane_count for col in route.path_cols):
            raise ValueError("route path column out of range")

    vehicle_ids = {str(vehicle.vehicle_id) for vehicle in sample.vehicles}
    start_ids = {start_entity_id(index) for index in range(len(sample.start_labels))}
    route_ids = {route_entity_id(route.label) for route in sample.route_options}
    route_cell_ids = {
        route_cell_entity_id(route.label, row)
        for route in sample.route_options
        for row in range(row_count)
    }
    known_entities = vehicle_ids | start_ids | route_ids | route_cell_ids
    if not set(sample.annotation_entity_ids) <= known_entities:
        raise ValueError("crossing annotation references unknown entities")

    count_mode = str(sample.count_mode)
    if count_mode == "route_intersections":
        marked = next((route for route in sample.route_options if route.label == sample.marked_route_label), None)
        if marked is None:
            raise ValueError("route intersection count requires marked route")
        expected_hit_ids = route_collision_vehicle_ids(marked, sample.vehicles, lane_count=lane_count)
        expected_answer = len(expected_hit_ids)
        expected_annotation = set(expected_hit_ids)
        if int(sample.answer) != int(expected_answer):
            raise ValueError("crossing answer does not match active query")
    elif count_mode in {"left_movers", "right_movers"}:
        target_direction = -1 if count_mode == "left_movers" else 1
        expected_annotation = {
            str(vehicle.vehicle_id)
            for vehicle in sample.vehicles
            if int(vehicle.direction) == int(target_direction)
        }
        expected_answer = len(expected_annotation)
        if int(sample.answer) != int(expected_answer):
            raise ValueError("crossing answer does not match active query")
    elif count_mode == "labeled_route_collision":
        marked = next((route for route in sample.route_options if route.label == sample.marked_route_label), None)
        if marked is None:
            raise ValueError("hit-object label task requires marked route")
        expected_hit_ids = route_collision_vehicle_ids(marked, sample.vehicles, lane_count=lane_count)
        if len(expected_hit_ids) != 1:
            raise ValueError("hit-object label task requires exactly one collision")
        vehicle_by_id = {str(vehicle.vehicle_id): vehicle for vehicle in sample.vehicles}
        target_vehicle = vehicle_by_id.get(str(expected_hit_ids[0]))
        if target_vehicle is None or target_vehicle.option_label is None:
            raise ValueError("hit-object label task collision target must have an option label")
        expected_answer = str(target_vehicle.option_label)
        expected_annotation = {str(target_vehicle.vehicle_id)}
        if str(sample.answer) != str(expected_answer):
            raise ValueError("crossing answer does not match active query")
    elif count_mode == "labeled_first_exit":
        labeled_vehicles = tuple(vehicle for vehicle in sample.vehicles if vehicle.option_label is not None)
        if len(labeled_vehicles) != 4:
            raise ValueError("first-exit label task requires exactly four labeled objects")
        exit_ticks = {
            str(vehicle.vehicle_id): vehicle_exit_tick(vehicle, lane_count=lane_count)
            for vehicle in labeled_vehicles
        }
        earliest_tick = min(exit_ticks.values())
        earliest_ids = tuple(sorted(vehicle_id for vehicle_id, tick in exit_ticks.items() if int(tick) == int(earliest_tick)))
        if len(earliest_ids) != 1:
            raise ValueError("first-exit label task requires one earliest labeled object")
        vehicle_by_id = {str(vehicle.vehicle_id): vehicle for vehicle in labeled_vehicles}
        target_vehicle = vehicle_by_id[str(earliest_ids[0])]
        expected_answer = str(target_vehicle.option_label)
        expected_annotation = {str(target_vehicle.vehicle_id)}
        if sample.route_options or sample.marked_route_label is not None:
            raise ValueError("first-exit label task must not use a route")
        if str(sample.answer) != str(expected_answer):
            raise ValueError("crossing answer does not match active query")
    else:
        raise ValueError(f"unsupported crossing count mode: {count_mode}")

    if set(sample.annotation_entity_ids) != set(expected_annotation):
        raise ValueError("crossing annotation ids do not match active query")


__all__ = [
    "route_collision_vehicle_ids",
    "route_first_collision_tick",
    "validate_crossing_sample",
    "vehicle_col_at_tick",
    "vehicle_exit_tick",
]
