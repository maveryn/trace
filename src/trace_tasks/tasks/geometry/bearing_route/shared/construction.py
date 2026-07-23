"""Semantic route-case construction for bearing-route tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.labeling import LABEL_POOL_SAFE_UPPER, assign_random_shuffled_labels
from trace_tasks.tasks.shared.fixed_query import geometry_selected_probability_map as _probability_map

from .spatial_primitives import bearing_to_unit_vector, normalize_bearing, route_unit_points
from .state import Point, RouteCase


ROUTE_CASES: Tuple[Tuple[int, int, int], ...] = (
    (3, 4, 5),
    (6, 8, 10),
    (5, 12, 13),
    (9, 12, 15),
    (8, 15, 17),
    (12, 16, 20),
    (7, 24, 25),
    (10, 24, 26),
    (20, 21, 29),
    (18, 24, 30),
    (16, 30, 34),
    (12, 35, 37),
)
ENDPOINT_GRID_ROUTE_CASES: Tuple[Tuple[int, int, int], ...] = (
    (3, 5, 6),
    (3, 6, 7),
    (3, 7, 8),
    (4, 6, 7),
    (4, 7, 8),
    (5, 7, 9),
)
CARDINAL_BEARINGS: Tuple[int, ...] = (0, 90, 180, 270)
FINAL_BEARING_VALUES: Tuple[int, ...] = tuple(range(0, 360, 5))
FINAL_BEARING_LEG_LENGTHS: Tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10, 12)
ENDPOINT_RESERVED_VISIBLE_LABELS: Tuple[str, ...] = ("N", "E", "S", "W", "F")
ENDPOINT_CANDIDATE_LABEL_POOL: Tuple[str, ...] = tuple(
    label for label in LABEL_POOL_SAFE_UPPER if label not in set(ENDPOINT_RESERVED_VISIBLE_LABELS)
)


def resolve_endpoint_route_case(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    option_count: int,
    route_case_namespace: str,
    orientation_namespace: str,
    target_index_namespace: str,
    labels_namespace: str,
) -> tuple[RouteCase, Dict[str, float]]:
    """Resolve a labeled-endpoint bearing route without public task routing."""

    explicit = params.get("target_displacement")
    if explicit is not None:
        target_displacement = int(explicit)
        matching = [case for case in ENDPOINT_GRID_ROUTE_CASES if int(case[2]) == target_displacement]
        if not matching:
            raise ValueError(f"target_displacement={target_displacement} is not supported")
        case_index = ENDPOINT_GRID_ROUTE_CASES.index(matching[0])
    else:
        route_rng = spawn_rng(int(instance_seed), str(route_case_namespace))
        case_index = int(uniform_choice(route_rng, tuple(range(len(ENDPOINT_GRID_ROUTE_CASES)))))
    leg_a, leg_b, displacement = ENDPOINT_GRID_ROUTE_CASES[case_index]

    orientation_rng = spawn_rng(int(instance_seed), str(orientation_namespace))
    bearing_a = int(uniform_choice(orientation_rng, CARDINAL_BEARINGS))
    turn_rng = spawn_rng(int(instance_seed), f"{orientation_namespace}.turn")
    turn_left = bool(uniform_choice(turn_rng, (True, False)))
    bearing_b = (int(bearing_a) - 90) % 360 if turn_left else (int(bearing_a) + 90) % 360
    turn_direction = "left" if turn_left else "right"

    if int(option_count) < 4:
        raise ValueError("bearing endpoint route requires at least four candidate positions")
    if int(option_count) > len(ENDPOINT_CANDIDATE_LABEL_POOL):
        raise ValueError("option_count exceeds safe label pool")

    explicit_index = params.get("target_index")
    if explicit_index is None:
        target_rng = spawn_rng(int(instance_seed), str(target_index_namespace))
        target_index = int(uniform_choice(target_rng, tuple(range(int(option_count)))))
    else:
        target_index = int(explicit_index)
        if target_index < 0 or target_index >= int(option_count):
            raise ValueError("target_index is outside option count")

    label_rng = spawn_rng(int(instance_seed), str(labels_namespace))
    labels = assign_random_shuffled_labels(
        label_rng,
        object_count=int(option_count),
        label_pool=ENDPOINT_CANDIDATE_LABEL_POOL,
    )
    answer_probabilities = _probability_map(tuple(range(int(option_count))))

    return (
        RouteCase(
            leg_a=int(leg_a),
            leg_b=int(leg_b),
            displacement=int(displacement),
            bearing_a=int(bearing_a),
            bearing_b=int(bearing_b),
            turn_direction=str(turn_direction),
            option_count=int(option_count),
            target_index=target_index,
            option_labels=tuple(labels),
            final_bearing=None,
        ),
        answer_probabilities,
    )


def resolve_final_bearing_route_case(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    bearing_namespace: str,
    leg_length_namespace: str,
    leg_order_namespace: str,
) -> tuple[RouteCase, Dict[str, float]]:
    """Resolve a two-leg route whose endpoint has a selected direct bearing."""

    explicit = params.get("target_bearing")
    if explicit is not None:
        target_bearing = normalize_bearing(float(explicit))
        if target_bearing not in set(FINAL_BEARING_VALUES):
            raise ValueError(f"target_bearing={target_bearing} is not supported")
        bearing_index = FINAL_BEARING_VALUES.index(int(target_bearing))
    else:
        bearing_rng = spawn_rng(int(instance_seed), str(bearing_namespace))
        bearing_index = int(uniform_choice(bearing_rng, tuple(range(len(FINAL_BEARING_VALUES)))))
        target_bearing = int(FINAL_BEARING_VALUES[int(bearing_index)])

    leg_rng = spawn_rng(int(instance_seed), str(leg_length_namespace))
    length_index = int(uniform_choice(leg_rng, tuple(range(len(FINAL_BEARING_LEG_LENGTHS)))))
    leg_length = int(FINAL_BEARING_LEG_LENGTHS[int(length_index)])

    bearing_a = normalize_bearing(int(target_bearing) - 45)
    bearing_b = normalize_bearing(int(target_bearing) + 45)
    order_rng = spawn_rng(int(instance_seed), str(leg_order_namespace))
    swap_index = int(uniform_choice(order_rng, (0, 1)))
    if swap_index % 2:
        bearing_a, bearing_b = bearing_b, bearing_a
    turn_delta = (int(bearing_b) - int(bearing_a)) % 360
    turn_direction = "right" if int(turn_delta) == 90 else "left"
    probabilities = _probability_map(
        tuple(int(value) for value in FINAL_BEARING_VALUES),
        int(target_bearing) if explicit is not None else None,
    )
    displacement = int(round(math.sqrt(2.0) * float(leg_length)))
    return (
        RouteCase(
            leg_a=int(leg_length),
            leg_b=int(leg_length),
            displacement=int(displacement),
            bearing_a=int(bearing_a),
            bearing_b=int(bearing_b),
            turn_direction=str(turn_direction),
            option_count=0,
            target_index=None,
            option_labels=(),
            final_bearing=int(target_bearing),
        ),
        probabilities,
    )


def candidate_unit_points(route_case: RouteCase) -> list[tuple[str, Point]]:
    """Return semantic endpoint candidates for the rendered option panel."""

    _p0, p1, p2 = route_unit_points(route_case)
    u1 = bearing_to_unit_vector(int(route_case.bearing_a))
    u2 = bearing_to_unit_vector(int(route_case.bearing_b))
    opposite_first = (-u1[0] * float(route_case.leg_a), -u1[1] * float(route_case.leg_a))
    swapped = (
        u1[0] * float(route_case.leg_b) + u2[0] * float(route_case.leg_a),
        u1[1] * float(route_case.leg_b) + u2[1] * float(route_case.leg_a),
    )
    wrong_turn = (
        u1[0] * float(route_case.leg_a) - u2[0] * float(route_case.leg_b),
        u1[1] * float(route_case.leg_a) - u2[1] * float(route_case.leg_b),
    )
    second_only = (u2[0] * float(route_case.leg_b), u2[1] * float(route_case.leg_b))
    candidates = [
        ("correct_endpoint", p2),
        ("first_leg_only", p1),
        ("second_leg_only", second_only),
        ("swapped_distances", swapped),
        ("opposite_second_turn", wrong_turn),
        ("opposite_first_leg", opposite_first),
    ]
    unique: list[tuple[str, Point]] = []
    seen: set[tuple[int, int]] = set()
    for name, point in candidates:
        key = (int(round(point[0] * 1000)), int(round(point[1] * 1000)))
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, point))
    while len(unique) < int(route_case.option_count):
        offset = float(len(unique) + 2)
        unique.append((f"distractor_{len(unique)}", (p2[0] + offset, p2[1] - offset)))
    return unique[: int(route_case.option_count)]


__all__ = [
    "CARDINAL_BEARINGS",
    "ENDPOINT_CANDIDATE_LABEL_POOL",
    "ENDPOINT_GRID_ROUTE_CASES",
    "ENDPOINT_RESERVED_VISIBLE_LABELS",
    "FINAL_BEARING_LEG_LENGTHS",
    "FINAL_BEARING_VALUES",
    "ROUTE_CASES",
    "candidate_unit_points",
    "resolve_endpoint_route_case",
    "resolve_final_bearing_route_case",
]
