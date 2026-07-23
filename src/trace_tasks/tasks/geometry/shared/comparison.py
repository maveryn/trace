"""Shared helpers for geometry/comparison task modules."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.sampling import normalize_positive_weights, weighted_choice
from ...shared.geometry_primitives import Point
from .graph_rendering import graph_units_to_pixel

COMPARISON_QUERY_TYPES: Tuple[str, str] = ("largest", "smallest")
COMPARISON_ANSWER_LABEL_POOL: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H", "I")
COMPARISON_REGION_SHAPE_FAMILIES: Tuple[str, str] = ("rectangle", "triangle")


def trace_numeric_value(value: float, *, precision: int = 6):
    """Return a JSON-friendly numeric value without noisy float tails."""

    numeric = round(float(value), int(precision))
    if float(numeric).is_integer():
        return int(numeric)
    return float(numeric)


def resolve_region_shape_family(params: Mapping[str, Any], *, fallback: str = "rectangle") -> str:
    """Resolve the polygon family used by area/perimeter comparison scenes."""

    selected = str(params.get("shape_family", fallback)).strip().lower()
    if selected not in set(COMPARISON_REGION_SHAPE_FAMILIES):
        raise ValueError(f"unsupported comparison shape_family: {selected}")
    return str(selected)


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (
        (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1]))
        - (float(b[1]) - float(a[1])) * (float(c[0]) - float(a[0]))
    )


def _point_segment_distance(point: Point, segment_a: Point, segment_b: Point) -> float:
    px, py = float(point[0]), float(point[1])
    ax, ay = float(segment_a[0]), float(segment_a[1])
    bx, by = float(segment_b[0]), float(segment_b[1])
    dx = bx - ax
    dy = by - ay
    denom = (dx * dx) + (dy * dy)
    if denom <= 1e-9:
        return math.hypot(px - ax, py - ay)
    t_value = max(0.0, min(1.0, (((px - ax) * dx) + ((py - ay) * dy)) / denom))
    closest = (ax + (t_value * dx), ay + (t_value * dy))
    return math.hypot(px - closest[0], py - closest[1])


def _segments_intersect(segment_a: Tuple[Point, Point], segment_b: Tuple[Point, Point]) -> bool:
    a0, a1 = segment_a
    b0, b1 = segment_b
    o1 = _orientation(a0, a1, b0)
    o2 = _orientation(a0, a1, b1)
    o3 = _orientation(b0, b1, a0)
    o4 = _orientation(b0, b1, a1)
    return (float(o1) * float(o2) < 0.0) and (float(o3) * float(o4) < 0.0)


def _segment_clearance(segment_a: Tuple[Point, Point], segment_b: Tuple[Point, Point]) -> float:
    return min(
        _point_segment_distance(segment_a[0], segment_b[0], segment_b[1]),
        _point_segment_distance(segment_a[1], segment_b[0], segment_b[1]),
        _point_segment_distance(segment_b[0], segment_a[0], segment_a[1]),
        _point_segment_distance(segment_b[1], segment_a[0], segment_a[1]),
    )


def _polygon_edges(vertices: Sequence[Point]) -> Tuple[Tuple[Point, Point], ...]:
    points = tuple((float(point[0]), float(point[1])) for point in vertices)
    if len(points) < 3:
        raise ValueError("polygon clearance requires at least three vertices")
    return tuple((points[index], points[(index + 1) % len(points)]) for index in range(len(points)))


def polygon_has_clearance(
    candidate_vertices: Sequence[Point],
    existing_polygons: Sequence[Sequence[Point]],
    *,
    min_segment_clearance_px: float,
    min_vertex_clearance_px: float,
) -> bool:
    """Return whether one polygon is separated from already placed polygons."""

    candidate_edges = _polygon_edges(candidate_vertices)
    candidate_points = tuple((float(point[0]), float(point[1])) for point in candidate_vertices)
    for existing_vertices in existing_polygons:
        existing_edges = _polygon_edges(existing_vertices)
        existing_points = tuple((float(point[0]), float(point[1])) for point in existing_vertices)
        for candidate_edge in candidate_edges:
            for existing_edge in existing_edges:
                if _segments_intersect(candidate_edge, existing_edge):
                    return False
                if _segment_clearance(candidate_edge, existing_edge) < float(min_segment_clearance_px):
                    return False
        for point_a in candidate_points:
            for point_b in existing_points:
                if math.hypot(float(point_a[0]) - float(point_b[0]), float(point_a[1]) - float(point_b[1])) < float(
                    min_vertex_clearance_px
                ):
                    return False
    return True


def resolve_comparison_query_type(
    rng,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
) -> Tuple[str, Dict[str, float]]:
    """Resolve comparison query type with weighted defaults."""

    explicit = params.get("query_type")
    if explicit is not None:
        selected = str(explicit).strip().lower()
        if selected not in set(COMPARISON_QUERY_TYPES):
            raise ValueError(f"unsupported query_type: {selected}")
        return selected, {
            key: (1.0 if key == selected else 0.0)
            for key in sorted(COMPARISON_QUERY_TYPES)
        }

    raw_weights = params.get(
        "query_type_weights",
        gen_defaults.get("query_type_weights", {key: 1.0 for key in COMPARISON_QUERY_TYPES}),
    )
    if not isinstance(raw_weights, Mapping):
        raise ValueError("query_type_weights must be a mapping when provided")
    weights = {
        str(key): float(value)
        for key, value in raw_weights.items()
        if str(key) in set(COMPARISON_QUERY_TYPES)
    }
    probabilities = normalize_positive_weights(
        weights,
        default_keys=COMPARISON_QUERY_TYPES,
    )
    selected = weighted_choice(rng, probabilities, sort_keys=True)
    return str(selected), {
        str(key): float(value) for key, value in sorted(probabilities.items())
    }


def resolve_comparison_object_count(
    rng,
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    fallback_min: int,
    fallback_max: int,
) -> Tuple[int, Dict[str, float]]:
    """Resolve how many compared objects appear in the scene."""

    min_count = int(params.get("object_count_min", gen_defaults.get("object_count_min", int(fallback_min))))
    max_count = int(params.get("object_count_max", gen_defaults.get("object_count_max", int(fallback_max))))
    if min_count < 2 or min_count > max_count:
        raise ValueError("invalid object_count_min/object_count_max for comparison task")
    supported_counts = [int(value) for value in range(int(min_count), int(max_count) + 1)]

    explicit = params.get("object_count")
    if explicit is not None:
        selected = int(explicit)
        if int(selected) not in set(supported_counts):
            raise ValueError("object_count is outside configured supported range")
        return int(selected), {
            str(value): (1.0 if int(value) == int(selected) else 0.0)
            for value in supported_counts
        }

    raw_weights = params.get(
        "object_count_weights",
        gen_defaults.get("object_count_weights", {str(value): 1.0 for value in supported_counts}),
    )
    if not isinstance(raw_weights, Mapping):
        raise ValueError("object_count_weights must be a mapping when provided")
    weights = {
        str(key): float(value)
        for key, value in raw_weights.items()
        if str(key) in {str(value) for value in supported_counts}
    }
    probabilities = normalize_positive_weights(
        weights,
        default_keys=[str(value) for value in supported_counts],
    )
    selected = int(weighted_choice(rng, probabilities, sort_keys=True))
    return int(selected), {
        str(key): float(value)
        for key, value in sorted(probabilities.items(), key=lambda item: int(item[0]))
    }


def resolve_comparison_winner_label(
    rng,
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    label_pool: Sequence[str] = COMPARISON_ANSWER_LABEL_POOL,
    selection_namespace: str = "comparison_winner_label",
) -> Tuple[str, Dict[str, float]]:
    """Resolve the intended winning answer label for one comparison scene."""

    normalized_pool = tuple(str(label).upper() for label in label_pool)
    explicit = params.get("winner_label")
    if explicit is not None:
        selected = str(explicit).strip().upper()
        if selected not in set(normalized_pool):
            raise ValueError(f"unsupported winner_label: {selected}")
        return selected, {
            key: (1.0 if key == selected else 0.0)
            for key in normalized_pool
        }

    raw_weights = params.get("winner_label_weights", {key: 1.0 for key in normalized_pool})
    if not isinstance(raw_weights, Mapping):
        raise ValueError("winner_label_weights must be a mapping when provided")
    weights = {
        str(key).upper(): float(value)
        for key, value in raw_weights.items()
        if str(key).upper() in set(normalized_pool)
    }
    probabilities = normalize_positive_weights(weights, default_keys=normalized_pool)
    selected = str(weighted_choice(rng, probabilities, sort_keys=True)).upper()

    return selected, {
        str(key): float(value) for key, value in sorted(probabilities.items())
    }


def apply_balanced_comparison_axes(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    query_type: str,
    query_type_probabilities: Mapping[str, float],
    object_count: int,
    object_count_probabilities: Mapping[str, float],
    query_types: Sequence[str] = COMPARISON_QUERY_TYPES,
) -> Tuple[str, Dict[str, float], int, Dict[str, float]]:
    """Return already sampled comparison axes and probability metadata."""

    resolved_query = str(query_type)
    resolved_count = int(object_count)
    query_probs = {str(key): float(value) for key, value in query_type_probabilities.items()}
    count_probs = {str(key): float(value) for key, value in object_count_probabilities.items()}

    return resolved_query, query_probs, resolved_count, count_probs


def slot_centers_graph_units(*, object_count: int, graph_cells: int, rng) -> List[Tuple[int, int]]:
    """Resolve a subset of well-separated graph-unit slot centers."""

    count = int(object_count)
    half_span = max(8, int(graph_cells // 2))
    columns = 3 if count > 6 else max(2, min(4, int((count + 1) // 2)))
    x_limit = min(max(7, int(round(float(graph_cells) * 0.38))), max(7, int(half_span - 3)))
    y_step = min(max(6, int(round(float(graph_cells) * 0.30))), max(6, int(half_span - 4)))
    if columns == 2:
        x_values = [-int(x_limit), int(x_limit)]
    else:
        x_values = [
            int(round(-float(x_limit) + (2.0 * float(x_limit) * float(index) / float(columns - 1))))
            for index in range(columns)
        ]
    y_values = [int(y_step), -int(y_step)] if count <= 6 else [int(y_step), 0, -int(y_step)]
    all_slots = [(int(x), int(y)) for y in y_values for x in x_values]
    rng.shuffle(all_slots)
    return list(all_slots[:count])


def bulky_slot_centers_graph_units(
    *,
    object_count: int,
    graph_cells: int,
    rng,
) -> List[Tuple[int, int]]:
    """Resolve a roomier subset of graph-unit slot centers for bulky objects.

    Comparison tasks over area/perimeter tend to need larger footprints than
    angle or segment scenes, so they use a wider two-column slot bank.
    """

    count = int(object_count)
    half_span = max(10, int(graph_cells // 2))
    columns = 3 if count > 6 else max(2, min(4, int((count + 1) // 2)))
    x_limit = min(max(9, int(round(float(graph_cells) * 0.38))), max(9, int(half_span - 3)))
    y_step = min(max(7, int(round(float(graph_cells) * 0.28))), max(7, int(half_span - 4)))
    if columns == 2:
        x_values = [-int(x_limit), int(x_limit)]
    else:
        x_values = [
            int(round(-float(x_limit) + (2.0 * float(x_limit) * float(index) / float(columns - 1))))
            for index in range(columns)
        ]
    y_values = [int(y_step), -int(y_step)] if count <= 6 else [int(y_step), 0, -int(y_step)]
    all_slots = [(int(x), int(y)) for y in y_values for x in x_values]
    rng.shuffle(all_slots)
    return list(all_slots[:count])


__all__ = [
    "COMPARISON_ANSWER_LABEL_POOL",
    "COMPARISON_REGION_SHAPE_FAMILIES",
    "COMPARISON_QUERY_TYPES",
    "apply_balanced_comparison_axes",
    "bulky_slot_centers_graph_units",
    "graph_units_to_pixel",
    "polygon_has_clearance",
    "resolve_comparison_object_count",
    "resolve_comparison_query_type",
    "resolve_comparison_winner_label",
    "resolve_region_shape_family",
    "slot_centers_graph_units",
    "trace_numeric_value",
]
