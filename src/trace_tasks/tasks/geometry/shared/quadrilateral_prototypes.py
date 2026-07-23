"""Reusable quadrilateral prototype helpers for geometry tasks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence, Tuple

Point = Tuple[float, float]


@dataclass(frozen=True)
class QuadrilateralPrototype:
    """One centered quadrilateral prototype plus its geometric classification."""

    local_vertices: Tuple[Point, Point, Point, Point]
    quadrilateral_kind: str
    side_lengths: Tuple[float, float, float, float]
    angles_degrees: Tuple[float, float, float, float]


def _edge_vectors(vertices: Sequence[Point]) -> Tuple[Point, Point, Point, Point]:
    """Return directed edge vectors for one ordered quadrilateral."""

    return tuple(
        (
            float(vertices[(index + 1) % 4][0]) - float(vertices[index][0]),
            float(vertices[(index + 1) % 4][1]) - float(vertices[index][1]),
        )
        for index in range(4)
    )  # type: ignore[return-value]


def _side_lengths(vertices: Sequence[Point]) -> Tuple[float, float, float, float]:
    """Return the four side lengths for one ordered quadrilateral."""

    return tuple(
        math.hypot(
            float(vertices[(index + 1) % 4][0]) - float(vertices[index][0]),
            float(vertices[(index + 1) % 4][1]) - float(vertices[index][1]),
        )
        for index in range(4)
    )  # type: ignore[return-value]


def _interior_angles(vertices: Sequence[Point]) -> Tuple[float, float, float, float]:
    """Return the four interior angles in degrees."""

    angles = []
    for index in range(4):
        prev_point = vertices[(index - 1) % 4]
        current_point = vertices[index]
        next_point = vertices[(index + 1) % 4]
        vector_a = (
            float(prev_point[0]) - float(current_point[0]),
            float(prev_point[1]) - float(current_point[1]),
        )
        vector_b = (
            float(next_point[0]) - float(current_point[0]),
            float(next_point[1]) - float(current_point[1]),
        )
        norm_a = math.hypot(float(vector_a[0]), float(vector_a[1]))
        norm_b = math.hypot(float(vector_b[0]), float(vector_b[1]))
        cosine = (
            (float(vector_a[0]) * float(vector_b[0]))
            + (float(vector_a[1]) * float(vector_b[1]))
        ) / max(1e-9, float(norm_a) * float(norm_b))
        cosine = max(-1.0, min(1.0, float(cosine)))
        angles.append(math.degrees(math.acos(float(cosine))))
    return tuple(float(value) for value in angles)  # type: ignore[return-value]


def _center_vertices(vertices: Sequence[Point]) -> Tuple[Point, Point, Point, Point]:
    """Translate vertices so the quadrilateral centroid sits at the origin."""

    cx = sum(float(x) for x, _ in vertices) / 4.0
    cy = sum(float(y) for _, y in vertices) / 4.0
    return tuple((float(x) - float(cx), float(y) - float(cy)) for x, y in vertices)  # type: ignore[return-value]


def classify_quadrilateral_kind(vertices: Sequence[Point], *, tolerance: float = 1e-6) -> str:
    """Return one exclusive quadrilateral class name for the supported families."""

    if len(vertices) != 4:
        raise ValueError("quadrilateral classification requires exactly 4 vertices")
    edges = _edge_vectors(vertices)
    lengths = _side_lengths(vertices)

    def _dot(vec_a: Point, vec_b: Point) -> float:
        return (float(vec_a[0]) * float(vec_b[0])) + (float(vec_a[1]) * float(vec_b[1]))

    def _cross(vec_a: Point, vec_b: Point) -> float:
        return (float(vec_a[0]) * float(vec_b[1])) - (float(vec_a[1]) * float(vec_b[0]))

    all_right = all(abs(_dot(edges[index], edges[(index + 1) % 4])) <= float(tolerance) for index in range(4))
    all_equal = max(lengths) - min(lengths) <= float(tolerance)
    opposite_parallel = (
        abs(_cross(edges[0], edges[2])) <= float(tolerance)
        and abs(_cross(edges[1], edges[3])) <= float(tolerance)
    )
    if bool(all_right) and bool(all_equal):
        return "square"
    if bool(all_right):
        return "rectangle_non_square"
    if bool(all_equal):
        return "rhombus_non_square"
    if bool(opposite_parallel):
        return "parallelogram_only"
    return "other"


def build_quadrilateral_prototype(vertices: Sequence[Point]) -> QuadrilateralPrototype:
    """Return one normalized quadrilateral prototype from ordered vertices."""

    centered = _center_vertices(vertices)
    return QuadrilateralPrototype(
        local_vertices=tuple(centered),
        quadrilateral_kind=str(classify_quadrilateral_kind(centered)),
        side_lengths=tuple(float(value) for value in _side_lengths(centered)),
        angles_degrees=tuple(float(value) for value in _interior_angles(centered)),
    )


def sample_square_prototype(rng, *, min_extent_units: float, max_extent_units: float) -> QuadrilateralPrototype:
    """Return one centered square prototype."""

    side = float(rng.uniform(float(min_extent_units), float(max_extent_units)))
    half = 0.5 * float(side)
    return build_quadrilateral_prototype(
        [(-half, -half), (half, -half), (half, half), (-half, half)]
    )


def sample_rectangle_non_square_prototype(
    rng,
    *,
    min_extent_units: float,
    max_extent_units: float,
    min_side_gap_units: float,
) -> QuadrilateralPrototype:
    """Return one centered rectangle that is not a square."""

    for _ in range(400):
        width = float(rng.uniform(float(min_extent_units), float(max_extent_units)))
        height = float(rng.uniform(float(min_extent_units), float(max_extent_units)))
        if abs(float(width) - float(height)) < float(min_side_gap_units):
            continue
        half_w = 0.5 * float(width)
        half_h = 0.5 * float(height)
        prototype = build_quadrilateral_prototype(
            [(-half_w, -half_h), (half_w, -half_h), (half_w, half_h), (-half_w, half_h)]
        )
        if str(prototype.quadrilateral_kind) == "rectangle_non_square":
            return prototype
    raise ValueError("failed to sample rectangle_non_square prototype")


def sample_rhombus_non_square_prototype(
    rng,
    *,
    min_extent_units: float,
    max_extent_units: float,
    min_side_gap_units: float,
) -> QuadrilateralPrototype:
    """Return one centered rhombus that is not a square."""

    for _ in range(400):
        diag_x = float(rng.uniform(float(min_extent_units), float(max_extent_units) * 1.4))
        diag_y = float(rng.uniform(float(min_extent_units), float(max_extent_units) * 1.4))
        if abs(float(diag_x) - float(diag_y)) < float(min_side_gap_units):
            continue
        prototype = build_quadrilateral_prototype(
            [(0.0, 0.5 * float(diag_y)), (0.5 * float(diag_x), 0.0), (0.0, -0.5 * float(diag_y)), (-0.5 * float(diag_x), 0.0)]
        )
        if str(prototype.quadrilateral_kind) == "rhombus_non_square":
            return prototype
    raise ValueError("failed to sample rhombus_non_square prototype")


def sample_parallelogram_only_prototype(
    rng,
    *,
    min_extent_units: float,
    max_extent_units: float,
    min_side_gap_units: float,
    min_slant_units: float,
) -> QuadrilateralPrototype:
    """Return one centered parallelogram that is neither rectangle nor rhombus."""

    for _ in range(600):
        width = float(rng.uniform(float(min_extent_units), float(max_extent_units)))
        height = float(rng.uniform(float(min_extent_units) * 0.9, float(max_extent_units)))
        slant = float(rng.uniform(float(min_slant_units), float(max_extent_units) * 0.7))
        if abs(float(width) - math.hypot(float(slant), float(height))) < float(min_side_gap_units):
            continue
        prototype = build_quadrilateral_prototype(
            [
                (-0.5 * float(width) - 0.5 * float(slant), -0.5 * float(height)),
                (0.5 * float(width) - 0.5 * float(slant), -0.5 * float(height)),
                (0.5 * float(width) + 0.5 * float(slant), 0.5 * float(height)),
                (-0.5 * float(width) + 0.5 * float(slant), 0.5 * float(height)),
            ]
        )
        if str(prototype.quadrilateral_kind) == "parallelogram_only":
            return prototype
    raise ValueError("failed to sample parallelogram_only prototype")


__all__ = [
    "QuadrilateralPrototype",
    "build_quadrilateral_prototype",
    "classify_quadrilateral_kind",
    "sample_parallelogram_only_prototype",
    "sample_rectangle_non_square_prototype",
    "sample_rhombus_non_square_prototype",
    "sample_square_prototype",
]
