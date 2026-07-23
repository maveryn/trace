"""Reusable lattice-polygon sampling and rigid-transform helpers.

These helpers are intentionally narrow: they support the first geometry
transformation/similarity families, which need asymmetric lattice polygons whose
vertices stay on the graph-paper lattice under translation, reflection, and
quarter-turn rotation.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Dict, Sequence, Tuple

from ...shared.geometry_primitives import Point


Polygon = Tuple[Point, ...]

RIGID_TRANSFORM_RECIPE_IDS: Tuple[str, ...] = (
    "identity",
    "reflect_vertical",
    "reflect_horizontal",
    "rotate_90_cw",
    "rotate_90_ccw",
    "rotate_180",
)

_TRIANGLE_TEMPLATES: Tuple[Polygon, ...] = (
    ((-3.0, -1.0), (2.0, -2.0), (1.0, 3.0)),
    ((-2.0, -3.0), (3.0, -1.0), (-1.0, 2.0)),
    ((-3.0, 1.0), (1.0, -2.0), (2.0, 3.0)),
)
_TRIANGLE_COMPACT_TEMPLATES: Tuple[Polygon, ...] = (
    ((-2.0, -1.0), (1.0, -2.0), (0.0, 2.0)),
    ((-2.0, 0.0), (2.0, -1.0), (1.0, 2.0)),
    ((-1.0, -2.0), (2.0, 0.0), (-2.0, 1.0)),
)
_QUADRILATERAL_TEMPLATES: Tuple[Polygon, ...] = (
    ((-3.0, -1.0), (-1.0, 2.0), (3.0, 1.0), (1.0, -3.0)),
    ((-2.0, -3.0), (-3.0, 1.0), (1.0, 3.0), (3.0, -1.0)),
    ((-3.0, 0.0), (-1.0, 3.0), (2.0, 2.0), (3.0, -2.0)),
)
_QUADRILATERAL_COMPACT_TEMPLATES: Tuple[Polygon, ...] = (
    ((-2.0, -1.0), (-1.0, 2.0), (2.0, 1.0), (1.0, -2.0)),
    ((-2.0, -2.0), (-2.0, 1.0), (1.0, 2.0), (2.0, -1.0)),
    ((-2.0, 0.0), (-1.0, 2.0), (2.0, 1.0), (1.0, -2.0)),
)

_TEMPLATES_BY_PROFILE: Dict[str, Dict[str, Tuple[Polygon, ...]]] = {
    "standard": {
        "triangle": _TRIANGLE_TEMPLATES,
        "quadrilateral": _QUADRILATERAL_TEMPLATES,
    },
    "compact": {
        "triangle": _TRIANGLE_COMPACT_TEMPLATES,
        "quadrilateral": _QUADRILATERAL_COMPACT_TEMPLATES,
    },
}


def sample_asymmetric_polygon_template(scene_variant: str, rng, *, profile: str = "standard") -> Polygon:
    """Sample one asymmetric lattice polygon template for the requested family."""

    family = str(scene_variant).strip().lower()
    normalized_profile = str(profile).strip().lower()
    profile_map = _TEMPLATES_BY_PROFILE.get(normalized_profile)
    if not profile_map:
        raise ValueError(f"unsupported polygon template profile: {profile}")
    templates = profile_map.get(family)
    if not templates:
        raise ValueError(f"unsupported transformation scene_variant: {scene_variant}")
    return tuple(rng.choice(list(templates)))


def translate_polygon(vertices: Sequence[Point], *, dx: int, dy: int) -> Polygon:
    """Translate one lattice polygon by an integer graph-unit vector."""

    return tuple((float(x_value) + float(dx), float(y_value) + float(dy)) for x_value, y_value in vertices)


def scale_polygon(vertices: Sequence[Point], *, factor: int) -> Polygon:
    """Scale one polygon uniformly about the origin by an integer factor."""

    multiplier = int(factor)
    if multiplier <= 0:
        raise ValueError("scale_polygon requires a positive integer factor")
    return tuple((float(x_value) * float(multiplier), float(y_value) * float(multiplier)) for x_value, y_value in vertices)


def scale_polygon_non_uniform(vertices: Sequence[Point], *, scale_x: int, scale_y: int) -> Polygon:
    """Scale one polygon anisotropically about the origin by integer factors."""

    multiplier_x = int(scale_x)
    multiplier_y = int(scale_y)
    if multiplier_x <= 0 or multiplier_y <= 0:
        raise ValueError("scale_polygon_non_uniform requires positive integer factors")
    return tuple(
        (float(x_value) * float(multiplier_x), float(y_value) * float(multiplier_y))
        for x_value, y_value in vertices
    )


def rotate_polygon_quarter_turns(vertices: Sequence[Point], *, quarter_turns: int) -> Polygon:
    """Rotate one lattice polygon about the origin by 90° steps."""

    steps = int(quarter_turns) % 4
    out = [(float(x_value), float(y_value)) for x_value, y_value in vertices]
    for _ in range(steps):
        out = [(float(y_value), -float(x_value)) for x_value, y_value in out]
    return tuple(out)


def reflect_polygon(vertices: Sequence[Point], *, axis_kind: str) -> Polygon:
    """Reflect one lattice polygon across the requested canonical axis."""

    normalized_axis = str(axis_kind).strip().lower()
    if normalized_axis == "vertical":
        return tuple((-float(x_value), float(y_value)) for x_value, y_value in vertices)
    if normalized_axis == "horizontal":
        return tuple((float(x_value), -float(y_value)) for x_value, y_value in vertices)
    raise ValueError(f"unsupported reflection axis_kind: {axis_kind}")


def apply_rigid_transform_recipe(vertices: Sequence[Point], *, recipe: str) -> Polygon:
    """Apply one supported rigid-transform recipe about the origin."""

    normalized = str(recipe).strip().lower()
    if normalized == "identity":
        return tuple((float(x_value), float(y_value)) for x_value, y_value in vertices)
    if normalized == "reflect_vertical":
        return reflect_polygon(vertices, axis_kind="vertical")
    if normalized == "reflect_horizontal":
        return reflect_polygon(vertices, axis_kind="horizontal")
    if normalized == "rotate_90_cw":
        return rotate_polygon_quarter_turns(vertices, quarter_turns=1)
    if normalized == "rotate_90_ccw":
        return rotate_polygon_quarter_turns(vertices, quarter_turns=3)
    if normalized == "rotate_180":
        return rotate_polygon_quarter_turns(vertices, quarter_turns=2)
    raise ValueError(f"unsupported rigid transform recipe: {recipe}")


def polygon_center(vertices: Sequence[Point]) -> Point:
    """Return the arithmetic center of one polygon vertex list."""

    if not vertices:
        raise ValueError("polygon_center requires at least one vertex")
    count = float(len(vertices))
    return (
        sum(float(x_value) for x_value, _ in vertices) / count,
        sum(float(y_value) for _, y_value in vertices) / count,
    )


def ordered_vertex_label_map(vertices: Sequence[Point]) -> Dict[str, Point]:
    """Build the canonical `vertex_i -> point` map used for prompt-facing annotation."""

    return {
        f"vertex_{int(index) + 1}": (float(point[0]), float(point[1]))
        for index, point in enumerate(vertices)
    }


def _sorted_pairwise_squared_distances(vertices: Sequence[Point]) -> Tuple[int, ...]:
    """Return the sorted pairwise squared-distance signature for one polygon."""

    distances = []
    for left_index in range(len(vertices)):
        x_left, y_left = vertices[left_index]
        for right_index in range(left_index + 1, len(vertices)):
            x_right, y_right = vertices[right_index]
            dx = float(x_left) - float(x_right)
            dy = float(y_left) - float(y_right)
            distances.append(int(round((dx * dx) + (dy * dy))))
    return tuple(sorted(int(value) for value in distances))


def polygons_are_congruent(reference_vertices: Sequence[Point], candidate_vertices: Sequence[Point]) -> bool:
    """Return whether two polygons are congruent via pairwise-distance identity."""

    if len(reference_vertices) != len(candidate_vertices):
        return False
    return _sorted_pairwise_squared_distances(reference_vertices) == _sorted_pairwise_squared_distances(candidate_vertices)


def polygons_are_similar(reference_vertices: Sequence[Point], candidate_vertices: Sequence[Point]) -> bool:
    """Return whether two polygons are similar via a uniform pairwise-distance scale."""

    if len(reference_vertices) != len(candidate_vertices):
        return False
    reference_signature = _sorted_pairwise_squared_distances(reference_vertices)
    candidate_signature = _sorted_pairwise_squared_distances(candidate_vertices)
    if len(reference_signature) != len(candidate_signature) or not reference_signature:
        return False
    scale_ratio: Fraction | None = None
    for reference_distance, candidate_distance in zip(reference_signature, candidate_signature):
        if int(reference_distance) <= 0 or int(candidate_distance) <= 0:
            return False
        current_ratio = Fraction(int(candidate_distance), int(reference_distance))
        if scale_ratio is None:
            scale_ratio = current_ratio
            continue
        if current_ratio != scale_ratio:
            return False
    return scale_ratio is not None


__all__ = [
    "Polygon",
    "RIGID_TRANSFORM_RECIPE_IDS",
    "apply_rigid_transform_recipe",
    "ordered_vertex_label_map",
    "polygon_center",
    "polygons_are_congruent",
    "polygons_are_similar",
    "reflect_polygon",
    "rotate_polygon_quarter_turns",
    "sample_asymmetric_polygon_template",
    "scale_polygon",
    "scale_polygon_non_uniform",
    "translate_polygon",
]
