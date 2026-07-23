"""Sampling and geometry primitives for orbital-motion diagrams."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence, Tuple

from trace_tasks.core.seed import hash64, spawn_rng

from .state import FOCUS_OPTION_LABELS, SCENE_NAMESPACE, SPEED_OPTION_LABELS, OrbitSpec


def rotated_point(center: Tuple[float, float], x: float, y: float, theta: float) -> Tuple[float, float]:
    """Return one point after rotating a local ellipse coordinate around center."""

    cos_t = math.cos(float(theta))
    sin_t = math.sin(float(theta))
    return (
        float(center[0] + x * cos_t - y * sin_t),
        float(center[1] + x * sin_t + y * cos_t),
    )


def major_axis_unit(theta: float) -> Tuple[float, float]:
    """Return the unit vector along the rotated major axis."""

    return (float(math.cos(theta)), float(math.sin(theta)))


def point_on_orbit(spec: OrbitSpec, angle_rad: float) -> Tuple[float, float]:
    """Return a point on the rendered ellipse."""

    return rotated_point(
        spec.center,
        float(spec.semi_major * math.cos(angle_rad)),
        float(spec.semi_minor * math.sin(angle_rad)),
        spec.rotation_rad,
    )


def _base_orbit_geometry(
    instance_seed: int,
    *,
    render_defaults: Mapping[str, Any],
    namespace: str,
) -> tuple[
    Tuple[float, float],
    float,
    float,
    float,
    int,
    Tuple[float, float],
    Tuple[float, float],
    Tuple[Tuple[float, float], Tuple[float, float]],
    float,
]:
    """Resolve the shared ellipse geometry used by both focus and speed objectives."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.orbit")
    canvas_width = int(render_defaults.get("canvas_width", 1040))
    canvas_height = int(render_defaults.get("canvas_height", 720))
    center = (
        float(canvas_width * 0.50 + rng.randint(-28, 28)),
        float(canvas_height * 0.52 + rng.randint(-18, 20)),
    )
    semi_major = float(rng.choice([270, 290, 310]))
    semi_minor = float(rng.choice([176, 192, 208]))
    if semi_minor >= semi_major:
        semi_minor = float(semi_major * 0.55)
    rotation_rad = math.radians(float(rng.choice([-24, -14, 0, 16, 26])))
    focus_distance = math.sqrt(float(semi_major * semi_major - semi_minor * semi_minor))
    axis_unit = major_axis_unit(rotation_rad)
    focus_side = 1 if int(hash64(int(instance_seed), f"{namespace}.focus_side", 0) % 2) == 0 else -1
    focus = (
        float(center[0] + focus_side * focus_distance * axis_unit[0]),
        float(center[1] + focus_side * focus_distance * axis_unit[1]),
    )
    opposite_focus = (
        float(center[0] - focus_side * focus_distance * axis_unit[0]),
        float(center[1] - focus_side * focus_distance * axis_unit[1]),
    )
    major_end_1 = (
        float(center[0] + semi_major * axis_unit[0]),
        float(center[1] + semi_major * axis_unit[1]),
    )
    major_end_2 = (
        float(center[0] - semi_major * axis_unit[0]),
        float(center[1] - semi_major * axis_unit[1]),
    )
    return (
        center,
        semi_major,
        semi_minor,
        rotation_rad,
        int(focus_side),
        focus,
        opposite_focus,
        (major_end_1, major_end_2),
        float(focus_distance / semi_major),
    )


def _shuffled_labels(instance_seed: int, *, namespace: str, labels: Sequence[str]) -> list[str]:
    labels = [str(label) for label in labels]
    rng = spawn_rng(int(instance_seed), f"{namespace}.orbit")
    rng.shuffle(labels)
    return labels


def _speed_candidate_angles(
    instance_seed: int,
    *,
    direction: str,
    focus_side: int,
    namespace: str,
    candidate_count: int,
) -> list[float]:
    """Sample non-vertex orbit positions with a unique nearest/farthest candidate."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.speed_candidates.{direction}")
    target_angle = 0.0 if int(focus_side) > 0 else math.pi
    if direction == "least":
        target_angle = (target_angle + math.pi) % (2.0 * math.pi)
    for _attempt in range(120):
        angles: list[float] = []
        while len(angles) < int(candidate_count):
            angle = float(rng.random() * 2.0 * math.pi)
            # Avoid exact major-axis vertices; the task asks among labeled
            # candidates, so the answer should not be a fixed perihelion/aphelion shortcut.
            major_axis_distance = min(
                abs((angle - 0.0 + math.pi) % (2.0 * math.pi) - math.pi),
                abs((angle - math.pi + math.pi) % (2.0 * math.pi) - math.pi),
            )
            if major_axis_distance < math.radians(13):
                continue
            if any(abs((angle - other + math.pi) % (2.0 * math.pi) - math.pi) < math.radians(24) for other in angles):
                continue
            angles.append(angle)
        order = sorted(
            range(len(angles)),
            key=lambda idx: abs((float(angles[idx]) - target_angle + math.pi) % (2.0 * math.pi) - math.pi),
        )
        # Make the selected candidate visually distinct from the next closest
        # candidate along the orbit without forcing it onto the axis vertex.
        if (
            abs((float(angles[order[1]]) - target_angle + math.pi) % (2.0 * math.pi) - math.pi)
            - abs((float(angles[order[0]]) - target_angle + math.pi) % (2.0 * math.pi) - math.pi)
            >= math.radians(18)
        ):
            return [float(angles[order[0]]), *[float(angles[idx]) for idx in order[1:]]]
    fallback_offsets = [26, 64, 126, 218]
    if direction == "least":
        fallback_offsets = [26, -64, 126, -218]
    return [float((target_angle + math.radians(offset)) % (2.0 * math.pi)) for offset in fallback_offsets]


def make_focus_location_spec(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> OrbitSpec:
    """Build an ellipse with labeled candidate points, one of which is a focus."""

    _ = dict(params or {})
    center, semi_major, semi_minor, rotation_rad, focus_side, focus, _opposite_focus, endpoints, eccentricity = (
        _base_orbit_geometry(int(instance_seed), render_defaults=render_defaults, namespace=str(namespace))
    )
    labels = _shuffled_labels(int(instance_seed), namespace=str(namespace), labels=FOCUS_OPTION_LABELS)
    focus_distance = math.sqrt(float(semi_major * semi_major - semi_minor * semi_minor))
    raw_points = [
        focus,
        center,
        rotated_point(center, focus_side * 0.34 * focus_distance, 0.0, rotation_rad),
        rotated_point(center, -focus_side * 0.58 * focus_distance, 0.0, rotation_rad),
        rotated_point(center, -focus_side * 0.18 * semi_major, 0.78 * semi_minor, rotation_rad),
        rotated_point(center, -focus_side * 0.42 * semi_major, -0.42 * semi_minor, rotation_rad),
    ]
    selected_label = str(labels[0])
    candidate_points = {str(label): tuple(point) for label, point in zip(labels, raw_points)}
    return OrbitSpec(
        center=tuple(center),
        semi_major=float(semi_major),
        semi_minor=float(semi_minor),
        rotation_rad=float(rotation_rad),
        focus_side=int(focus_side),
        candidate_points=candidate_points,
        selected_label=selected_label,
        selected_point=tuple(focus),
        sun_point=None,
        major_axis_endpoints=tuple(endpoints),
        eccentricity=float(eccentricity),
    )


def make_speed_extremum_spec(
    instance_seed: int,
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    speed_direction: str,
    namespace: str = SCENE_NAMESPACE,
) -> OrbitSpec:
    """Build an orbit with a visible Sun and candidate positions on the ellipse."""

    _ = dict(params or {})
    direction = str(speed_direction)
    if direction not in {"greatest", "least"}:
        raise ValueError(f"unsupported orbital speed direction: {direction}")
    center, semi_major, semi_minor, rotation_rad, focus_side, focus, _opposite_focus, endpoints, eccentricity = (
        _base_orbit_geometry(int(instance_seed), render_defaults=render_defaults, namespace=str(namespace))
    )

    labels = _shuffled_labels(int(instance_seed), namespace=str(namespace), labels=SPEED_OPTION_LABELS)
    selected_label = str(labels[0])
    base_spec = OrbitSpec(
        center=tuple(center),
        semi_major=float(semi_major),
        semi_minor=float(semi_minor),
        rotation_rad=float(rotation_rad),
        focus_side=int(focus_side),
        candidate_points={},
        selected_label=selected_label,
        selected_point=tuple(focus),
        sun_point=tuple(focus),
        major_axis_endpoints=tuple(endpoints),
        eccentricity=float(eccentricity),
    )
    angles = _speed_candidate_angles(
        int(instance_seed),
        direction=direction,
        focus_side=int(focus_side),
        namespace=str(namespace),
        candidate_count=len(labels),
    )
    points = [point_on_orbit(base_spec, angle) for angle in angles]
    selected_point = tuple(points[0])
    candidate_points = {str(label): tuple(point) for label, point in zip(labels, points)}
    return OrbitSpec(
        center=tuple(center),
        semi_major=float(semi_major),
        semi_minor=float(semi_minor),
        rotation_rad=float(rotation_rad),
        focus_side=int(focus_side),
        candidate_points={str(k): tuple(v) for k, v in candidate_points.items()},
        selected_label=selected_label,
        selected_point=tuple(selected_point),
        sun_point=tuple(focus),
        major_axis_endpoints=tuple(endpoints),
        eccentricity=float(eccentricity),
    )


def probability_map(values: Sequence[str]) -> dict[str, float]:
    """Return a uniform probability map over visible branch values."""

    support = tuple(str(value) for value in values if str(value))
    if not support:
        return {}
    probability = 1.0 / float(len(support))
    return {value: float(probability) for value in support}


__all__ = [
    "make_focus_location_spec",
    "make_speed_extremum_spec",
    "major_axis_unit",
    "point_on_orbit",
    "probability_map",
    "rotated_point",
]
