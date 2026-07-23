"""Sampling helpers for polar graph paper readout tasks."""

from __future__ import annotations

from collections.abc import Mapping
import random
from typing import Any

from .state import (
    PolarCoordinateCountCase,
    PolarDifferenceCase,
    PolarPointSpec,
    PolarReadoutCase,
    ReadoutComponent,
)

POINT_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _int_param(
    params: Mapping[str, Any],
    name: str,
    *,
    default: int,
    minimum: int | None = None,
    maximum: int | None = None,
) -> int:
    value = int(params.get(name, default))
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ValueError(f"{name} must be <= {maximum}, got {value}")
    return value


def _angle_support(step_degrees: int) -> tuple[int, ...]:
    if step_degrees <= 0 or 360 % step_degrees != 0:
        raise ValueError("sample_angle_step_degrees must divide 360")
    return tuple(range(0, 360, step_degrees))


def _smaller_angle_difference(theta_a: int, theta_b: int) -> int:
    raw = abs(int(theta_a) - int(theta_b)) % 360
    return min(raw, 360 - raw)


def _sample_distinct_pair(rng: random.Random, support: tuple[int, ...]) -> tuple[int, int]:
    if len(support) < 2:
        raise ValueError("support must contain at least two values")
    first, second = rng.sample(list(support), 2)
    return int(first), int(second)


def select_polar_readout_case(
    *,
    rng: random.Random,
    component: ReadoutComponent,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> PolarReadoutCase:
    """Sample one grid-aligned point and bind the requested readout value."""

    radius_min = _int_param(generation_defaults, "radius_min", default=1, minimum=1)
    radius_max = _int_param(
        generation_defaults,
        "radius_max",
        default=8,
        minimum=radius_min,
    )
    angle_step = _int_param(
        generation_defaults,
        "sample_angle_step_degrees",
        default=30,
        minimum=1,
        maximum=180,
    )
    angle_support = _angle_support(angle_step)
    radius_support = tuple(range(radius_min, radius_max + 1))

    if "radius" in params:
        radius = _int_param(params, "radius", default=radius_min, minimum=radius_min, maximum=radius_max)
    else:
        radius = rng.choice(radius_support)

    if "theta_degrees" in params:
        theta_degrees = _int_param(params, "theta_degrees", default=0, minimum=0, maximum=359)
        if theta_degrees not in angle_support:
            raise ValueError("theta_degrees must lie on the configured angular support")
    else:
        theta_degrees = rng.choice(angle_support)

    if component == "radius":
        correct_value = radius
    elif component == "angle_degrees":
        correct_value = theta_degrees
    else:  # pragma: no cover - typing guard
        raise ValueError(f"unknown readout component: {component}")

    return PolarReadoutCase(
        component=component,
        radius=radius,
        theta_degrees=theta_degrees,
        correct_value=correct_value,
    )


def select_polar_difference_case(
    *,
    rng: random.Random,
    component: ReadoutComponent,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> PolarDifferenceCase:
    """Sample two grid-aligned polar points and bind the requested difference."""

    radius_min = _int_param(generation_defaults, "radius_min", default=1, minimum=1)
    radius_max = _int_param(
        generation_defaults,
        "radius_max",
        default=8,
        minimum=radius_min + 1,
    )
    angle_step = _int_param(
        generation_defaults,
        "sample_angle_step_degrees",
        default=30,
        minimum=1,
        maximum=180,
    )
    angle_support = _angle_support(angle_step)
    radius_support = tuple(range(radius_min, radius_max + 1))

    if "radius_p" in params:
        radius_p = _int_param(params, "radius_p", default=radius_min, minimum=radius_min, maximum=radius_max)
    else:
        radius_p = rng.choice(radius_support)
    if "radius_q" in params:
        radius_q = _int_param(params, "radius_q", default=radius_max, minimum=radius_min, maximum=radius_max)
    else:
        radius_q = rng.choice(radius_support)

    if component == "radius":
        attempts = 0
        while radius_q == radius_p and attempts < 32:
            radius_q = rng.choice(radius_support)
            attempts += 1
        if radius_q == radius_p:
            radius_p, radius_q = radius_support[0], radius_support[-1]
    elif radius_q == radius_p:
        radius_q = radius_support[-1] if radius_p != radius_support[-1] else radius_support[0]

    if "theta_degrees_p" in params:
        theta_p = _int_param(params, "theta_degrees_p", default=0, minimum=0, maximum=359)
        if theta_p not in angle_support:
            raise ValueError("theta_degrees_p must lie on the configured angular support")
    else:
        theta_p = rng.choice(angle_support)
    if "theta_degrees_q" in params:
        theta_q = _int_param(params, "theta_degrees_q", default=angle_step, minimum=0, maximum=359)
        if theta_q not in angle_support:
            raise ValueError("theta_degrees_q must lie on the configured angular support")
    else:
        theta_q = rng.choice(angle_support)

    if component == "angle_degrees":
        attempts = 0
        while _smaller_angle_difference(theta_p, theta_q) == 0 and attempts < 32:
            theta_q = rng.choice(angle_support)
            attempts += 1
        if _smaller_angle_difference(theta_p, theta_q) == 0:
            theta_p, theta_q = _sample_distinct_pair(rng, angle_support)
    elif _smaller_angle_difference(theta_p, theta_q) == 0:
        theta_q = (theta_p + angle_step) % 360

    if component == "radius":
        correct_value = abs(int(radius_p) - int(radius_q))
    elif component == "angle_degrees":
        correct_value = _smaller_angle_difference(theta_p, theta_q)
    else:  # pragma: no cover - typing guard
        raise ValueError(f"unknown difference component: {component}")

    return PolarDifferenceCase(
        component=component,
        point_p=PolarPointSpec(label="P", radius=int(radius_p), theta_degrees=int(theta_p)),
        point_q=PolarPointSpec(label="Q", radius=int(radius_q), theta_degrees=int(theta_q)),
        correct_value=int(correct_value),
    )


def _sample_count_param(
    *,
    rng: random.Random,
    params: Mapping[str, Any],
    name: str,
    minimum: int,
    maximum: int,
) -> int:
    if name in params:
        return _int_param(params, name, default=minimum, minimum=minimum, maximum=maximum)
    return int(rng.randint(int(minimum), int(maximum)))


def _sample_unique_grid_points(
    rng: random.Random,
    candidates: list[tuple[int, int]],
    *,
    count: int,
    excluded: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    available = [candidate for candidate in candidates if candidate not in (excluded or set())]
    if len(available) < count:
        raise ValueError(f"not enough unique polar grid points to sample {count} points")
    return [(int(radius), int(theta)) for radius, theta in rng.sample(available, int(count))]


def select_polar_coordinate_count_case(
    *,
    rng: random.Random,
    component: ReadoutComponent,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
) -> PolarCoordinateCountCase:
    """Sample 8-12 marked points with an exact radius/angle match count."""

    radius_min = _int_param(generation_defaults, "radius_min", default=1, minimum=1)
    radius_max = _int_param(
        generation_defaults,
        "radius_max",
        default=10,
        minimum=radius_min,
    )
    angle_step = _int_param(
        generation_defaults,
        "sample_angle_step_degrees",
        default=30,
        minimum=1,
        maximum=180,
    )
    answer_min = _int_param(generation_defaults, "coordinate_count_min", default=1, minimum=1)
    answer_max = _int_param(
        generation_defaults,
        "coordinate_count_max",
        default=5,
        minimum=answer_min,
    )
    total_min = _int_param(generation_defaults, "coordinate_count_total_min", default=8, minimum=answer_min)
    total_max = _int_param(
        generation_defaults,
        "coordinate_count_total_max",
        default=12,
        minimum=total_min,
        maximum=len(POINT_LABELS),
    )

    angle_support = _angle_support(angle_step)
    radius_support = tuple(range(radius_min, radius_max + 1))
    answer_count = _sample_count_param(
        rng=rng,
        params=params,
        name="answer_count",
        minimum=answer_min,
        maximum=min(answer_max, total_max),
    )
    total_count = _sample_count_param(
        rng=rng,
        params=params,
        name="total_point_count",
        minimum=max(total_min, answer_count),
        maximum=total_max,
    )

    if component == "radius":
        if "target_radius" in params:
            target_value = _int_param(
                params,
                "target_radius",
                default=radius_min,
                minimum=radius_min,
                maximum=radius_max,
            )
        else:
            target_value = int(rng.choice(radius_support))
        matching_coords = [
            (int(target_value), int(theta))
            for theta in rng.sample(list(angle_support), int(answer_count))
        ]
        distractor_candidates = [
            (int(radius), int(theta))
            for radius in radius_support
            if int(radius) != int(target_value)
            for theta in angle_support
        ]
    elif component == "angle_degrees":
        if "target_angle_degrees" in params:
            target_value = _int_param(params, "target_angle_degrees", default=0, minimum=0, maximum=359)
            if target_value not in angle_support:
                raise ValueError("target_angle_degrees must lie on the configured angular support")
        else:
            target_value = int(rng.choice(angle_support))
        matching_coords = [
            (int(radius), int(target_value))
            for radius in rng.sample(list(radius_support), int(answer_count))
        ]
        distractor_candidates = [
            (int(radius), int(theta))
            for radius in radius_support
            for theta in angle_support
            if int(theta) != int(target_value)
        ]
    else:  # pragma: no cover - typing guard
        raise ValueError(f"unknown count component: {component}")

    distractor_coords = _sample_unique_grid_points(
        rng,
        distractor_candidates,
        count=int(total_count) - int(answer_count),
        excluded=set(matching_coords),
    )
    all_coords = [*matching_coords, *distractor_coords]
    rng.shuffle(all_coords)

    points: list[PolarPointSpec] = []
    matching_labels: list[str] = []
    for index, (radius, theta) in enumerate(all_coords):
        label = POINT_LABELS[index]
        point = PolarPointSpec(label=label, radius=int(radius), theta_degrees=int(theta))
        points.append(point)
        if (component == "radius" and int(radius) == int(target_value)) or (
            component == "angle_degrees" and int(theta) == int(target_value)
        ):
            matching_labels.append(label)

    if len(matching_labels) != int(answer_count):
        raise RuntimeError("sampled polar coordinate count does not match target answer")

    return PolarCoordinateCountCase(
        component=component,
        target_value=int(target_value),
        points=tuple(points),
        matching_labels=tuple(matching_labels),
        correct_value=int(answer_count),
    )
