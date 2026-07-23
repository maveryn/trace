"""Identity-free sampling primitives for circular-sector formula cases."""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.measurement_rendering import round1
from trace_tasks.tasks.shared.fixed_query import geometry_selected_probability_map

from .state import SectorProblem, SectorValues

PI_VALUE = math.pi

RADIUS_SUPPORT: tuple[int, ...] = (4, 5, 6, 7, 8, 9, 10, 11, 12)
DIRECT_THETA_SUPPORT: tuple[int, ...] = (45, 60, 72, 75, 90, 105, 120, 135, 150)
COMPLEMENT_THETA_SUPPORT: tuple[int, ...] = (25, 30, 35, 40, 45, 50, 55, 60, 65)
SUPPLEMENT_THETA_SUPPORT: tuple[int, ...] = (40, 45, 60, 72, 75, 90, 105, 120, 135, 150)
REMAINING_THETA_SUPPORT: tuple[int, ...] = (45, 60, 72, 90, 108, 120, 135, 144, 150, 180)


def _sector_values(
    *,
    radius_units: int,
    theta_degrees: int,
    target_angle_total: int | None = None,
) -> SectorValues:
    radius = float(radius_units)
    theta = float(theta_degrees)
    arc_length = round1((theta / 360.0) * 2.0 * PI_VALUE * radius)
    sector_area = round1((theta / 360.0) * PI_VALUE * radius**2)
    angle_from_arc = round1((360.0 * arc_length) / (2.0 * PI_VALUE * radius))
    angle_from_area = round1((360.0 * sector_area) / (PI_VALUE * radius**2))
    return SectorValues(
        radius_units=int(radius_units),
        theta_degrees=int(theta_degrees),
        arc_length=float(arc_length),
        sector_area=float(sector_area),
        angle_from_arc_length=float(angle_from_arc),
        angle_from_sector_area=float(angle_from_area),
        adjacent_angle_degrees=None if target_angle_total is None else int(target_angle_total) - int(theta_degrees),
        target_angle_total=None if target_angle_total is None else int(target_angle_total),
    )


def _same_one_decimal(left: float, right: float) -> bool:
    return abs(round1(float(left)) - round1(float(right))) <= 1e-9


def _area_from_arc_is_consistent(values: SectorValues) -> bool:
    return _same_one_decimal(
        0.5 * float(values.radius_units) * float(values.arc_length),
        float(values.sector_area),
    )


def _arc_from_area_is_consistent(values: SectorValues) -> bool:
    return _same_one_decimal(
        (2.0 * float(values.sector_area)) / float(values.radius_units),
        float(values.arc_length),
    )


def _angle_from_arc_is_consistent(values: SectorValues) -> bool:
    return _same_one_decimal(float(values.angle_from_arc_length), float(values.theta_degrees))


def _angle_from_area_is_consistent(values: SectorValues) -> bool:
    return _same_one_decimal(float(values.angle_from_sector_area), float(values.theta_degrees))


def _sample_values(
    *,
    instance_seed: int,
    seed_namespace: str,
    params: Mapping[str, Any],
    theta_support: Sequence[int],
    target_angle_total: int | None = None,
    value_filter: Callable[[SectorValues], bool] | None = None,
    balance_key: Callable[[SectorValues], float] | None = None,
) -> tuple[int, SectorValues, dict[str, float]]:
    """Select one valid sector case while preserving one-decimal formula consistency."""

    theta_values = tuple(int(value) for value in theta_support)
    candidate_cases: list[tuple[int, SectorValues]] = []
    for theta_degrees in theta_values:
        for radius_units in RADIUS_SUPPORT:
            index = (
                theta_values.index(int(theta_degrees)) * len(RADIUS_SUPPORT)
                + RADIUS_SUPPORT.index(int(radius_units))
            )
            values = _sector_values(
                radius_units=int(radius_units),
                theta_degrees=int(theta_degrees),
                target_angle_total=target_angle_total,
            )
            if value_filter is not None and not bool(value_filter(values)):
                continue
            candidate_cases.append((int(index), values))
    if not candidate_cases:
        raise ValueError("sector case pool is empty after consistency filtering")
    if "radius_units" in params:
        radius_units = int(params["radius_units"])
        if radius_units not in set(RADIUS_SUPPORT):
            raise ValueError(f"radius_units={radius_units} is outside supported radius values")
    else:
        rng = spawn_rng(int(instance_seed), f"{seed_namespace}.radius")
        radius_units = int(uniform_choice(rng, RADIUS_SUPPORT))
    if "theta_degrees" in params:
        theta_degrees = int(params["theta_degrees"])
        if theta_degrees not in set(theta_values):
            raise ValueError(f"theta_degrees={theta_degrees} is outside supported angle values")
    else:
        theta_degrees = None
    if "radius_units" in params or "theta_degrees" in params:
        matches = [
            (index, values)
            for index, values in candidate_cases
            if ("radius_units" not in params or int(values.radius_units) == int(radius_units))
            and ("theta_degrees" not in params or int(values.theta_degrees) == int(theta_degrees))
        ]
        if not matches:
            raise ValueError("requested sector parameters are outside the filtered case pool")
        rng = spawn_rng(int(instance_seed), f"{seed_namespace}.case")
        index, values = uniform_choice(rng, tuple(matches))
    elif balance_key is not None:
        buckets: dict[float, list[tuple[int, SectorValues]]] = {}
        for index, case_values in candidate_cases:
            key = round1(float(balance_key(case_values)))
            buckets.setdefault(float(key), []).append((int(index), case_values))
        if not buckets:
            raise ValueError("sector balance pool is empty")
        rng = spawn_rng(int(instance_seed), f"{seed_namespace}.answer")
        selected_key = uniform_choice(rng, tuple(sorted(buckets)))
        case_rng = spawn_rng(int(instance_seed), f"{seed_namespace}.case")
        index, values = uniform_choice(case_rng, tuple(buckets[float(selected_key)]))
    else:
        rng = spawn_rng(int(instance_seed), f"{seed_namespace}.case")
        index, values = uniform_choice(rng, tuple(candidate_cases))
    support = geometry_selected_probability_map(
        theta_values,
        int(values.theta_degrees),
        is_selected=lambda value, selected: float(value) == float(selected),
    )
    return int(index), values, support


def sample_sector_area_from_complement_angle(
    instance_seed: int,
    *,
    seed_namespace: str,
    params: Mapping[str, Any],
) -> SectorProblem:
    """Sample a case where sector area follows from radius and a complementary relation."""

    case_index, values, _support = _sample_values(
        instance_seed=int(instance_seed),
        seed_namespace=str(seed_namespace),
        params=params,
        theta_support=COMPLEMENT_THETA_SUPPORT,
        target_angle_total=90,
    )
    return SectorProblem(
        answer=float(values.sector_area),
        values=values,
        formula_family="sector_area_from_complement_angle",
        visual_case="measure_from_complement",
        target_kind="sector_area",
        visible_measure_kind="complement_relation",
        reasoning_steps=2,
        case_index=int(case_index),
        layout_seed=int(instance_seed),
    )


def sample_arc_length_from_supplement_angle(
    instance_seed: int,
    *,
    seed_namespace: str,
    params: Mapping[str, Any],
) -> SectorProblem:
    """Sample a case where arc length follows from radius and a supplementary relation."""

    case_index, values, _support = _sample_values(
        instance_seed=int(instance_seed),
        seed_namespace=str(seed_namespace),
        params=params,
        theta_support=SUPPLEMENT_THETA_SUPPORT,
        target_angle_total=180,
    )
    return SectorProblem(
        answer=float(values.arc_length),
        values=values,
        formula_family="sector_arc_from_supplement_angle",
        visual_case="measure_from_supplement",
        target_kind="arc_length",
        visible_measure_kind="supplement_relation",
        reasoning_steps=2,
        case_index=int(case_index),
        layout_seed=int(instance_seed),
    )


def sample_sector_area_from_arc_length(
    instance_seed: int,
    *,
    seed_namespace: str,
    params: Mapping[str, Any],
) -> SectorProblem:
    """Sample a case where sector area follows from radius and arc length."""

    case_index, values, _support = _sample_values(
        instance_seed=int(instance_seed),
        seed_namespace=str(seed_namespace),
        params=params,
        theta_support=DIRECT_THETA_SUPPORT,
        value_filter=_area_from_arc_is_consistent,
    )
    return SectorProblem(
        answer=round1(0.5 * float(values.radius_units) * float(values.arc_length)),
        values=values,
        formula_family="sector_area_from_arc",
        visual_case="single_sector",
        target_kind="sector_area",
        visible_measure_kind="arc_length",
        reasoning_steps=2,
        case_index=int(case_index),
        layout_seed=int(instance_seed),
    )


def sample_arc_length_from_sector_area(
    instance_seed: int,
    *,
    seed_namespace: str,
    params: Mapping[str, Any],
) -> SectorProblem:
    """Sample a case where arc length follows from radius and sector area."""

    case_index, values, _support = _sample_values(
        instance_seed=int(instance_seed),
        seed_namespace=str(seed_namespace),
        params=params,
        theta_support=DIRECT_THETA_SUPPORT,
        value_filter=_arc_from_area_is_consistent,
    )
    return SectorProblem(
        answer=round1((2.0 * float(values.sector_area)) / float(values.radius_units)),
        values=values,
        formula_family="sector_arc_from_area",
        visual_case="single_sector",
        target_kind="arc_length",
        visible_measure_kind="sector_area",
        reasoning_steps=2,
        case_index=int(case_index),
        layout_seed=int(instance_seed),
    )


def sample_sector_angle_from_arc_length(
    instance_seed: int,
    *,
    seed_namespace: str,
    params: Mapping[str, Any],
) -> SectorProblem:
    """Sample a case where the central angle follows from radius and arc length."""

    case_index, values, _support = _sample_values(
        instance_seed=int(instance_seed),
        seed_namespace=str(seed_namespace),
        params=params,
        theta_support=DIRECT_THETA_SUPPORT,
        value_filter=_angle_from_arc_is_consistent,
    )
    return SectorProblem(
        answer=float(values.angle_from_arc_length),
        values=values,
        formula_family="sector_angle_from_arc",
        visual_case="single_sector",
        target_kind="sector_angle",
        visible_measure_kind="arc_length",
        reasoning_steps=1,
        case_index=int(case_index),
        layout_seed=int(instance_seed),
    )


def sample_sector_angle_from_area(
    instance_seed: int,
    *,
    seed_namespace: str,
    params: Mapping[str, Any],
) -> SectorProblem:
    """Sample a case where the central angle follows from radius and sector area."""

    case_index, values, _support = _sample_values(
        instance_seed=int(instance_seed),
        seed_namespace=str(seed_namespace),
        params=params,
        theta_support=DIRECT_THETA_SUPPORT,
        value_filter=_angle_from_area_is_consistent,
    )
    return SectorProblem(
        answer=float(values.angle_from_sector_area),
        values=values,
        formula_family="sector_angle_from_area",
        visual_case="single_sector",
        target_kind="sector_angle",
        visible_measure_kind="sector_area",
        reasoning_steps=1,
        case_index=int(case_index),
        layout_seed=int(instance_seed),
    )


def sample_complement_angle_from_arc_length(
    instance_seed: int,
    *,
    seed_namespace: str,
    params: Mapping[str, Any],
) -> SectorProblem:
    """Sample a related angle from a sector angle and complementary relation."""

    case_index, values, _support = _sample_values(
        instance_seed=int(instance_seed),
        seed_namespace=str(seed_namespace),
        params=params,
        theta_support=COMPLEMENT_THETA_SUPPORT,
        target_angle_total=90,
        value_filter=_angle_from_arc_is_consistent,
        balance_key=lambda values: 90.0 - float(values.angle_from_arc_length),
    )
    return SectorProblem(
        answer=round1(90.0 - float(values.angle_from_arc_length)),
        values=values,
        formula_family="sector_angle_then_complement",
        visual_case="adjacent_complement",
        target_kind="related_angle",
        visible_measure_kind="arc_length",
        reasoning_steps=2,
        case_index=int(case_index),
        layout_seed=int(instance_seed),
    )


def sample_supplement_angle_from_area(
    instance_seed: int,
    *,
    seed_namespace: str,
    params: Mapping[str, Any],
) -> SectorProblem:
    """Sample a related angle from a sector angle and supplementary relation."""

    case_index, values, _support = _sample_values(
        instance_seed=int(instance_seed),
        seed_namespace=str(seed_namespace),
        params=params,
        theta_support=SUPPLEMENT_THETA_SUPPORT,
        target_angle_total=180,
        value_filter=_angle_from_area_is_consistent,
    )
    return SectorProblem(
        answer=round1(180.0 - float(values.angle_from_sector_area)),
        values=values,
        formula_family="sector_angle_then_supplement",
        visual_case="adjacent_supplement",
        target_kind="related_angle",
        visible_measure_kind="sector_area",
        reasoning_steps=2,
        case_index=int(case_index),
        layout_seed=int(instance_seed),
    )


def sample_remaining_angle_from_sector_measure(
    instance_seed: int,
    *,
    seed_namespace: str,
    params: Mapping[str, Any],
) -> SectorProblem:
    """Sample a remaining-around-circle angle from a sector arc length."""

    case_index, values, _support = _sample_values(
        instance_seed=int(instance_seed),
        seed_namespace=str(seed_namespace),
        params=params,
        theta_support=REMAINING_THETA_SUPPORT,
        target_angle_total=360,
        value_filter=_angle_from_arc_is_consistent,
    )
    return SectorProblem(
        answer=round1(360.0 - float(values.angle_from_arc_length)),
        values=values,
        formula_family="sector_angle_then_remaining",
        visual_case="remaining_circle_angle",
        target_kind="related_angle",
        visible_measure_kind="arc_length",
        reasoning_steps=2,
        case_index=int(case_index),
        layout_seed=int(instance_seed),
    )


__all__ = [
    "PI_VALUE",
    "sample_arc_length_from_sector_area",
    "sample_arc_length_from_supplement_angle",
    "sample_complement_angle_from_arc_length",
    "sample_remaining_angle_from_sector_measure",
    "sample_sector_angle_from_arc_length",
    "sample_sector_angle_from_area",
    "sample_sector_area_from_arc_length",
    "sample_sector_area_from_complement_angle",
    "sample_supplement_angle_from_area",
]
