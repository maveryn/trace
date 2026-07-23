"""Deterministic case-pool helpers for solid-revolution tasks."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.pythagorean import integer_right_triangles

from .measurements import round_volume, volume_cone, volume_cylinder, volume_double_cone, volume_frustum

T = TypeVar("T")


@dataclass(frozen=True)
class CylinderCase:
    """One cylinder revolution construction."""

    radial_input_kind: str
    diameter: int
    height: int
    diagonal: int | None
    answer: float


@dataclass(frozen=True)
class ConeCase:
    """One cone revolution construction."""

    radius: int
    height: int
    slant_height: int
    answer: float


@dataclass(frozen=True)
class DoubleConeCase:
    """One double-cone revolution construction."""

    radius: int
    half_height: int
    answer: float


@dataclass(frozen=True)
class FrustumCase:
    """One frustum revolution construction."""

    top_radius: int
    bottom_radius: int
    height: int
    answer: float


def _dedupe_by_answer(cases: Iterable[T], *, key: Callable[[T], float]) -> tuple[T, ...]:
    seen: set[float] = set()
    resolved: list[T] = []
    for case in cases:
        answer = round_volume(float(key(case)))
        if answer in seen:
            continue
        seen.add(answer)
        resolved.append(case)
    return tuple(resolved)


def _require_support(cases: Sequence[T], *, minimum: int, label: str) -> tuple[T, ...]:
    if len(tuple(cases)) < int(minimum):
        raise ValueError(f"{label} case pool must contain at least {minimum} unique answers")
    return tuple(cases)


def _direct_cylinder_cases() -> tuple[CylinderCase, ...]:
    return tuple(
        CylinderCase(
            radial_input_kind="diameter",
            diameter=int(diameter),
            height=int(height),
            diagonal=None,
            answer=round_volume(volume_cylinder(diameter=diameter, height=height)),
        )
        for diameter in range(4, 37, 2)
        for height in range(3, 33)
    )


def _diagonal_cylinder_cases() -> tuple[CylinderCase, ...]:
    return tuple(
        CylinderCase(
            radial_input_kind="diagonal",
            diameter=int(triangle.leg_a),
            height=int(triangle.leg_b),
            diagonal=int(triangle.hypotenuse),
            answer=round_volume(
                volume_cylinder(diameter=triangle.leg_a, height=triangle.leg_b)
            ),
        )
        for triangle in integer_right_triangles(
            min_leg=4,
            max_leg=80,
            max_hypotenuse=180,
            include_swapped=True,
        )
        if int(triangle.leg_a) % 2 == 0 and int(triangle.leg_b) >= 3
    )


@lru_cache(maxsize=1)
def cylinder_direct_case_pool() -> tuple[CylinderCase, ...]:
    """Return direct-diameter cylinder cases with unique rounded volume answers."""

    cases = sorted(
        _dedupe_by_answer(_direct_cylinder_cases(), key=lambda case: case.answer),
        key=lambda case: (float(case.answer), int(case.diameter), int(case.height)),
    )
    return _require_support(cases, minimum=50, label="direct cylinder")


@lru_cache(maxsize=1)
def cylinder_diagonal_case_pool() -> tuple[CylinderCase, ...]:
    """Return diagonal-derived cylinder cases with unique rounded volume answers."""

    cases = sorted(
        _dedupe_by_answer(_diagonal_cylinder_cases(), key=lambda case: case.answer),
        key=lambda case: (
            float(case.answer),
            int(case.diagonal or 0),
            int(case.diameter),
            int(case.height),
        ),
    )
    return _require_support(cases, minimum=50, label="diagonal cylinder")


@lru_cache(maxsize=1)
def cylinder_case_pool() -> tuple[CylinderCase, ...]:
    """Return all cylinder cases with unique rounded volume answers."""

    cases = sorted(
        _dedupe_by_answer((*_direct_cylinder_cases(), *_diagonal_cylinder_cases()), key=lambda case: case.answer),
        key=lambda case: (
            float(case.answer),
            str(case.radial_input_kind),
            int(case.diameter),
            int(case.height),
        ),
    )
    return _require_support(cases, minimum=50, label="cylinder")


@lru_cache(maxsize=1)
def cone_case_pool() -> tuple[ConeCase, ...]:
    """Return cone cases with unique rounded volume answers."""

    raw_cases = (
        ConeCase(
            radius=int(triangle.leg_a),
            height=int(triangle.leg_b),
            slant_height=int(triangle.hypotenuse),
            answer=round_volume(volume_cone(radius=triangle.leg_a, height=triangle.leg_b)),
        )
        for triangle in integer_right_triangles(
            min_leg=3,
            max_leg=60,
            max_hypotenuse=130,
            include_swapped=True,
        )
        if int(triangle.leg_a) >= 3 and int(triangle.leg_b) >= 4
    )
    cases = sorted(
        _dedupe_by_answer(raw_cases, key=lambda case: case.answer),
        key=lambda case: (float(case.answer), int(case.radius), int(case.height)),
    )
    return _require_support(cases, minimum=50, label="cone")


@lru_cache(maxsize=1)
def double_cone_case_pool() -> tuple[DoubleConeCase, ...]:
    """Return double-cone cases with unique rounded volume answers."""

    raw_cases = (
        DoubleConeCase(
            radius=int(radius),
            half_height=int(half_height),
            answer=round_volume(volume_double_cone(radius=radius, half_height=half_height)),
        )
        for radius in range(3, 31)
        for half_height in range(3, 31)
    )
    cases = sorted(
        _dedupe_by_answer(raw_cases, key=lambda case: case.answer),
        key=lambda case: (float(case.answer), int(case.radius), int(case.half_height)),
    )
    return _require_support(cases, minimum=50, label="double_cone")


@lru_cache(maxsize=1)
def frustum_case_pool() -> tuple[FrustumCase, ...]:
    """Return frustum cases with unique rounded volume answers."""

    raw_cases = (
        FrustumCase(
            top_radius=int(top_radius),
            bottom_radius=int(bottom_radius),
            height=int(height),
            answer=round_volume(
                volume_frustum(
                    top_radius=top_radius,
                    bottom_radius=bottom_radius,
                    height=height,
                )
            ),
        )
        for top_radius in range(2, 17)
        for bottom_radius in range(top_radius + 2, 34)
        for height in range(3, 29)
    )
    cases = sorted(
        _dedupe_by_answer(raw_cases, key=lambda case: case.answer),
        key=lambda case: (
            float(case.answer),
            int(case.top_radius),
            int(case.bottom_radius),
            int(case.height),
        ),
    )
    return _require_support(cases, minimum=50, label="frustum")


def select_case_from_pool(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    cases: Sequence[T],
) -> T:
    """Select one construction from a task-owned case pool."""

    values = tuple(cases)
    if not values:
        raise ValueError("case pool must be non-empty")
    rng = spawn_rng(int(instance_seed), str(namespace))
    return uniform_choice(rng, values)


def support_from_cases(cases: Sequence[Any]) -> tuple[float, ...]:
    """Return the answer support represented by a unique-answer case pool."""

    return tuple(float(case.answer) for case in tuple(cases))


__all__ = [
    "ConeCase",
    "CylinderCase",
    "DoubleConeCase",
    "FrustumCase",
    "cone_case_pool",
    "cylinder_case_pool",
    "cylinder_diagonal_case_pool",
    "cylinder_direct_case_pool",
    "double_cone_case_pool",
    "frustum_case_pool",
    "select_case_from_pool",
    "support_from_cases",
]
