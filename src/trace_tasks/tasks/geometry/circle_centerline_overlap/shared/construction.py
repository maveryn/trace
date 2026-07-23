"""Construction primitives for circle-centerline-overlap diagrams."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import integer_range_choice, uniform_choice, weighted_support_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.fixed_query import geometry_selected_probability_map

from .state import (
    BOUNDARY_PAIRS,
    BOUNDARY_TARGET_ROLES,
    DEFAULT_CIRCLE_COUNT_WEIGHTS,
    LABEL_MODES,
    SUPPORTED_CIRCLE_COUNTS,
    CircleOverlapCase,
)

RADIUS_A_RANGE: tuple[int, int] = (8, 18)
RADIUS_B_RANGE: tuple[int, int] = (12, 28)
RADIUS_C_RANGE: tuple[int, int] = (8, 18)
OVERLAP_RANGE: tuple[int, int] = (2, 10)


def _select_int_inclusive(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
    low: int,
    high: int,
) -> int:
    """Select one deterministic integer from an inclusive range."""

    low_int = int(low)
    high_int = int(high)
    if high_int < low_int:
        raise ValueError(f"invalid integer range: {low_int}..{high_int}")
    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, _probabilities = integer_range_choice(rng, low_int, high_int)
    return int(selected)


def _max_overlap_for_pair(left_radius: int, right_radius: int) -> int:
    """Return the largest overlap that preserves clear positive boundary segments."""

    return min(int(OVERLAP_RANGE[1]), int(left_radius) - 3, int(right_radius) - 3)


def _sample_overlap_for_pair(
    *,
    left_radius: int,
    right_radius: int,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> int:
    """Sample one valid adjacent overlap for a selected radius pair."""

    high = _max_overlap_for_pair(int(left_radius), int(right_radius))
    return _select_int_inclusive(
        instance_seed=int(instance_seed),
        params=params,
        namespace=str(namespace),
        low=int(OVERLAP_RANGE[0]),
        high=int(high),
    )


def _sample_generated_overlap_case(
    *,
    circle_count: int,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> CircleOverlapCase:
    """Sample a broad deterministic valid overlap case instead of a tiny bank."""

    radius_a = _select_int_inclusive(
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{namespace}.radius_a",
        low=RADIUS_A_RANGE[0],
        high=RADIUS_A_RANGE[1],
    )
    radius_b = _select_int_inclusive(
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{namespace}.radius_b",
        low=RADIUS_B_RANGE[0],
        high=RADIUS_B_RANGE[1],
    )
    overlap_ab = _sample_overlap_for_pair(
        left_radius=radius_a,
        right_radius=radius_b,
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{namespace}.overlap_ab",
    )
    if int(circle_count) == 2:
        case = CircleOverlapCase(
            radius_a=int(radius_a),
            radius_b=int(radius_b),
            radius_c=0,
            overlap_ab=int(overlap_ab),
            overlap_bc=0,
        )
        validate_overlap_case(case)
        return case

    radius_c = _select_int_inclusive(
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{namespace}.radius_c",
        low=RADIUS_C_RANGE[0],
        high=RADIUS_C_RANGE[1],
    )
    overlap_bc = _sample_overlap_for_pair(
        left_radius=radius_b,
        right_radius=radius_c,
        instance_seed=int(instance_seed),
        params=params,
        namespace=f"{namespace}.overlap_bc",
    )
    case = CircleOverlapCase(
        radius_a=int(radius_a),
        radius_b=int(radius_b),
        radius_c=int(radius_c),
        overlap_ab=int(overlap_ab),
        overlap_bc=int(overlap_bc),
    )
    validate_overlap_case(case)
    return case


@lru_cache(maxsize=1)
def _all_generated_overlap_cases() -> tuple[CircleOverlapCase, ...]:
    """Return the finite support induced by the broad range sampler."""

    cases: list[CircleOverlapCase] = []
    for radius_a in range(RADIUS_A_RANGE[0], RADIUS_A_RANGE[1] + 1):
        for radius_b in range(RADIUS_B_RANGE[0], RADIUS_B_RANGE[1] + 1):
            max_overlap_ab = _max_overlap_for_pair(radius_a, radius_b)
            for overlap_ab in range(OVERLAP_RANGE[0], max_overlap_ab + 1):
                case = CircleOverlapCase(
                    radius_a=int(radius_a),
                    radius_b=int(radius_b),
                    radius_c=0,
                    overlap_ab=int(overlap_ab),
                    overlap_bc=0,
                )
                validate_overlap_case(case)
                cases.append(case)
            for radius_c in range(RADIUS_C_RANGE[0], RADIUS_C_RANGE[1] + 1):
                max_overlap_bc = _max_overlap_for_pair(radius_b, radius_c)
                for overlap_ab in range(OVERLAP_RANGE[0], max_overlap_ab + 1):
                    for overlap_bc in range(OVERLAP_RANGE[0], max_overlap_bc + 1):
                        case = CircleOverlapCase(
                            radius_a=int(radius_a),
                            radius_b=int(radius_b),
                            radius_c=int(radius_c),
                            overlap_ab=int(overlap_ab),
                            overlap_bc=int(overlap_bc),
                        )
                        validate_overlap_case(case)
                        cases.append(case)
    return tuple(cases)


def generated_overlap_cases(circle_count: int | None = None) -> tuple[CircleOverlapCase, ...]:
    """Return generated cases, optionally filtered by circle count."""

    cases = _all_generated_overlap_cases()
    if circle_count is None:
        return cases
    normalized = int(circle_count)
    if normalized not in SUPPORTED_CIRCLE_COUNTS:
        raise ValueError(f"circle_count must be one of {SUPPORTED_CIRCLE_COUNTS}")
    return tuple(case for case in cases if int(case.circle_count) == normalized)


def segment_length(case: CircleOverlapCase, pair: str, role: str) -> int:
    """Return one boundary-to-center or center-to-boundary segment length."""

    if str(pair) == "AB":
        left_radius, right_radius, distance = int(case.radius_a), int(case.radius_b), int(case.distance_ab)
    elif str(pair) == "BC":
        left_radius, right_radius, distance = int(case.radius_b), int(case.radius_c), int(case.distance_bc)
    else:
        raise ValueError(f"unsupported boundary pair: {pair}")
    if str(role) == "left_center_to_right_boundary":
        return int(distance) - int(right_radius)
    if str(role) == "left_boundary_to_right_center":
        return int(distance) - int(left_radius)
    raise ValueError(f"unsupported boundary target role: {role}")


def validate_overlap_case(case: CircleOverlapCase) -> None:
    """Reject invalid radii/overlap combinations before rendering."""

    radii = (int(case.radius_a), int(case.radius_b))
    if int(case.circle_count) == 3:
        radii = (*radii, int(case.radius_c))
    if min(radii) <= 0:
        raise ValueError("circle radii must be positive")
    if int(case.overlap_ab) <= 0:
        raise ValueError("adjacent overlaps must be positive")
    d_ab = int(case.distance_ab)
    if not (abs(case.radius_a - case.radius_b) + 1 < d_ab < case.radius_a + case.radius_b):
        raise ValueError("AB must be a proper adjacent overlap without containment")
    valid_pairs = ("AB",)
    if int(case.circle_count) == 3:
        if int(case.overlap_bc) <= 0:
            raise ValueError("adjacent overlaps must be positive")
        d_bc = int(case.distance_bc)
        d_ac = int(case.distance_ac)
        if not (abs(case.radius_b - case.radius_c) + 1 < d_bc < case.radius_b + case.radius_c):
            raise ValueError("BC must be a proper adjacent overlap without containment")
        if d_ac <= int(case.radius_a) + int(case.radius_c) + 1:
            raise ValueError("non-adjacent circles A and C must not overlap")
        valid_pairs = BOUNDARY_PAIRS
    for pair in valid_pairs:
        for role in BOUNDARY_TARGET_ROLES:
            if segment_length(case, pair, role) < 3:
                raise ValueError("boundary segment answers must be at least 3")


def center_distance_length(case: CircleOverlapCase) -> int:
    """Return the requested full center-to-center distance for the active chain."""

    return int(case.distance_ab if int(case.circle_count) == 2 else case.distance_ac)


def select_circle_count(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[int, dict[str, float]]:
    """Select whether this sample uses two or three adjacent overlapping circles."""

    explicit = params.get("circle_count")
    if explicit is not None:
        value = int(explicit)
        if value not in SUPPORTED_CIRCLE_COUNTS:
            raise ValueError(f"circle_count must be one of {SUPPORTED_CIRCLE_COUNTS}")
        return value, geometry_selected_probability_map(SUPPORTED_CIRCLE_COUNTS, selected=value)
    explicit_case = params.get("overlap_case")
    if explicit_case is not None:
        if not isinstance(explicit_case, Sequence) or isinstance(explicit_case, (str, bytes)) or len(explicit_case) not in {3, 5}:
            raise ValueError("overlap_case must be [radius_a, radius_b, overlap_ab] or [radius_a, radius_b, radius_c, overlap_ab, overlap_bc]")
        value = 2 if len(explicit_case) == 3 else 3
        return value, geometry_selected_probability_map(SUPPORTED_CIRCLE_COUNTS, selected=value)
    if str(params.get("boundary_pair", "")) == "BC":
        return 3, geometry_selected_probability_map(SUPPORTED_CIRCLE_COUNTS, selected=3)
    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = weighted_support_choice(
        rng,
        SUPPORTED_CIRCLE_COUNTS,
        weights=DEFAULT_CIRCLE_COUNT_WEIGHTS,
        sort_keys=True,
    )
    return int(selected), {str(key): float(value) for key, value in probabilities.items()}


def select_overlap_case(
    *,
    circle_count: int | None = None,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> tuple[CircleOverlapCase, dict[str, float]]:
    """Select or validate one deterministic overlap case."""

    explicit = params.get("overlap_case")
    if explicit is not None:
        if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)) or len(explicit) not in {3, 5}:
            raise ValueError("overlap_case must be [radius_a, radius_b, overlap_ab] or [radius_a, radius_b, radius_c, overlap_ab, overlap_bc]")
        if len(explicit) == 3:
            case = CircleOverlapCase(int(explicit[0]), int(explicit[1]), 0, int(explicit[2]), 0)
        else:
            case = CircleOverlapCase(*(int(value) for value in explicit))
        validate_overlap_case(case)
        if circle_count is not None and int(case.circle_count) != int(circle_count):
            raise ValueError("overlap_case circle count does not match circle_count")
        return case, {case.key: 1.0}
    resolved_count = int(circle_count) if circle_count is not None else 3
    case = _sample_generated_overlap_case(
        circle_count=resolved_count,
        instance_seed=int(instance_seed),
        params=params,
        namespace=str(namespace),
    )
    return case, {case.key: 1.0}


def _select_case_by_answer(
    answer_cases: Mapping[int, Sequence[CircleOverlapCase]],
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> tuple[CircleOverlapCase, dict[str, float]]:
    """Select an overlap case by final integer answer first."""

    answer_values = tuple(sorted(int(value) for value in answer_cases))
    if not answer_values:
        raise ValueError("overlap answer support must not be empty")
    explicit_answer = params.get("target_answer")
    if explicit_answer is not None:
        answer = int(explicit_answer)
        if answer not in answer_cases:
            raise ValueError(f"target_answer={answer} is not supported")
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.answer")
        answer = int(uniform_choice(rng, answer_values))
    cases = tuple(answer_cases[int(answer)])
    rng = spawn_rng(int(instance_seed), f"{namespace}.case.{answer}")
    case = uniform_choice(rng, cases)
    return case, {case.key: 1.0}


def select_center_distance_overlap_case(
    *,
    circle_count: int,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> tuple[CircleOverlapCase, dict[str, float]]:
    """Select an overlap case by final center-to-center distance."""

    if params.get("overlap_case") is not None:
        return select_overlap_case(
            circle_count=int(circle_count),
            instance_seed=int(instance_seed),
            params=params,
            namespace=str(namespace),
        )
    grouped: dict[int, list[CircleOverlapCase]] = {}
    for case in generated_overlap_cases(int(circle_count)):
        grouped.setdefault(int(center_distance_length(case)), []).append(case)
    return _select_case_by_answer(
        grouped,
        instance_seed=int(instance_seed),
        params=params,
        namespace=str(namespace),
    )


def select_boundary_segment_overlap_case(
    *,
    boundary_pair: str,
    boundary_target_role: str,
    circle_count: int,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> tuple[CircleOverlapCase, dict[str, float]]:
    """Select an overlap case by final boundary segment length."""

    if params.get("overlap_case") is not None:
        return select_overlap_case(
            circle_count=int(circle_count),
            instance_seed=int(instance_seed),
            params=params,
            namespace=str(namespace),
        )
    pair = str(boundary_pair)
    role = str(boundary_target_role)
    grouped: dict[int, list[CircleOverlapCase]] = {}
    for case in generated_overlap_cases(int(circle_count)):
        grouped.setdefault(int(segment_length(case, pair, role)), []).append(case)
    return _select_case_by_answer(
        grouped,
        instance_seed=int(instance_seed),
        params=params,
        namespace=str(namespace),
    )


def select_label_mode(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Select the radius/diameter readout style for circle labels."""

    explicit = params.get("label_mode")
    if explicit is not None:
        value = str(explicit)
        if value not in LABEL_MODES:
            raise ValueError(f"label_mode must be one of {LABEL_MODES}")
        return value, geometry_selected_probability_map(LABEL_MODES, selected=value)
    value = LABEL_MODES[0]
    return str(value), geometry_selected_probability_map(LABEL_MODES)


def select_boundary_pair(
    *,
    circle_count: int,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Select the adjacent circle pair used by a boundary segment."""

    support = BOUNDARY_PAIRS if int(circle_count) == 3 else ("AB",)
    explicit = params.get("boundary_pair")
    if explicit is not None:
        value = str(explicit)
        if value not in support:
            raise ValueError(f"boundary_pair must be one of {support}")
        return value, geometry_selected_probability_map(support, selected=value)
    rng = spawn_rng(int(instance_seed), str(namespace))
    value = uniform_choice(rng, support)
    return str(value), geometry_selected_probability_map(support)


def select_boundary_target_role(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float]]:
    """Select which half-overlap centerline segment is unknown."""

    explicit = params.get("boundary_target_role")
    if explicit is not None:
        value = str(explicit)
        if value not in BOUNDARY_TARGET_ROLES:
            raise ValueError(f"boundary_target_role must be one of {BOUNDARY_TARGET_ROLES}")
        return value, geometry_selected_probability_map(BOUNDARY_TARGET_ROLES, selected=value)
    rng = spawn_rng(int(instance_seed), str(namespace))
    value = uniform_choice(rng, BOUNDARY_TARGET_ROLES)
    return str(value), geometry_selected_probability_map(BOUNDARY_TARGET_ROLES)


def boundary_names(pair: str, role: str) -> tuple[str, str, tuple[str, str], tuple[str, str]]:
    """Return target and known segment labels for one adjacent boundary pair."""

    if str(pair) == "AB":
        left_center, right_center = "A", "B"
        left_boundary, right_boundary = "P", "Q"
    elif str(pair) == "BC":
        left_center, right_center = "B", "C"
        left_boundary, right_boundary = "R", "S"
    else:
        raise ValueError(f"unsupported boundary pair: {pair}")
    if str(role) == "left_center_to_right_boundary":
        target_name = f"{left_center}{left_boundary}"
        known_name = f"{right_boundary}{right_center}"
        target_points = (left_center, left_boundary)
        known_points = (right_boundary, right_center)
    elif str(role) == "left_boundary_to_right_center":
        target_name = f"{right_boundary}{right_center}"
        known_name = f"{left_center}{left_boundary}"
        target_points = (right_boundary, right_center)
        known_points = (left_center, left_boundary)
    else:
        raise ValueError(f"unsupported boundary target role: {role}")
    return target_name, known_name, target_points, known_points


def center_distance_answer_support(selected: int) -> dict[str, float]:
    """Return support probabilities for possible full centerline distances."""

    support = tuple(sorted({int(center_distance_length(case)) for case in generated_overlap_cases()} | {int(selected)}))
    return geometry_selected_probability_map(support, selected=int(selected))


def boundary_segment_answer_support(selected: int) -> dict[str, float]:
    """Return support probabilities for possible boundary segment lengths."""

    support = tuple(
        sorted(
            {
                segment_length(case, pair, role)
                for case in generated_overlap_cases()
                for pair in BOUNDARY_PAIRS
                for role in BOUNDARY_TARGET_ROLES
                if not (int(case.circle_count) == 2 and str(pair) == "BC")
            }
            | {int(selected)}
        )
    )
    return geometry_selected_probability_map(support, selected=int(selected))


__all__ = [
    "boundary_names",
    "boundary_segment_answer_support",
    "center_distance_length",
    "center_distance_answer_support",
    "generated_overlap_cases",
    "segment_length",
    "select_boundary_pair",
    "select_boundary_segment_overlap_case",
    "select_boundary_target_role",
    "select_circle_count",
    "select_center_distance_overlap_case",
    "select_label_mode",
    "select_overlap_case",
    "validate_overlap_case",
]
