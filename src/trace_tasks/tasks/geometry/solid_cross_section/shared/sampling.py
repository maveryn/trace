"""Answer-first sampling helpers for solid cross-section cases."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable, Mapping, Sequence, TypeVar

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.measurement_rendering import fmt_measure
from trace_tasks.tasks.shared.fixed_query import geometry_selected_probability_map

from .measurements import (
    cone_parallel_slice_area,
    square_pyramid_parallel_slice_area,
    validate_cone_case,
    validate_pyramid_case,
)
from .state import ConeSliceCase, PyramidSliceCase

CaseT = TypeVar("CaseT", ConeSliceCase, PyramidSliceCase)

CLEAN_SIMILARITY_RATIOS: tuple[tuple[int, int], ...] = (
    (1, 3),
    (2, 5),
    (1, 2),
    (3, 5),
    (2, 3),
    (3, 4),
)
HEIGHT_MULTIPLIERS: tuple[int, ...] = (4, 5, 6, 7)

CONE_SLICE_CASES: tuple[ConeSliceCase, ...] = tuple(
    ConeSliceCase(int(base_radius), int(ratio_den * height_multiplier), int(ratio_num * height_multiplier))
    for base_radius in range(4, 17)
    for ratio_num, ratio_den in CLEAN_SIMILARITY_RATIOS
    for height_multiplier in HEIGHT_MULTIPLIERS
)
PYRAMID_SLICE_CASES: tuple[PyramidSliceCase, ...] = tuple(
    PyramidSliceCase(int(base_side), int(ratio_den * height_multiplier), int(ratio_num * height_multiplier))
    for base_side in range(6, 21)
    for ratio_num, ratio_den in CLEAN_SIMILARITY_RATIOS
    for height_multiplier in HEIGHT_MULTIPLIERS
)


def _answer_key(value: float) -> str:
    return fmt_measure(float(value))


def _sort_answer_key(value: str) -> float:
    return float(value)


@lru_cache(maxsize=None)
def _cone_cases_by_answer() -> dict[str, tuple[ConeSliceCase, ...]]:
    return _group_cases_by_answer(CONE_SLICE_CASES, cone_parallel_slice_area)


@lru_cache(maxsize=None)
def _pyramid_cases_by_answer() -> dict[str, tuple[PyramidSliceCase, ...]]:
    return _group_cases_by_answer(PYRAMID_SLICE_CASES, square_pyramid_parallel_slice_area)


def _group_cases_by_answer(cases: Sequence[CaseT], answer_fn: Callable[[CaseT], float]) -> dict[str, tuple[CaseT, ...]]:
    grouped: dict[str, list[CaseT]] = {}
    for case in cases:
        grouped.setdefault(_answer_key(float(answer_fn(case))), []).append(case)
    return {str(key): tuple(values) for key, values in grouped.items()}


def _resolve_answer_first_case(
    *,
    cases_by_answer: Mapping[str, tuple[CaseT, ...]],
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> tuple[CaseT, int, tuple[float, ...], int]:
    answer_keys = tuple(sorted(cases_by_answer, key=_sort_answer_key))
    if not answer_keys:
        raise ValueError("solid cross-section case support is empty")
    explicit_answer = params.get("target_answer")
    if explicit_answer is None:
        rng = spawn_rng(int(instance_seed), f"{namespace}.answer")
        answer_key = str(uniform_choice(rng, answer_keys))
    else:
        answer_key = _answer_key(float(explicit_answer))
        if answer_key not in cases_by_answer:
            raise ValueError(f"target_answer={explicit_answer} is not supported")
    answer_cases = tuple(cases_by_answer[str(answer_key)])
    rng = spawn_rng(int(instance_seed), f"{namespace}.case.{answer_key}")
    selected = uniform_choice(rng, answer_cases)
    case_index = answer_cases.index(selected)
    return selected, int(case_index), tuple(float(value) for value in answer_keys), len(answer_cases)


def _support_probabilities(support_values: Sequence[float], answer: float) -> dict[str, float]:
    return geometry_selected_probability_map(
        tuple(float(value) for value in support_values),
        float(answer),
        key_fn=fmt_measure,
        sort_unique=True,
    )


def resolve_cone_slice_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> tuple[ConeSliceCase, dict[str, float], int]:
    """Resolve one cone case by answer support, then construction."""

    has_dimension_override = any(key in params for key in ("base_radius", "solid_height", "slice_distance_from_apex"))
    if has_dimension_override:
        selected, _case_index, support_values, _case_count = _resolve_answer_first_case(
            cases_by_answer=_cone_cases_by_answer(),
            instance_seed=int(instance_seed),
            params=params,
            namespace=str(namespace),
        )
        case = ConeSliceCase(
            int(params.get("base_radius", selected.base_radius)),
            int(params.get("solid_height", selected.solid_height)),
            int(params.get("slice_distance_from_apex", selected.slice_distance_from_apex)),
        )
        validate_cone_case(case)
        answer = cone_parallel_slice_area(case)
        return case, _support_probabilities(support_values, answer), 1
    selected, _case_index, support_values, case_count = _resolve_answer_first_case(
        cases_by_answer=_cone_cases_by_answer(),
        instance_seed=int(instance_seed),
        params=params,
        namespace=str(namespace),
    )
    return selected, _support_probabilities(support_values, cone_parallel_slice_area(selected)), int(case_count)


def resolve_pyramid_slice_case(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> tuple[PyramidSliceCase, dict[str, float], int]:
    """Resolve one square-pyramid case by answer support, then construction."""

    has_dimension_override = any(key in params for key in ("base_side", "solid_height", "slice_distance_from_apex"))
    if has_dimension_override:
        selected, _case_index, support_values, _case_count = _resolve_answer_first_case(
            cases_by_answer=_pyramid_cases_by_answer(),
            instance_seed=int(instance_seed),
            params=params,
            namespace=str(namespace),
        )
        case = PyramidSliceCase(
            int(params.get("base_side", selected.base_side)),
            int(params.get("solid_height", selected.solid_height)),
            int(params.get("slice_distance_from_apex", selected.slice_distance_from_apex)),
        )
        validate_pyramid_case(case)
        answer = square_pyramid_parallel_slice_area(case)
        return case, _support_probabilities(support_values, answer), 1
    selected, _case_index, support_values, case_count = _resolve_answer_first_case(
        cases_by_answer=_pyramid_cases_by_answer(),
        instance_seed=int(instance_seed),
        params=params,
        namespace=str(namespace),
    )
    return selected, _support_probabilities(support_values, square_pyramid_parallel_slice_area(selected)), int(case_count)


def cone_answer_support_size() -> int:
    """Return the number of unique cone-slice answers."""

    return len(_cone_cases_by_answer())


def pyramid_answer_support_size() -> int:
    """Return the number of unique pyramid-slice answers."""

    return len(_pyramid_cases_by_answer())


__all__ = [
    "cone_answer_support_size",
    "CONE_SLICE_CASES",
    "PYRAMID_SLICE_CASES",
    "pyramid_answer_support_size",
    "resolve_cone_slice_case",
    "resolve_pyramid_slice_case",
]
