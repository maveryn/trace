"""Container volume formulas and validation."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, Sequence

from .state import ResolvedProblem

FILL_COUNT_MIN = 2
FILL_COUNT_MAX = 60


def case_probability_map(case_keys: Sequence[str], selected: str) -> Dict[str, float]:
    return {key: (1.0 if key == str(selected) else 0.0) for key in case_keys}


def cone_source_volume(source_base_area: int, source_height: int) -> int:
    numerator = int(source_base_area) * int(source_height)
    if numerator % 3 != 0:
        raise ValueError("cone source base_area * height must be divisible by 3")
    return int(numerator // 3)


def cylinder_source_volume(source_base_area: int, source_height: int) -> int:
    return int(source_base_area) * int(source_height)


def target_cylinder_volume(target_base_area: int, target_height: int) -> int:
    return int(target_base_area) * int(target_height)


def target_cuboid_volume(target_length: int, target_width: int, target_height: int) -> int:
    return int(target_length) * int(target_width) * int(target_height)


def round1(value: float) -> float:
    return round(float(value) + 1e-9, 1)


def json_answer_value(value: int | float) -> int | float:
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return int(round(number))
    return round1(number)


def fmt_number(value: int | float) -> str:
    number = float(value)
    if abs(number - round(number)) < 1e-9:
        return str(int(round(number)))
    return f"{round1(number):.1f}"


def uniform_answer_support(values: Sequence[int | float]) -> Dict[str, float]:
    support = sorted({json_answer_value(value) for value in values})
    probability = 1.0 / float(max(1, len(support)))
    return {str(value): probability for value in support}


def support_from_cases(cases: Sequence[Sequence[int]], resolver: Callable[[Sequence[int]], ResolvedProblem], field_name: str) -> Dict[str, float]:
    return uniform_answer_support([getattr(resolver(case), str(field_name)) for case in cases])


def validate_fill_count(source_volume: int, target_volume: int) -> int:
    if int(source_volume) <= 0 or int(target_volume) <= 0:
        raise ValueError("source and target volumes must be positive")
    if int(target_volume) % int(source_volume) != 0:
        raise ValueError("target volume must be an exact multiple of source volume")
    fill_count = int(target_volume) // int(source_volume)
    if not (FILL_COUNT_MIN <= int(fill_count) <= FILL_COUNT_MAX):
        raise ValueError(f"fill_count must be in the v1 support range {FILL_COUNT_MIN}..{FILL_COUNT_MAX}")
    return int(fill_count)


def validate_resulting_height(source_volume: int, target_base_area: int, target_height: int, pour_count: int) -> float:
    if min(int(source_volume), int(target_base_area), int(target_height), int(pour_count)) <= 0:
        raise ValueError("resulting-height operands must be positive")
    result = float(int(source_volume) * int(pour_count)) / float(int(target_base_area))
    if not (1.5 <= result <= float(target_height) * 0.9):
        raise ValueError("resulting height must be visible and below target capacity")
    return round1(result)


def validate_cone_fill_case(case: Sequence[int]) -> None:
    source_base_area, source_height, target_base_area, target_height = [int(value) for value in case]
    if min(source_base_area, source_height, target_base_area, target_height) <= 0:
        raise ValueError("container dimensions must be positive")
    validate_fill_count(
        cone_source_volume(source_base_area, source_height),
        target_cylinder_volume(target_base_area, target_height),
    )


def validate_cylinder_fill_case(case: Sequence[int]) -> None:
    source_base_area, source_height, target_length, target_width, target_height = [int(value) for value in case]
    if min(source_base_area, source_height, target_length, target_width, target_height) <= 0:
        raise ValueError("container dimensions must be positive")
    validate_fill_count(
        cylinder_source_volume(source_base_area, source_height),
        target_cuboid_volume(target_length, target_width, target_height),
    )


def validate_cone_height_case(case: Sequence[int]) -> None:
    source_base_area, source_height, target_base_area, target_height, pour_count = [int(value) for value in case]
    if min(source_base_area, source_height, target_base_area, target_height, pour_count) <= 0:
        raise ValueError("container dimensions and pour count must be positive")
    validate_resulting_height(
        cone_source_volume(source_base_area, source_height),
        target_base_area,
        target_height,
        pour_count,
    )


def validate_cylinder_height_case(case: Sequence[int]) -> None:
    source_base_area, source_height, target_length, target_width, target_height, pour_count = [int(value) for value in case]
    if min(source_base_area, source_height, target_length, target_width, target_height, pour_count) <= 0:
        raise ValueError("container dimensions and pour count must be positive")
    validate_resulting_height(
        cylinder_source_volume(source_base_area, source_height),
        int(target_length) * int(target_width),
        target_height,
        pour_count,
    )


def resolve_cone_fill_count(case: Sequence[int]) -> ResolvedProblem:
    validate_cone_fill_case(case)
    source_base_area, source_height, target_base_area, target_height = [int(value) for value in case]
    source_volume = cone_source_volume(source_base_area, source_height)
    target_volume = target_cylinder_volume(target_base_area, target_height)
    fill_count = validate_fill_count(source_volume, target_volume)
    return ResolvedProblem(
        objective="fill_count",
        diagram_mode="fill_target",
        source_shape="cone",
        target_shape="cylinder",
        source_base_area=source_base_area,
        source_height=source_height,
        source_volume=source_volume,
        target_base_area=target_base_area,
        target_height=target_height,
        target_length=0,
        target_width=0,
        target_volume=target_volume,
        fill_count=fill_count,
        pour_count=fill_count,
        resulting_height=float(target_height),
        answer=fill_count,
        formula_family="container_volume_transfer_fill_count",
        formula="fill_count = target_volume / source_volume; cone_volume = base_area*height/3; cylinder_volume = base_area*height",
        query_probabilities={},
        case_probabilities={},
        answer_support_probabilities={},
    )


def resolve_cylinder_fill_count(case: Sequence[int]) -> ResolvedProblem:
    validate_cylinder_fill_case(case)
    source_base_area, source_height, target_length, target_width, target_height = [int(value) for value in case]
    source_volume = cylinder_source_volume(source_base_area, source_height)
    target_volume = target_cuboid_volume(target_length, target_width, target_height)
    fill_count = validate_fill_count(source_volume, target_volume)
    return ResolvedProblem(
        objective="fill_count",
        diagram_mode="fill_target",
        source_shape="cylinder",
        target_shape="cuboid",
        source_base_area=source_base_area,
        source_height=source_height,
        source_volume=source_volume,
        target_base_area=0,
        target_height=target_height,
        target_length=target_length,
        target_width=target_width,
        target_volume=target_volume,
        fill_count=fill_count,
        pour_count=fill_count,
        resulting_height=float(target_height),
        answer=fill_count,
        formula_family="container_volume_transfer_fill_count",
        formula="fill_count = target_volume / source_volume; cylinder_volume = base_area*height; cuboid_volume = length*width*height",
        query_probabilities={},
        case_probabilities={},
        answer_support_probabilities={},
    )


def resolve_cone_resulting_height(case: Sequence[int]) -> ResolvedProblem:
    validate_cone_height_case(case)
    source_base_area, source_height, target_base_area, target_height, pour_count = [int(value) for value in case]
    source_volume = cone_source_volume(source_base_area, source_height)
    target_volume = target_cylinder_volume(target_base_area, target_height)
    resulting_height = validate_resulting_height(source_volume, target_base_area, target_height, pour_count)
    return ResolvedProblem(
        objective="resulting_height",
        diagram_mode="resulting_fill_height",
        source_shape="cone",
        target_shape="cylinder",
        source_base_area=source_base_area,
        source_height=source_height,
        source_volume=source_volume,
        target_base_area=target_base_area,
        target_height=target_height,
        target_length=0,
        target_width=0,
        target_volume=target_volume,
        fill_count=0,
        pour_count=pour_count,
        resulting_height=float(resulting_height),
        answer=json_answer_value(resulting_height),
        formula_family="container_volume_transfer_resulting_height",
        formula="resulting_height = pour_count * source_volume / target_base_area; cone_volume = base_area*height/3",
        query_probabilities={},
        case_probabilities={},
        answer_support_probabilities={},
    )


def resolve_cylinder_resulting_height(case: Sequence[int]) -> ResolvedProblem:
    validate_cylinder_height_case(case)
    source_base_area, source_height, target_length, target_width, target_height, pour_count = [int(value) for value in case]
    source_volume = cylinder_source_volume(source_base_area, source_height)
    target_base_area = int(target_length) * int(target_width)
    target_volume = target_cuboid_volume(target_length, target_width, target_height)
    resulting_height = validate_resulting_height(source_volume, target_base_area, target_height, pour_count)
    return ResolvedProblem(
        objective="resulting_height",
        diagram_mode="resulting_fill_height",
        source_shape="cylinder",
        target_shape="cuboid",
        source_base_area=source_base_area,
        source_height=source_height,
        source_volume=source_volume,
        target_base_area=target_base_area,
        target_height=target_height,
        target_length=target_length,
        target_width=target_width,
        target_volume=target_volume,
        fill_count=0,
        pour_count=pour_count,
        resulting_height=float(resulting_height),
        answer=json_answer_value(resulting_height),
        formula_family="container_volume_transfer_resulting_height",
        formula="resulting_height = pour_count * source_volume / (target_length * target_width); cylinder_volume = base_area*height",
        query_probabilities={},
        case_probabilities={},
        answer_support_probabilities={},
    )


def validate_explicit_pour_params(problem: ResolvedProblem, params: dict[str, Any]) -> None:
    explicit_answer = params.get("fill_count")
    if explicit_answer is not None and int(explicit_answer) != int(problem.fill_count):
        raise ValueError("fill_count must equal target_volume / source_volume")
    explicit_pour_count = params.get("pour_count")
    if explicit_pour_count is not None and int(explicit_pour_count) != int(problem.pour_count):
        raise ValueError("pour_count must match the selected transfer_case")


def bind_sampling_metadata(
    problem: ResolvedProblem,
    *,
    query_probabilities: dict[str, float],
    case_probabilities: dict[str, float],
    answer_support_probabilities: dict[str, float],
    params: dict[str, Any],
) -> ResolvedProblem:
    validate_explicit_pour_params(problem, params)
    return replace(
        problem,
        query_probabilities=dict(query_probabilities),
        case_probabilities=dict(case_probabilities),
        answer_support_probabilities=dict(answer_support_probabilities),
    )


__all__ = [
    "case_probability_map",
    "bind_sampling_metadata",
    "FILL_COUNT_MAX",
    "FILL_COUNT_MIN",
    "fmt_number",
    "json_answer_value",
    "resolve_cone_fill_count",
    "resolve_cone_resulting_height",
    "resolve_cylinder_fill_count",
    "resolve_cylinder_resulting_height",
    "support_from_cases",
    "uniform_answer_support",
    "validate_cone_fill_case",
    "validate_cone_height_case",
    "validate_cylinder_fill_case",
    "validate_cylinder_height_case",
    "validate_explicit_pour_params",
]
