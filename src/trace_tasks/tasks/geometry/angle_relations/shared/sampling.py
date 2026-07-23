"""Sampling primitives for angle-relations scene cases."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, TypeVar

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng


_CaseT = TypeVar("_CaseT")


def select_indexed_case(
    *,
    cases: Sequence[_CaseT],
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[_CaseT, int]:
    """Select one pre-built scene case using explicit params or a resolved index."""

    if not cases:
        raise ValueError("angle-relations case support must not be empty")
    explicit_case = params.get("case_index")
    if explicit_case is not None:
        case_index = int(explicit_case)
        if case_index < 0 or case_index >= len(cases):
            raise ValueError("case_index is outside angle-relations case support")
    else:
        rng = spawn_rng(int(instance_seed), str(namespace))
        case_index = int(uniform_choice(rng, tuple(range(len(cases)))))
    return cases[case_index], int(case_index)


def algebraic_case_parameters_for_answer(answer_value: int, *, variant_index: int) -> tuple[int, int, int, int, int]:
    """Derive a valid algebraic angle case from a target answer."""

    answer = int(answer_value)
    given_min = max(35, 180 - answer - 92)
    given_max = min(64, 180 - answer - 42)
    if given_min > given_max:
        raise ValueError(f"target answer cannot form a stable triangle: {answer}")
    span = int(given_max - given_min + 1)
    given_angle = int(given_min + ((answer * 7 + int(variant_index) * 11) % span))

    coefficient_order = (1, 2, 3, 4, 5, 6)
    start = answer + int(variant_index)
    while start >= len(coefficient_order):
        start -= len(coefficient_order)
    rotated_coefficients = coefficient_order[start:] + coefficient_order[:start]
    for offset, coefficient in enumerate(rotated_coefficients):
        target_coeff = int(coefficient)
        max_x = min(24, (answer - 5) // target_coeff)
        if max_x < 10:
            continue
        x_span = int(max_x - 10 + 1)
        x_value = int(10 + ((answer * 5 + int(variant_index) * 13 + offset) % x_span))
        target_const = int(answer - target_coeff * x_value)
        for delta in (1, 2, 3):
            exterior_coeff = int(target_coeff + delta)
            exterior_const = int(given_angle + answer - exterior_coeff * x_value)
            if 10 <= exterior_const <= 90:
                return given_angle, x_value, target_coeff, target_const, exterior_coeff
    raise ValueError(f"could not derive algebraic case parameters for answer: {answer}")


def triangle_exterior_parameters_for_answer(answer_value: int, *, variant_index: int) -> tuple[int, int]:
    """Derive a valid exterior-angle triangle case from a target answer."""

    answer = int(answer_value)
    given_min = max(35, 180 - answer - 92)
    given_max = min(66, 180 - answer - 38)
    if given_min > given_max:
        raise ValueError(f"target answer cannot form an exterior-angle triangle: {answer}")
    span = int(given_max - given_min + 1)
    given_angle = int(given_min + ((answer * 5 + int(variant_index) * 7) % span))
    return given_angle, answer


__all__ = [
    "algebraic_case_parameters_for_answer",
    "select_indexed_case",
    "triangle_exterior_parameters_for_answer",
]
