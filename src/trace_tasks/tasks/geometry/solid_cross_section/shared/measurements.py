"""Formula helpers for solid cross-section tasks."""

from __future__ import annotations

import math

from trace_tasks.tasks.geometry.shared.measurement_rendering import round1

from .state import ConeSliceCase, PyramidSliceCase, SolidCrossSectionProblem


def cone_slice_radius(case: ConeSliceCase) -> float:
    """Return the radius of a cone slice parallel to the base."""

    return float(case.base_radius) * float(case.slice_distance_from_apex) / float(case.solid_height)


def cone_parallel_slice_area(case: ConeSliceCase) -> float:
    """Return the rounded area of a cone slice parallel to the base."""

    return round1(math.pi * cone_slice_radius(case) ** 2)


def pyramid_slice_side(case: PyramidSliceCase) -> float:
    """Return the side length of a square-pyramid slice parallel to the base."""

    return float(case.base_side) * float(case.slice_distance_from_apex) / float(case.solid_height)


def square_pyramid_parallel_slice_area(case: PyramidSliceCase) -> float:
    """Return the rounded area of a square-pyramid slice parallel to the base."""

    return round1(pyramid_slice_side(case) ** 2)


def validate_cone_case(case: ConeSliceCase) -> None:
    """Validate one cone cross-section case before rendering."""

    if int(case.base_radius) <= 0 or int(case.solid_height) <= 0:
        raise ValueError("cone slice dimensions must be positive")
    if not 0 < int(case.slice_distance_from_apex) < int(case.solid_height):
        raise ValueError("cone slice distance must lie inside the solid")


def validate_pyramid_case(case: PyramidSliceCase) -> None:
    """Validate one square-pyramid cross-section case before rendering."""

    if int(case.base_side) <= 0 or int(case.solid_height) <= 0:
        raise ValueError("pyramid slice dimensions must be positive")
    if not 0 < int(case.slice_distance_from_apex) < int(case.solid_height):
        raise ValueError("pyramid slice distance must lie inside the solid")


def cone_problem_from_case(
    case: ConeSliceCase,
    *,
    formula_family: str,
    formula: str,
    answer_support_probabilities: dict[str, float],
    construction_case_count_for_answer: int,
) -> SolidCrossSectionProblem:
    """Bind one cone case to the task-level formula contract."""

    validate_cone_case(case)
    radius = cone_slice_radius(case)
    return SolidCrossSectionProblem(
        solid_kind="cone",
        answer=cone_parallel_slice_area(case),
        formula_family=str(formula_family),
        formula=str(formula),
        base_radius=float(case.base_radius),
        solid_height=float(case.solid_height),
        slice_distance_from_apex=float(case.slice_distance_from_apex),
        slice_radius=float(radius),
        answer_support_probabilities=dict(answer_support_probabilities),
        construction_case_count_for_answer=int(construction_case_count_for_answer),
    )


def pyramid_problem_from_case(
    case: PyramidSliceCase,
    *,
    formula_family: str,
    formula: str,
    answer_support_probabilities: dict[str, float],
    construction_case_count_for_answer: int,
) -> SolidCrossSectionProblem:
    """Bind one square-pyramid case to the task-level formula contract."""

    validate_pyramid_case(case)
    side = pyramid_slice_side(case)
    return SolidCrossSectionProblem(
        solid_kind="square_pyramid",
        answer=square_pyramid_parallel_slice_area(case),
        formula_family=str(formula_family),
        formula=str(formula),
        base_side=float(case.base_side),
        solid_height=float(case.solid_height),
        slice_distance_from_apex=float(case.slice_distance_from_apex),
        slice_side=float(side),
        answer_support_probabilities=dict(answer_support_probabilities),
        construction_case_count_for_answer=int(construction_case_count_for_answer),
    )
