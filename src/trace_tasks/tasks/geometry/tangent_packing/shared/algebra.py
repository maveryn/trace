"""Shared algebraic objective construction for tangent-packing diagrams."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Callable, Mapping

from .measurements import case_trace_values, fmt_measure
from .sampling import RADIUS_SUPPORT, choose_radius, uniform_probability_map
from .state import TangentPackingCase, TangentPackingProblem


@dataclass(frozen=True)
class RadiusFromGapAreaSpec:
    """Formula parameters for solving a radius from a visible gap area."""

    namespace_key: str
    family: str
    construction: str
    coefficient: float
    gap_area_fn: Callable[[int], float]
    formula_text: str
    extra_trace_fn: Callable[[TangentPackingCase, float], Mapping[str, float]]
    derived_radius_key: str


def circle_in_square_radius_trace_values(case: TangentPackingCase, shaded_area: float) -> dict[str, float]:
    """Return construction-specific trace values for a circle inside a square."""

    return {
        "square_area": float(case.square_side * case.square_side),
        "circle_area": float(math.pi * case.radius * case.radius),
    }


def two_circles_rectangle_radius_trace_values(case: TangentPackingCase, shaded_area: float) -> dict[str, float]:
    """Return construction-specific trace values for two circles inside a rectangle."""

    return {
        "rectangle_area": float(case.packed_rectangle_width * case.packed_rectangle_height),
        "total_circle_area": float(2.0 * math.pi * case.radius * case.radius),
    }


def prepare_radius_from_gap_area(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    spec: RadiusFromGapAreaSpec,
) -> tuple[TangentPackingProblem, float, dict[str, Any]]:
    """Bind one inverse-radius gap-area problem without public-task routing logic."""

    case, radius_probabilities = choose_radius(
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{spec.namespace_key}.radius",
    )
    answer = float(case.radius)
    shaded_area = float(spec.gap_area_fn(int(case.radius)))
    radius_check = math.sqrt(float(shaded_area) / float(spec.coefficient))
    support_probabilities = uniform_probability_map(
        (float(radius) for radius in RADIUS_SUPPORT),
        key_fn=lambda value: f"{float(value):.1f}",
    )
    problem = TangentPackingProblem(
        construction_kind=str(spec.construction),
        target_kind="radius",
        support_kind="shaded_area",
        target_text="r=?",
        support_text=f"shaded area={fmt_measure(shaded_area)}",
        answer=answer,
        case=case,
        formula_family=str(spec.family),
        formula_text=str(spec.formula_text),
        reasoning_steps=1,
        answer_type="number",
        answer_rounding="one_decimal",
        radius_probabilities=dict(radius_probabilities),
        answer_support_probabilities=dict(support_probabilities),
    )
    trace_values = {
        **case_trace_values(case),
        "formula_family": str(spec.family),
        "construction_kind": str(spec.construction),
        "target_kind": "radius",
        "support_kind": "shaded_area",
        "radius_probabilities": dict(radius_probabilities),
        "target_support_probabilities": dict(support_probabilities),
        "visible_shaded_area": float(shaded_area),
        "gap_coefficient": float(spec.coefficient),
        str(spec.derived_radius_key): float(radius_check),
        **dict(spec.extra_trace_fn(case, shaded_area)),
    }
    return problem, answer, trace_values


__all__ = [
    "RadiusFromGapAreaSpec",
    "circle_in_square_radius_trace_values",
    "prepare_radius_from_gap_area",
    "two_circles_rectangle_radius_trace_values",
]
