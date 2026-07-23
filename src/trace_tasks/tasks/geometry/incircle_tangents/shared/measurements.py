"""Measurement primitives for incircle-tangent diagrams."""

from __future__ import annotations

import math
from typing import Dict

from trace_tasks.tasks.geometry.shared.measurement_rendering import round1

from .state import IncircleDiagramSpec, Point, TangentTriangleCase


def triangle_measurements_from_tangents(case: TangentTriangleCase) -> dict[str, float]:
    """Compute side lengths, area, semiperimeter, and inradius from tangent lengths."""

    tangent_a = float(case.tangent_a)
    tangent_b = float(case.tangent_b)
    tangent_c = float(case.tangent_c)
    side_ab = tangent_a + tangent_b
    side_bc = tangent_b + tangent_c
    side_ca = tangent_c + tangent_a
    semiperimeter = tangent_a + tangent_b + tangent_c
    area = math.sqrt(max(0.0, semiperimeter * tangent_a * tangent_b * tangent_c))
    return {
        "tangent_a": tangent_a,
        "tangent_b": tangent_b,
        "tangent_c": tangent_c,
        "side_ab": side_ab,
        "side_bc": side_bc,
        "side_ca": side_ca,
        "semiperimeter": semiperimeter,
        "area": area,
        "displayed_area": round1(area),
        "inradius": area / semiperimeter,
    }


def perimeter_answer_for_case(case: TangentTriangleCase) -> int:
    """Return the triangle perimeter implied by tangent lengths."""

    values = triangle_measurements_from_tangents(case)
    return int(round(2.0 * float(values["semiperimeter"])))


def radius_answer_for_case(case: TangentTriangleCase) -> float:
    """Return the incircle radius implied by the tangent lengths."""

    values = triangle_measurements_from_tangents(case)
    return round1(float(values["inradius"]))


def incircle_spec_from_case(
    *,
    case: TangentTriangleCase,
    answer: int | float,
    answer_type: str,
    answer_rounding: str,
    unknown_measure: str,
    formula_family: str,
    unknown_label: str,
    show_area_label: bool,
    show_radius_segment: bool,
    annotation_roles: tuple[str, ...],
) -> IncircleDiagramSpec:
    """Build one complete semantic diagram specification."""

    values = triangle_measurements_from_tangents(case)
    return IncircleDiagramSpec(
        answer=answer,
        answer_type=str(answer_type),
        answer_rounding=str(answer_rounding),
        unknown_measure=str(unknown_measure),
        formula_family=str(formula_family),
        tangent_a=float(values["tangent_a"]),
        tangent_b=float(values["tangent_b"]),
        tangent_c=float(values["tangent_c"]),
        side_ab=float(values["side_ab"]),
        side_bc=float(values["side_bc"]),
        side_ca=float(values["side_ca"]),
        semiperimeter=float(values["semiperimeter"]),
        area=float(values["area"]),
        displayed_area=float(values["displayed_area"]),
        inradius=float(values["inradius"]),
        unknown_label=str(unknown_label),
        show_area_label=bool(show_area_label),
        show_radius_segment=bool(show_radius_segment),
        annotation_roles=tuple(str(role) for role in annotation_roles),
    )


def triangle_layout(spec: IncircleDiagramSpec) -> Dict[str, Point | float]:
    """Return unscaled triangle, tangent, and incenter points."""

    ax, ay = 0.0, 0.0
    bx, by = float(spec.side_ab), 0.0
    cx = (
        (float(spec.side_ca) * float(spec.side_ca))
        + (float(spec.side_ab) * float(spec.side_ab))
        - (float(spec.side_bc) * float(spec.side_bc))
    ) / (2.0 * float(spec.side_ab))
    cy = math.sqrt(max(0.0, float(spec.side_ca) * float(spec.side_ca) - cx * cx))
    perimeter = float(spec.side_ab) + float(spec.side_bc) + float(spec.side_ca)
    o = (
        (float(spec.side_ca) * bx + float(spec.side_ab) * cx) / perimeter,
        (float(spec.side_ab) * cy) / perimeter,
    )
    d = (float(spec.tangent_a), 0.0)
    e = (
        bx + (cx - bx) * (float(spec.tangent_b) / float(spec.side_bc)),
        by + (cy - by) * (float(spec.tangent_b) / float(spec.side_bc)),
    )
    f = (
        (cx * float(spec.tangent_a)) / float(spec.side_ca),
        (cy * float(spec.tangent_a)) / float(spec.side_ca),
    )
    return {
        "A": (ax, ay),
        "B": (bx, by),
        "C": (cx, cy),
        "D": d,
        "E": e,
        "F": f,
        "O": o,
        "inradius": float(spec.inradius),
    }


__all__ = [
    "incircle_spec_from_case",
    "perimeter_answer_for_case",
    "radius_answer_for_case",
    "triangle_layout",
    "triangle_measurements_from_tangents",
]
