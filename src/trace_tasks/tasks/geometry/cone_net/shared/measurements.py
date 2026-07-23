"""Measurement formulas for cone-sector net diagrams."""

from __future__ import annotations

import math
from typing import Sequence

from trace_tasks.tasks.geometry.shared.measurement_rendering import fmt_measure, round1

from .state import ConeNetCase, ConeNetDiagramSpec


def base_radius_from_sector(case: ConeNetCase) -> float:
    """Return folded cone base radius from sector radius and angle."""

    return round1(float(case.slant_height) * float(case.theta_degrees) / 360.0)


def height_from_sector(case: ConeNetCase) -> float:
    """Return folded cone height from slant height and derived base radius."""

    radius = float(case.slant_height) * float(case.theta_degrees) / 360.0
    return round1(math.sqrt(max(0.0, float(case.slant_height) ** 2 - float(radius) ** 2)))


def arc_length_from_sector(case: ConeNetCase) -> float:
    """Return the sector arc length corresponding to the folded cone base."""

    arc_length = (float(case.theta_degrees) / 360.0) * 2.0 * math.pi * float(case.slant_height)
    return round1(arc_length)


def base_radius_support_values(cases: Sequence[ConeNetCase]) -> tuple[float, ...]:
    """Return the base-radius answer support for a case pool."""

    return tuple(sorted({base_radius_from_sector(case) for case in cases}))


def height_support_values(cases: Sequence[ConeNetCase]) -> tuple[float, ...]:
    """Return the height answer support for a case pool."""

    return tuple(sorted({height_from_sector(case) for case in cases}))


def cone_net_diagram_spec(
    case: ConeNetCase,
    *,
    answer: float,
    target_measure: str,
    target_label: str,
    target_label_anchor: str,
    annotation_roles: tuple[str, ...],
    formula_family: str,
    reasoning_steps: int,
) -> ConeNetDiagramSpec:
    """Bind one case to task-specific labels and annotation roles."""

    return ConeNetDiagramSpec(
        answer=round1(float(answer)),
        slant_height=int(case.slant_height),
        theta_degrees=int(case.theta_degrees),
        base_radius=base_radius_from_sector(case),
        cone_height=height_from_sector(case),
        arc_length=arc_length_from_sector(case),
        target_measure=str(target_measure),
        target_label=str(target_label),
        target_label_anchor=str(target_label_anchor),
        annotation_roles=tuple(str(role) for role in annotation_roles),
        formula_family=str(formula_family),
        reasoning_steps=int(reasoning_steps),
    )


__all__ = [
    "arc_length_from_sector",
    "base_radius_from_sector",
    "base_radius_support_values",
    "cone_net_diagram_spec",
    "fmt_measure",
    "height_from_sector",
    "height_support_values",
    "round1",
]
