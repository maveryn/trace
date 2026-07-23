"""Measurement formulas for concentric-chord tasks."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.geometry.shared.measurement_rendering import fmt_measure

from .state import ConcentricChordCase, ConcentricChordDiagramSpec


def chord_length_from_case(case: ConcentricChordCase) -> int:
    """Return the outer chord length from one tangent-chord case."""

    return int(case.chord_length)


def inner_radius_from_case(case: ConcentricChordCase) -> int:
    """Return the inner radius from one tangent-chord case."""

    return int(case.inner_radius)


def chord_length_support_values(cases: Sequence[ConcentricChordCase]) -> tuple[int, ...]:
    """Return unique chord lengths represented by the case pool."""

    return tuple(sorted({int(case.chord_length) for case in cases}))


def inner_radius_support_values(cases: Sequence[ConcentricChordCase]) -> tuple[int, ...]:
    """Return unique inner radii represented by the case pool."""

    return tuple(sorted({int(case.inner_radius) for case in cases}))


def tangent_chord_diagram_spec(
    case: ConcentricChordCase,
    *,
    answer: int,
    inner_radius_label: str,
    chord_label: str,
    formula_family: str,
    unknown_measure: str,
) -> ConcentricChordDiagramSpec:
    """Build a render spec from a task-owned unknown measurement choice."""

    return ConcentricChordDiagramSpec(
        answer=int(answer),
        outer_radius=int(case.outer_radius),
        inner_radius=int(case.inner_radius),
        half_chord=int(case.half_chord),
        chord_length=int(case.chord_length),
        outer_radius_label=f"R={fmt_measure(case.outer_radius)}",
        inner_radius_label=str(inner_radius_label),
        chord_label=str(chord_label),
        formula_family=str(formula_family),
        unknown_measure=str(unknown_measure),
    )


__all__ = [
    "chord_length_from_case",
    "chord_length_support_values",
    "fmt_measure",
    "inner_radius_from_case",
    "inner_radius_support_values",
    "tangent_chord_diagram_spec",
]
