"""Identity-free construction math for paper-fold diagrams."""

from __future__ import annotations

from functools import lru_cache
import math
from typing import Tuple

from trace_tasks.tasks.geometry.shared.measurement_rendering import round1
from trace_tasks.tasks.geometry.shared.pythagorean import (
    IntegerRightTriangle,
    integer_right_triangles,
    validate_integer_right_triangle,
)

from .state import FoldGeometry, FoldSegmentCase, FoldSegmentGeometry

FoldCase = Tuple[int, int]
FoldAnswerCases = Tuple[float, Tuple[FoldCase, ...]]
FoldSegmentAnswerCases = Tuple[int, Tuple[FoldSegmentCase, ...]]

_MIN_HEIGHT_UNITS = 10
_MAX_HEIGHT_UNITS = 34
_MIN_FOLDED_OFFSET_UNITS = 4
_MIN_ANSWER_DEGREES = 50.0
_MAX_ANSWER_DEGREES = 78.0
_MAX_SEGMENT_LEG_UNITS = 400
_MAX_SEGMENT_HYPOTENUSE_UNITS = 580
_MIN_SEGMENT_LEG_RATIO = 1.35
_MAX_SEGMENT_LEG_RATIO = 3.0


def fold_geometry(height_units: float, folded_offset_units: float) -> FoldGeometry:
    """Return the analytic geometry for a corner folded onto the bottom edge."""

    height = float(height_units)
    offset = float(folded_offset_units)
    upper = (height * height + offset * offset) / (2.0 * height)
    lower = height - upper
    crease_top_x = (height * height + offset * offset) / (2.0 * offset)
    width_units = max(18.0, crease_top_x + 4.0, offset + 7.0)
    half_angle = 90.0 - math.degrees(math.atan(offset / height))
    return FoldGeometry(
        height_units=height,
        folded_offset_units=offset,
        width_units=width_units,
        upper_segment_units=upper,
        lower_segment_units=lower,
        half_angle_degrees=half_angle,
        total_angle_degrees=2.0 * half_angle,
    )


@lru_cache(maxsize=1)
def fold_answer_cases() -> tuple[FoldAnswerCases, ...]:
    """Return visually stable fold cases grouped by rounded answer value.

    Sampling uses the rounded answer as the first-stage support so review and
    training data do not over-represent common integer height/offset ratios
    that collapse to the same angle.
    """

    grouped: dict[float, list[FoldCase]] = {}
    for height in range(_MIN_HEIGHT_UNITS, _MAX_HEIGHT_UNITS + 1):
        for offset in range(_MIN_FOLDED_OFFSET_UNITS, height - 2):
            geometry = fold_geometry(float(height), float(offset))
            answer = round1(geometry.half_angle_degrees)
            if _MIN_ANSWER_DEGREES <= float(answer) <= _MAX_ANSWER_DEGREES:
                grouped.setdefault(float(answer), []).append((int(height), int(offset)))
    return tuple(
        (float(answer), tuple(cases))
        for answer, cases in sorted(grouped.items(), key=lambda item: float(item[0]))
        if cases
    )


def fold_answer_support() -> tuple[float, ...]:
    """Return the rounded answer support for paper-fold angle samples."""

    return tuple(float(answer) for answer, _cases in fold_answer_cases())


def fold_segment_geometry(leg_ae: int, leg_af: int) -> FoldSegmentGeometry:
    """Return fold geometry where crease EF folds original point A onto edge point P."""

    leg_ae = int(leg_ae)
    leg_af = int(leg_af)
    hypotenuse = int(math.isqrt(leg_ae * leg_ae + leg_af * leg_af))
    validate_integer_right_triangle(
        IntegerRightTriangle(
            leg_a=int(leg_ae),
            leg_b=int(leg_af),
            hypotenuse=int(hypotenuse),
        )
    )
    hypotenuse_sq = float(hypotenuse * hypotenuse)
    folded_x = (2.0 * float(leg_af) * float(leg_ae * leg_ae)) / hypotenuse_sq
    folded_y = (2.0 * float(leg_ae) * float(leg_af * leg_af)) / hypotenuse_sq
    if folded_y <= float(leg_ae):
        raise ValueError("folded point P must land below E on the paper edge")
    width_units = max(float(leg_af) + 6.0, float(folded_x) + 6.0, 18.0)
    height_units = float(folded_y)
    return FoldSegmentGeometry(
        leg_ae=int(leg_ae),
        leg_af=int(leg_af),
        crease_ef=int(hypotenuse),
        width_units=float(width_units),
        height_units=float(height_units),
        folded_point_units=(float(folded_x), float(folded_y)),
    )


def _segment_cases_from_triangle(triangle: IntegerRightTriangle) -> tuple[FoldSegmentCase, ...]:
    leg_ae = int(triangle.leg_a)
    leg_af = int(triangle.leg_b)
    crease_ef = int(triangle.hypotenuse)
    if leg_af <= leg_ae:
        return tuple()
    leg_ratio = float(max(leg_ae, leg_af)) / float(min(leg_ae, leg_af))
    if leg_ratio < _MIN_SEGMENT_LEG_RATIO or leg_ratio > _MAX_SEGMENT_LEG_RATIO:
        return tuple()
    return (
        FoldSegmentCase(
            leg_ae=int(leg_ae),
            leg_af=int(leg_af),
            crease_ef=int(crease_ef),
            target_segment="EP",
            known_leg_segment="AF",
            target_answer=int(leg_ae),
        ),
        FoldSegmentCase(
            leg_ae=int(leg_ae),
            leg_af=int(leg_af),
            crease_ef=int(crease_ef),
            target_segment="FP",
            known_leg_segment="AE",
            target_answer=int(leg_af),
        ),
    )


@lru_cache(maxsize=1)
def fold_segment_answer_cases() -> tuple[FoldSegmentAnswerCases, ...]:
    """Return folded-segment cases grouped by integer answer value."""

    grouped: dict[int, list[FoldSegmentCase]] = {}
    for triangle in integer_right_triangles(
        min_leg=5,
        max_leg=_MAX_SEGMENT_LEG_UNITS,
        max_hypotenuse=_MAX_SEGMENT_HYPOTENUSE_UNITS,
        include_swapped=True,
    ):
        for case in _segment_cases_from_triangle(triangle):
            grouped.setdefault(int(case.target_answer), []).append(case)
    return tuple(
        (int(answer), tuple(cases))
        for answer, cases in sorted(grouped.items(), key=lambda item: int(item[0]))
        if cases
    )


def fold_segment_answer_support() -> tuple[int, ...]:
    """Return the integer answer support for folded-segment samples."""

    return tuple(int(answer) for answer, _cases in fold_segment_answer_cases())


__all__ = [
    "FoldAnswerCases",
    "FoldCase",
    "FoldSegmentAnswerCases",
    "fold_answer_cases",
    "fold_answer_support",
    "fold_geometry",
    "fold_segment_answer_cases",
    "fold_segment_answer_support",
    "fold_segment_geometry",
]
