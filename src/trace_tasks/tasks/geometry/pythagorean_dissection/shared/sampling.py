"""Sampling helpers for Pythagorean dissection cases."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Mapping

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng

from .state import PythagoreanDissectionPlan, PythagoreanLegCase


def _candidate_cases() -> tuple[PythagoreanLegCase, ...]:
    """Return a broad finite support of leg pairs with readable segment sizes."""

    cases: list[PythagoreanLegCase] = []
    for leg_a in range(3, 15):
        for leg_b in range(leg_a + 1, 19):
            if 8 <= leg_a + leg_b <= 30 and float(leg_b) / float(leg_a) <= 3.2:
                cases.append(PythagoreanLegCase(int(leg_a), int(leg_b)))
    return tuple(cases)


SQUARE_AREA_CASES: tuple[PythagoreanLegCase, ...] = _candidate_cases()
SQUARE_AREA_ANSWER_SUPPORT: tuple[int, ...] = tuple(
    sorted({case.central_square_area for case in SQUARE_AREA_CASES})
)


def _int_param(params: Mapping[str, Any], key: str) -> int | None:
    value = params.get(str(key))
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError(f"{key} must be an integer")
    return int(value)


def _explicit_leg_case(params: Mapping[str, Any]) -> PythagoreanLegCase | None:
    leg_a = _int_param(params, "leg_a")
    leg_b = _int_param(params, "leg_b")
    if leg_a is None:
        leg_a = _int_param(params, "leg_vertical")
    if leg_b is None:
        leg_b = _int_param(params, "leg_horizontal")
    if leg_a is None and leg_b is None:
        return None
    if leg_a is None or leg_b is None:
        raise ValueError("explicit Pythagorean dissection override requires both legs")
    case = PythagoreanLegCase(int(leg_a), int(leg_b))
    if case.central_square_area not in set(SQUARE_AREA_ANSWER_SUPPORT):
        raise ValueError("explicit Pythagorean dissection legs are outside supported readable range")
    return case


@lru_cache(maxsize=None)
def _cases_by_answer() -> dict[int, tuple[PythagoreanLegCase, ...]]:
    grouped: dict[int, list[PythagoreanLegCase]] = {}
    for case in SQUARE_AREA_CASES:
        grouped.setdefault(int(case.central_square_area), []).append(case)
    return {int(key): tuple(values) for key, values in grouped.items()}


def select_square_area_answer(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> int:
    """Select or validate the integer central-square area before construction."""

    explicit_case = _explicit_leg_case(params)
    if explicit_case is not None:
        return int(explicit_case.central_square_area)
    explicit_answer = params.get("answer_value", params.get("target_answer"))
    if explicit_answer is not None:
        answer = int(explicit_answer)
        if answer not in set(SQUARE_AREA_ANSWER_SUPPORT):
            raise ValueError(f"answer_value={answer!r} is outside supported square-area values")
        return int(answer)
    rng = spawn_rng(int(instance_seed), str(namespace))
    return int(uniform_choice(rng, SQUARE_AREA_ANSWER_SUPPORT))


def build_square_area_plan(
    *,
    answer: int,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> PythagoreanDissectionPlan:
    """Bind one selected area answer to a compatible leg-pair construction."""

    answer_int = int(answer)
    cases_by_answer = _cases_by_answer()
    if answer_int not in cases_by_answer:
        raise ValueError(f"unsupported Pythagorean dissection answer: {answer_int}")
    explicit_case = _explicit_leg_case(params)
    if explicit_case is not None:
        if explicit_case.central_square_area != answer_int:
            raise ValueError("explicit leg case conflicts with selected answer")
        case = explicit_case
    else:
        candidates = tuple(cases_by_answer[answer_int])
        rng = spawn_rng(int(instance_seed), f"{namespace}.{answer_int}")
        case = uniform_choice(rng, candidates)
    case_index = SQUARE_AREA_CASES.index(case)
    leg_a = int(case.leg_a)
    leg_b = int(case.leg_b)
    outer_side = int(leg_a + leg_b)
    witness = {
        "leg_a": int(leg_a),
        "leg_b": int(leg_b),
        "leg_vertical": int(leg_a),
        "leg_horizontal": int(leg_b),
        "outer_square_side": int(outer_side),
        "outer_square_area": int(outer_side * outer_side),
        "corner_triangle_area_each": float(leg_a * leg_b / 2.0),
        "corner_triangle_count": 4,
        "vertical_square_area": int(leg_a * leg_a),
        "horizontal_square_area": int(leg_b * leg_b),
        "central_square_area": int(answer_int),
        "central_square_side": round(float(case.central_square_side), 3),
        "case_index": int(case_index),
        "formula_family": "central_square_area_from_triangle_legs",
        "dissection_relation": "square EFGH area = leg_a^2 + leg_b^2",
    }
    return PythagoreanDissectionPlan(
        answer=int(answer_int),
        leg_a=int(leg_a),
        leg_b=int(leg_b),
        outer_square_side=int(outer_side),
        vertical_square_area=int(leg_a * leg_a),
        horizontal_square_area=int(leg_b * leg_b),
        central_square_side=float(case.central_square_side),
        case_index=int(case_index),
        answer_support=tuple(SQUARE_AREA_ANSWER_SUPPORT),
        witness=dict(witness),
    )


__all__ = [
    "SQUARE_AREA_ANSWER_SUPPORT",
    "SQUARE_AREA_CASES",
    "build_square_area_plan",
    "select_square_area_answer",
]
