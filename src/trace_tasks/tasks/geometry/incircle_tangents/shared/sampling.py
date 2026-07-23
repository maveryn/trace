"""Sampling helpers for incircle-tangent measurements."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Dict, Mapping, Sequence, TypeVar

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.measurement_rendering import fmt_measure
from trace_tasks.tasks.shared.fixed_query import geometry_selected_probability_map

from .state import TangentTriangleCase

AnswerT = TypeVar("AnswerT", int, float)


def all_tangent_triangle_cases(
    *,
    min_tangent: int = 2,
    max_tangent: int = 24,
    min_semiperimeter: int = 10,
    max_semiperimeter: int = 72,
) -> tuple[TangentTriangleCase, ...]:
    """Enumerate readable positive integer tangent triples."""

    cases: list[TangentTriangleCase] = []
    for tangent_a in range(int(min_tangent), int(max_tangent) + 1):
        for tangent_b in range(int(min_tangent), int(max_tangent) + 1):
            for tangent_c in range(int(min_tangent), int(max_tangent) + 1):
                semiperimeter = int(tangent_a) + int(tangent_b) + int(tangent_c)
                if not (int(min_semiperimeter) <= semiperimeter <= int(max_semiperimeter)):
                    continue
                cases.append(
                    TangentTriangleCase(
                        tangent_a=int(tangent_a),
                        tangent_b=int(tangent_b),
                        tangent_c=int(tangent_c),
                    )
                )
    return tuple(cases)


def group_cases_by_answer(
    *,
    cases: Sequence[TangentTriangleCase],
    answer_fn: Callable[[TangentTriangleCase], AnswerT],
) -> Dict[AnswerT, tuple[TangentTriangleCase, ...]]:
    """Group valid tangent cases by final answer."""

    grouped: dict[AnswerT, list[TangentTriangleCase]] = defaultdict(list)
    for case in cases:
        grouped[answer_fn(case)].append(case)
    return {
        answer: tuple(values)
        for answer, values in sorted(grouped.items(), key=lambda item: float(item[0]))
        if values
    }


def explicit_tangent_case(params: Mapping[str, object]) -> TangentTriangleCase | None:
    """Return an explicitly requested tangent case, if all tangent params are present."""

    keys = ("tangent_a", "tangent_b", "tangent_c")
    if not all(key in params for key in keys):
        return None
    values = {key: int(params[key]) for key in keys}
    if min(values.values()) <= 0:
        raise ValueError("tangent_a, tangent_b, and tangent_c must be positive")
    return TangentTriangleCase(
        tangent_a=int(values["tangent_a"]),
        tangent_b=int(values["tangent_b"]),
        tangent_c=int(values["tangent_c"]),
    )


def select_answer_balanced_case(
    *,
    answer_cases: Mapping[AnswerT, Sequence[TangentTriangleCase]],
    instance_seed: int,
    params: Mapping[str, object],
    namespace: str,
    selected_answer: AnswerT | None = None,
    key_fn: Callable[[AnswerT], str] | None = None,
) -> tuple[TangentTriangleCase, int, dict[str, float]]:
    """Select an answer uniformly, then select one construction for that answer."""

    answer_values = tuple(sorted(answer_cases.keys(), key=float))
    if not answer_values:
        raise ValueError("answer_cases must not be empty")
    if selected_answer is not None:
        if selected_answer not in answer_cases:
            answer_values = tuple(sorted(set(answer_values + (selected_answer,)), key=float))
        cases = tuple(answer_cases.get(selected_answer, ()))
        if not cases:
            raise ValueError(f"selected_answer has no cases: {selected_answer}")
        selected_index = 0
        support = geometry_selected_probability_map(
            answer_values,
            selected_answer,
            key_fn=key_fn,
            sort_unique=True,
        )
        return cases[0], selected_index, support

    rng = spawn_rng(int(instance_seed), f"{namespace}.answer")
    answer = uniform_choice(rng, answer_values)
    cases = tuple(answer_cases[answer])
    rng = spawn_rng(int(instance_seed), f"{namespace}.case.{fmt_measure(answer)}")
    case = uniform_choice(rng, cases)
    case_index = cases.index(case)
    support = geometry_selected_probability_map(
        answer_values,
        answer,
        key_fn=key_fn,
        sort_unique=True,
    )
    return case, int(case_index), support


def support_for_answer(
    *,
    answer_cases: Mapping[AnswerT, Sequence[TangentTriangleCase]],
    answer: AnswerT,
    key_fn: Callable[[AnswerT], str] | None = None,
) -> dict[str, float]:
    """Return answer-support probabilities for an explicitly supplied case."""

    answer_values = tuple(sorted(set(answer_cases.keys()) | {answer}, key=float))
    return geometry_selected_probability_map(
        answer_values,
        answer,
        key_fn=key_fn,
        sort_unique=True,
    )


def format_number_answer_key(value: int | float) -> str:
    """Format numeric answer support keys like the visible answer."""

    return fmt_measure(float(value))


__all__ = [
    "all_tangent_triangle_cases",
    "explicit_tangent_case",
    "format_number_answer_key",
    "group_cases_by_answer",
    "select_answer_balanced_case",
    "support_for_answer",
]
