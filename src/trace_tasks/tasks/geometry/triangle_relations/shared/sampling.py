"""Sampling helpers for triangle-relations construction cases."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Mapping, Sequence, TypeVar

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.geometry.shared.measurement_rendering import fmt_measure

T = TypeVar("T")


def uniform_probability_map(values: Sequence[str]) -> dict[str, float]:
    """Return a uniform probability map over unique values."""

    keys = tuple(dict.fromkeys(str(value) for value in values))
    if not keys:
        return {}
    probability = 1.0 / float(len(keys))
    return {key: probability for key in keys}


def choose_case_by_answer(
    *,
    cases: Sequence[T],
    answer_fn: Callable[[T], int | float],
    params: Mapping[str, object],
    instance_seed: int,
    namespace: str,
) -> tuple[T, dict[str, float]]:
    """Choose a case by uniformly sampling the public answer support first."""

    by_answer: dict[str, list[T]] = defaultdict(list)
    for case in cases:
        by_answer[str(fmt_measure(float(answer_fn(case))))].append(case)
    if not by_answer:
        raise ValueError("empty triangle-relations answer support")

    answer_keys = tuple(sorted(by_answer.keys(), key=lambda value: (float(value), value)))
    explicit = params.get("answer_value")
    if explicit is None:
        explicit = params.get("target_answer")
    if explicit is not None:
        selected_key = str(fmt_measure(float(explicit)))
        if selected_key not in by_answer:
            raise ValueError(f"unsupported triangle-relations answer: {selected_key}")
        answer_probabilities = {key: (1.0 if key == selected_key else 0.0) for key in answer_keys}
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.answer")
        selected_key = str(uniform_choice(rng, answer_keys))
        answer_probabilities = uniform_probability_map(answer_keys)

    variants = tuple(by_answer[selected_key])
    rng = spawn_rng(int(instance_seed), f"{namespace}.case.{selected_key}")
    return uniform_choice(rng, variants), dict(answer_probabilities)


__all__ = ["choose_case_by_answer", "uniform_probability_map"]
