"""Sampling helpers for trapezoid-extension construction cases."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng

from .measurements import fmt_measure
from .state import TrapezoidExtensionCase


def uniform_probability_map(values: Sequence[Any], *, key_fn: Callable[[Any], str] = str) -> dict[str, float]:
    """Return a uniform probability map over unique formatted values."""

    keys = tuple(dict.fromkeys(str(key_fn(value)) for value in values))
    if not keys:
        return {}
    probability = 1.0 / float(len(keys))
    return {key: probability for key in keys}


def choose_case_by_answer(
    *,
    cases: Sequence[TrapezoidExtensionCase],
    answer_fn: Callable[[TrapezoidExtensionCase], float],
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[TrapezoidExtensionCase, dict[str, float]]:
    """Choose a construction by first sampling uniformly over answer values."""

    by_answer: dict[str, list[TrapezoidExtensionCase]] = defaultdict(list)
    for case in cases:
        by_answer[str(fmt_measure(answer_fn(case)))].append(case)
    if not by_answer:
        raise ValueError("empty trapezoid-extension answer support")

    answer_keys = tuple(sorted(by_answer.keys(), key=lambda value: (float(value), value)))
    explicit = params.get("answer_value")
    if explicit is None:
        explicit = params.get("target_answer")
    if explicit is not None:
        selected_key = str(fmt_measure(float(explicit)))
        if selected_key not in by_answer:
            raise ValueError(f"unsupported trapezoid-extension answer: {selected_key}")
        answer_probabilities = {key: (1.0 if key == selected_key else 0.0) for key in answer_keys}
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.answer")
        selected_key = str(uniform_choice(rng, answer_keys))
        answer_probabilities = uniform_probability_map(answer_keys)

    variants = tuple(by_answer[selected_key])
    rng = spawn_rng(int(instance_seed), f"{namespace}.case.{selected_key}")
    return uniform_choice(rng, variants), dict(answer_probabilities)


__all__ = ["choose_case_by_answer", "uniform_probability_map"]
