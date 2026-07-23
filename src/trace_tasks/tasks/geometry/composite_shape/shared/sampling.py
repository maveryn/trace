"""Identity-free sampling helpers for composite-shape cases."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, TypeVar

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng

T = TypeVar("T")


def select_case_value(
    values: Sequence[T],
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> T:
    """Select one case value, honoring an explicit case index."""

    if not values:
        raise ValueError("case values must be non-empty")
    explicit_case = params.get("case_index")
    if explicit_case is not None:
        index = int(explicit_case)
        if index < 0 or index >= len(values):
            raise ValueError(f"case_index must be in [0, {len(values) - 1}]")
        return values[index]
    else:
        rng = spawn_rng(int(instance_seed), str(namespace))
        return uniform_choice(rng, tuple(values))


def answer_key(value: Any) -> str:
    """Return a stable support key for numeric final answers."""

    if isinstance(value, float):
        return f"{float(value):.1f}"
    return str(value)


def group_cases_by_answer(
    values: Sequence[T],
    *,
    answer_fn: Callable[[T], Any],
) -> Dict[str, Tuple[T, ...]]:
    """Group candidate cases by their final prompt-facing answer."""

    grouped: dict[str, list[T]] = defaultdict(list)
    for value in values:
        grouped[answer_key(answer_fn(value))].append(value)
    if not grouped:
        raise ValueError("answer-balanced case pool must not be empty")
    return {key: tuple(cases) for key, cases in grouped.items()}


def _answer_sort_key(key: str) -> tuple[int, float | str]:
    try:
        return (0, float(key))
    except ValueError:
        return (1, str(key))


def select_answer_balanced_case(
    answer_cases: Mapping[str, Sequence[T]],
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> tuple[T, Dict[str, float]]:
    """Select a candidate by final answer first, then by case within that answer.

    The historical name is kept because many composite-shape tasks already
    import it. The policy is answer-first uniform sampling rather than
    consecutive-seed balancing.
    """

    keys = tuple(sorted((str(key) for key in answer_cases), key=_answer_sort_key))
    if not keys:
        raise ValueError("answer_cases must contain at least one final answer")
    probability = 1.0 / float(len(keys))
    support_probabilities = {key: float(probability) for key in keys}

    explicit_case = params.get("case_index")
    if explicit_case is not None:
        flat_cases = tuple(
            case for key in keys for case in tuple(answer_cases[str(key)])
        )
        if not flat_cases:
            raise ValueError("answer_cases must contain at least one case")
        case_index = int(explicit_case)
        if case_index < 0 or case_index >= len(flat_cases):
            raise ValueError(f"case_index must be in [0, {len(flat_cases) - 1}]")
        return flat_cases[int(case_index)], support_probabilities

    explicit_answer = params.get("target_answer")
    if explicit_answer is not None:
        answer = answer_key(explicit_answer)
        if answer not in answer_cases:
            raise ValueError(f"target_answer {explicit_answer!r} is not in answer support")
    else:
        rng = spawn_rng(int(instance_seed), f"{namespace}.answer")
        answer = str(uniform_choice(rng, keys))
    cases = tuple(answer_cases[str(answer)])
    if not cases:
        raise ValueError(f"answer {answer!r} has no candidate cases")
    rng = spawn_rng(int(instance_seed), f"{namespace}.case.{answer}")
    return uniform_choice(rng, cases), support_probabilities
