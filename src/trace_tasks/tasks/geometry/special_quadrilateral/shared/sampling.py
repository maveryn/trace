"""Case-selection primitives for special-quadrilateral constructions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng

from .state import QuadrilateralCase


def _probability_map(values: Sequence[int], selected: int | None = None) -> dict[str, float]:
    resolved = tuple(int(value) for value in values)
    if not resolved:
        return {}
    if selected is not None:
        return {str(value): (1.0 if int(value) == int(selected) else 0.0) for value in resolved}
    weight = 1.0 / float(len(resolved))
    return {str(value): float(weight) for value in resolved}


def select_case_from_answer_support(
    *,
    cases: Sequence[QuadrilateralCase],
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[QuadrilateralCase, int, dict[str, float]]:
    """Choose an answer uniformly first, then choose a compatible case."""

    grouped: dict[int, list[tuple[int, QuadrilateralCase]]] = defaultdict(list)
    for index, case in enumerate(cases):
        grouped[int(case.answer)].append((int(index), case))
    if not grouped:
        raise ValueError("special quadrilateral case support must be non-empty")

    answer_support = tuple(sorted(grouped))
    forced_answer = params.get("answer")
    rng = spawn_rng(int(instance_seed), str(namespace))
    if forced_answer is not None:
        answer = int(forced_answer)
        if answer not in grouped:
            raise ValueError(f"unsupported answer for special quadrilateral case support: {answer}")
        answer_probabilities = _probability_map(answer_support, selected=answer)
    else:
        answer = int(answer_support[int(rng.randrange(len(answer_support)))])
        answer_probabilities = _probability_map(answer_support)

    compatible = tuple(grouped[answer])
    selected_index, selected_case = compatible[int(rng.randrange(len(compatible)))]
    return selected_case, int(selected_index), dict(answer_probabilities)


__all__ = ["select_case_from_answer_support"]
