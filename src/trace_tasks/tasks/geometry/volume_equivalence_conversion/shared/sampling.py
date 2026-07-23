"""Scene-neutral sampling primitives for conversion cases."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng


def case_key(branch_name: str, case: Sequence[int]) -> str:
    return f"{branch_name}:" + "_".join(str(int(value)) for value in case)


def select_conversion_case(
    *,
    branch_name: str,
    cases: tuple[tuple[int, ...], ...],
    instance_seed: int,
    params: Mapping[str, Any],
    namespace: str,
) -> tuple[tuple[int, ...], dict[str, float]]:
    """Select one numeric construction case or validate an explicit case."""

    explicit = params.get("conversion_case")
    if explicit is not None:
        if not isinstance(explicit, Sequence) or isinstance(explicit, (str, bytes)):
            raise ValueError("conversion_case must be a numeric sequence")
        case = tuple(int(value) for value in explicit)
        keys = tuple(case_key(str(branch_name), candidate) for candidate in cases) + (
            case_key(str(branch_name), case),
        )
        return case, {
            key: (1.0 if key == case_key(str(branch_name), case) else 0.0)
            for key in dict.fromkeys(keys)
        }

    rng = spawn_rng(int(instance_seed), str(namespace))
    case = tuple(uniform_choice(rng, cases))
    probability = 1.0 / float(len(cases))
    return case, {
        case_key(str(branch_name), candidate): probability
        for candidate in cases
    }


def support_probabilities(values: Sequence[int | str]) -> dict[str, float]:
    support = tuple(dict.fromkeys(str(value) for value in values))
    if not support:
        return {}
    probability = 1.0 / float(len(support))
    return {str(value): probability for value in support}


__all__ = ["case_key", "select_conversion_case", "support_probabilities"]
