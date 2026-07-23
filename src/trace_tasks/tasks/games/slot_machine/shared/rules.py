"""Identity-free rules helpers for slot-machine tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support


@dataclass(frozen=True)
class TargetCountSelection:
    """Resolved integer target and review metadata for construction."""

    value: int
    probabilities: dict[str, float]
    support: tuple[int, ...]


def resolve_target_count(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> TargetCountSelection:
    """Resolve a task-owned target count without knowing the public task id."""

    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    target, probabilities = resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=support,
        namespace=str(namespace),
        balanced_flag_key=str(balanced_flag_key),
        namespace_support_permutation=True,
    )
    return TargetCountSelection(
        value=int(target),
        probabilities=dict(probabilities),
        support=tuple(int(value) for value in support),
    )


def target_count_trace_params(
    *,
    target: TargetCountSelection,
    explicit_key: str,
    support_key: str,
    probabilities_key: str,
) -> dict[str, Any]:
    """Return trace params for a resolved target count."""

    return {
        str(explicit_key): int(target.value),
        str(support_key): [int(value) for value in target.support],
        str(probabilities_key): dict(target.probabilities),
    }


__all__ = [
    "TargetCountSelection",
    "resolve_target_count",
    "target_count_trace_params",
]
