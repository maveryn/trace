"""Scene-local sampling primitives for RPG house tasks."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.sampling import support_probability_map, uniform_choice_with_probabilities
from trace_tasks.tasks.illustrations.shared.task_support import uniform_string_probability_map
from trace_tasks.tasks.shared.support_sampling import resolve_integer_choice, resolve_integer_support


def select_count_from_support(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
) -> tuple[int, Mapping[str, float]]:
    """Select a count from explicit params, config support, or caller-provided support."""

    return resolve_integer_choice(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        support_key=str(support_key),
        explicit_key=str(explicit_key),
        fallback_support=tuple(int(value) for value in fallback_support),
        namespace=str(namespace),
        balanced_flag_key="balanced_sampling",
        use_instance_seed_cycle=True,
    )


def select_feasible_count_from_support(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    feasible: Callable[[int], bool],
    namespace: str,
    empty_context: str,
) -> tuple[int, Mapping[str, float]]:
    """Select a count after filtering configured support through caller feasibility."""

    configured = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    support = tuple(int(value) for value in configured if feasible(int(value)))
    if not support:
        raise ValueError(f"{support_key} has no feasible values for {empty_context}")
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        selected = int(explicit)
        if selected not in set(support):
            raise ValueError(f"{explicit_key} must be in {support}, got {selected}")
        return selected, support_probability_map(support, selected=selected, sort_keys=True)
    sample_namespace = str(namespace)
    if params.get("_sample_cursor") is not None:
        sample_namespace = f"{sample_namespace}:{int(params['_sample_cursor'])}"
    rng = spawn_rng(int(instance_seed), sample_namespace)
    selected, probabilities = uniform_choice_with_probabilities(rng, support, sort_keys=True)
    return int(selected), dict(probabilities)


def select_string_from_support(
    *,
    params: Mapping[str, Any],
    support: Sequence[str],
    explicit_key: str,
    namespace: str,
    instance_seed: int,
) -> tuple[str, Mapping[str, float]]:
    """Select one string value from support with deterministic cursor/seed handling."""

    support_tuple = tuple(str(value) for value in support)
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        value = str(explicit)
        if value not in set(support_tuple):
            raise ValueError(f"{explicit_key} must be one of {support_tuple}")
        return value, uniform_string_probability_map(support_tuple, selected=value)
    sample_namespace = str(namespace)
    if params.get("_sample_cursor") is not None:
        sample_namespace = f"{sample_namespace}:{int(params['_sample_cursor'])}"
    rng = spawn_rng(int(instance_seed), sample_namespace)
    value, probabilities = uniform_choice_with_probabilities(rng, support_tuple, sort_keys=False)
    return str(value), dict(probabilities)


__all__ = [
    "select_count_from_support",
    "select_feasible_count_from_support",
    "select_string_from_support",
]
