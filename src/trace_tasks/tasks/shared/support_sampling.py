"""Shared integer-support sampling helpers for task domains."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple, TypeVar

from ...core.seed import spawn_rng
from ...core.sampling import support_probability_map, uniform_choice_with_probabilities
from .config_defaults import group_default

T = TypeVar("T")


def resolve_integer_support(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
) -> Tuple[int, ...]:
    """Resolve one explicit integer support list from params/defaults."""

    raw_support = params.get(str(key), group_default(gen_defaults, str(key), tuple(int(value) for value in fallback)))
    support: list[int] = []
    for raw_value in raw_support:
        value = int(raw_value)
        if value not in support:
            support.append(int(value))
    if not support:
        raise ValueError(f"{key} must contain at least one integer")
    return tuple(sorted(support))


def resolve_integer_choice(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
    use_instance_seed_cycle: bool = False,
    namespace_support_permutation: bool = False,
) -> Tuple[int, Dict[str, float]]:
    """Resolve one integer choice from explicit support with seeded RNG sampling.

    When an internal finite-support caller provides ``_sample_cursor`` and the
    configured balance flag is enabled, this uses the cursor as a round-robin
    coverage index over the explicit support. Normal dataset sampling still
    uses seeded RNG draws over the support instead of seed modulo.
    """

    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=fallback_support,
    )
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        selected = int(explicit)
        if int(selected) not in set(support):
            raise ValueError(f"unsupported {explicit_key}: {selected}")
        return int(selected), support_probability_map(support, selected=int(selected), sort_keys=True)

    balanced = bool(params.get(str(balanced_flag_key), group_default(gen_defaults, str(balanced_flag_key), True)))
    sample_cursor = params.get("_sample_cursor")
    if bool(balanced) and sample_cursor is not None:
        values = tuple(int(value) for value in support)
        if bool(namespace_support_permutation) and len(values) > 1:
            shuffled = list(values)
            permutation_rng = spawn_rng(0, f"{namespace}.support_permutation")
            permutation_rng.shuffle(shuffled)
            values = tuple(int(value) for value in shuffled)
        selected = values[abs(int(sample_cursor)) % len(values)]
        return int(selected), support_probability_map(support, selected=int(selected), sort_keys=True)

    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = uniform_choice_with_probabilities(
        rng,
        support,
        sort_keys=True,
    )
    return int(selected), dict(probabilities)


def resolve_support_choice(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    support: Sequence[T],
    explicit_key: str | Sequence[str] | None,
    namespace: str,
    sort_keys: bool = False,
) -> Tuple[T, Dict[str, float]]:
    """Resolve one finite-support value with explicit override validation.

    This is the non-integer companion to ``resolve_integer_choice``. It samples
    by seeded RNG over an explicit support, not by seed/hash modulo.
    """

    values = tuple(support)
    if not values:
        raise ValueError("support must contain at least one value")
    keys = {str(value): value for value in values}
    if len(keys) != len(values):
        raise ValueError("support values must have unique string keys")

    explicit_keys: tuple[str, ...]
    if explicit_key is None:
        explicit_keys = ()
    elif isinstance(explicit_key, str):
        explicit_keys = (str(explicit_key),)
    else:
        explicit_keys = tuple(str(value) for value in explicit_key)
    for key in explicit_keys:
        raw_value = params.get(str(key))
        if raw_value is None:
            continue
        selected_key = str(raw_value)
        if selected_key not in keys:
            raise ValueError(f"unsupported {key}: {raw_value}")
        selected = keys[selected_key]
        return selected, support_probability_map(values, selected=selected, sort_keys=sort_keys)

    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = uniform_choice_with_probabilities(
        rng,
        values,
        sort_keys=bool(sort_keys),
    )
    return selected, dict(probabilities)


__all__ = [
    "resolve_integer_choice",
    "resolve_integer_support",
    "resolve_support_choice",
]
