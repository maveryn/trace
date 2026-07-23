"""Scene-neutral sampling helpers for pages task packages."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.sampling import uniform_choice_with_probabilities
from ...shared.config_defaults import group_default
from ...shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant


def resolve_named_axis(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace_root: str,
    supported: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced named axis without encoding scene/task identity."""

    rng = spawn_rng(int(instance_seed), f"{namespace_root}.{namespace}")
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=[str(value) for value in supported],
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    balanced = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=[str(value) for value in supported],
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=f"{namespace_root}.{namespace}",
    )
    if str(balanced) != str(selected) and params.get(str(explicit_key)) is not None:
        return str(balanced), {str(key): (1.0 if str(key) == str(balanced) else 0.0) for key in supported}
    return str(balanced), dict(probabilities)


def resolve_int_support(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
) -> Tuple[int, ...]:
    """Resolve a de-duplicated integer support list from params or defaults."""

    raw_values = params.get(str(key), group_default(gen_defaults, str(key), fallback))
    values: List[int] = []
    for raw_value in raw_values:
        value = int(raw_value)
        if int(value) not in values:
            values.append(int(value))
    if not values:
        raise ValueError(f"{key} must not be empty")
    return tuple(int(value) for value in values)


def resolve_supported_int(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace_root: str,
    explicit_key: str,
    support_key: str,
    fallback: Sequence[int],
    instance_seed: int,
    namespace: str,
) -> Tuple[int, Tuple[int, ...], Dict[str, float]]:
    """Resolve one supported integer operand and its sampling probabilities."""

    support = resolve_int_support(params=params, gen_defaults=gen_defaults, key=support_key, fallback=fallback)
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        selected = int(explicit)
        if int(selected) not in set(support):
            raise ValueError(f"{explicit_key} must be in {support}")
        return int(selected), tuple(support), {str(int(selected)): 1.0}
    rng = spawn_rng(int(instance_seed), f"{namespace_root}.{namespace}")
    selected, probabilities = uniform_choice_with_probabilities(
        rng,
        support,
        sort_keys=True,
    )
    return int(selected), tuple(support), dict(probabilities)


__all__ = [
    "resolve_int_support",
    "resolve_named_axis",
    "resolve_supported_int",
]
