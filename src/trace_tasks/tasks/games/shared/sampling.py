"""Shared semantic/visual axis sampling helpers for games-domain tasks."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ...shared.config_defaults import group_default
from ...shared.support_sampling import resolve_integer_choice, resolve_integer_support
from ...shared.variant_sampling import apply_balanced_variant_sampling, resolve_variant


def get_games_int_param(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: int,
) -> int:
    """Resolve one integer parameter using game task/group precedence."""

    return int(params.get(str(key), group_default(defaults, str(key), int(fallback))))


def get_games_int_range(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
) -> Tuple[int, int]:
    """Resolve and validate an inclusive integer range for games-domain samplers."""

    lower = get_games_int_param(params, defaults, str(min_key), int(fallback_min))
    upper = get_games_int_param(params, defaults, str(max_key), int(fallback_max))
    if int(lower) > int(upper):
        raise ValueError(f"{min_key} must be <= {max_key}")
    return int(lower), int(upper)


def resolve_games_named_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    namespace: str,
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    supported_variants: Sequence[str],
) -> Tuple[str, Dict[str, float]]:
    """Resolve one balanced named visual or semantic axis for a games task."""

    sampling_namespace = str(namespace)
    rng = spawn_rng(int(instance_seed), sampling_namespace)
    selected, probabilities = resolve_variant(
        rng,
        params=params,
        gen_defaults=gen_defaults,
        supported_variants=[str(item) for item in supported_variants],
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
    )
    selected = apply_balanced_variant_sampling(
        instance_seed=int(instance_seed),
        params=params,
        gen_defaults=gen_defaults,
        selected_variant=str(selected),
        variant_probabilities=probabilities,
        supported_variants=[str(item) for item in supported_variants],
        balance_flag_key=str(balance_flag_key),
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        sampling_namespace=sampling_namespace,
    )
    return str(selected), dict(probabilities)


def resolve_games_integer_axis(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support_key: str,
    explicit_key: str,
    fallback_support: Sequence[int],
    namespace: str,
    balanced_flag_key: str,
) -> tuple[int, tuple[int, ...], Dict[str, float]]:
    """Resolve one integer-valued axis for games scene samplers."""

    support = resolve_integer_support(
        params,
        gen_defaults=gen_defaults,
        key=str(support_key),
        fallback=tuple(int(value) for value in fallback_support),
    )
    value, probabilities = resolve_integer_choice(
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
    return int(value), tuple(int(item) for item in support), dict(probabilities)


__all__ = [
    "get_games_int_param",
    "get_games_int_range",
    "resolve_games_integer_axis",
    "resolve_games_named_axis",
]
