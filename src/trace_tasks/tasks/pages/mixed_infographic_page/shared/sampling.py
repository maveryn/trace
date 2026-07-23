"""Sampling helpers shared by mixed infographic page generation and queries."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ...shared.sampling import (
    resolve_int_support as resolve_pages_int_support,
    resolve_named_axis as resolve_pages_named_axis,
    resolve_supported_int as resolve_pages_supported_int,
)


def resolve_named_variant(
    *,
    sampling_namespace: str,
    gen_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    supported: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    return resolve_pages_named_axis(
        params=params,
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        namespace_root=sampling_namespace,
        supported=supported,
        explicit_key=explicit_key,
        weights_key=weights_key,
        balance_flag_key=balance_flag_key,
        namespace=namespace,
    )


def resolve_int_support(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
) -> Tuple[int, ...]:
    return resolve_pages_int_support(params=params, gen_defaults=gen_defaults, key=key, fallback=fallback)


def resolve_supported_int(
    *,
    sampling_namespace: str,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    explicit_key: str,
    support_key: str,
    fallback: Sequence[int],
    instance_seed: int,
    namespace: str,
) -> Tuple[int, Tuple[int, ...], Dict[str, float]]:
    return resolve_pages_supported_int(
        params=params,
        gen_defaults=gen_defaults,
        namespace_root=sampling_namespace,
        explicit_key=explicit_key,
        support_key=support_key,
        fallback=fallback,
        instance_seed=int(instance_seed),
        namespace=namespace,
    )


__all__ = [
    "resolve_int_support",
    "resolve_named_variant",
    "resolve_supported_int",
]
