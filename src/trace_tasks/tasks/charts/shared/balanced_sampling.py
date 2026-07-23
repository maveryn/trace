"""Balanced sampling helpers for chart scene supports."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ....core.sampling import uniform_choice
from ....core.seed import spawn_rng


def uses_uniform_query_id_cycle(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    query_id_probabilities: Mapping[str, float],
    supported_query_ids: Sequence[str],
    explicit_key: str = "query_id",
    weights_key: str = "query_id_weights",
    balance_flag_key: str = "balanced_query_id_sampling",
) -> bool:
    """Return whether support sampling should decouple from query-id cycling."""

    if params.get(str(explicit_key)) is not None or params.get(str(weights_key)) is not None:
        return False
    enabled = bool(params.get(str(balance_flag_key), gen_defaults.get(str(balance_flag_key), True)))
    if not enabled:
        return False
    positives = [float(value) for value in query_id_probabilities.values() if float(value) > 0.0]
    if len(positives) != len(tuple(supported_query_ids)):
        return False
    return max(positives) - min(positives) <= 1e-9


def support_sampling_params_for_uniform_query_cycle(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    query_id_probabilities: Mapping[str, float],
    supported_query_ids: Sequence[str],
) -> Dict[str, Any]:
    """Decouple `_sample_cursor` from uniform public query-id cycling."""

    support_params = dict(params)
    sampling_index = support_params.get("_sample_cursor")
    if sampling_index is None:
        return support_params
    if not uses_uniform_query_id_cycle(
        params,
        gen_defaults=gen_defaults,
        query_id_probabilities=query_id_probabilities,
        supported_query_ids=supported_query_ids,
    ):
        return support_params
    support_params["_sample_cursor"] = abs(int(sampling_index)) // max(1, len(tuple(supported_query_ids)))
    return support_params


def decouple_sample_cursor_for_axis_lengths(
    params: Mapping[str, Any],
    *,
    axes: Sequence[Tuple[int, Sequence[str]]],
    explicit_policy: str = "present",
    use_abs: bool = False,
) -> Dict[str, Any]:
    """Decouple `_sample_cursor` by fixed axis lengths when axes are implicit."""

    support_params = dict(params)
    if "_sample_cursor" not in support_params:
        return support_params
    divisor = 1
    for axis_length, explicit_keys in axes:
        keys = tuple(str(key) for key in explicit_keys)
        if str(explicit_policy) == "present":
            axis_explicit = any(key in support_params for key in keys)
        elif str(explicit_policy) == "non_null":
            axis_explicit = any(support_params.get(key) is not None for key in keys)
        else:
            raise ValueError(f"unsupported explicit_policy: {explicit_policy}")
        if not axis_explicit:
            divisor *= max(1, int(axis_length))
    cursor = abs(int(support_params["_sample_cursor"])) if bool(use_abs) else int(support_params["_sample_cursor"])
    support_params["_sample_cursor"] = int(cursor) // max(1, int(divisor))
    return support_params


def balanced_int_from_support(
    support: Sequence[int],
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> int:
    """Select one integer from an ordered support using seeded RNG."""

    ordered = [int(value) for value in support]
    if not ordered:
        raise ValueError(f"empty support for {namespace}")
    rng = spawn_rng(int(instance_seed), str(namespace))
    return int(uniform_choice(rng, ordered, sort_keys=True))


__all__ = [
    "balanced_int_from_support",
    "decouple_sample_cursor_for_axis_lengths",
    "support_sampling_params_for_uniform_query_cycle",
    "uses_uniform_query_id_cycle",
]
