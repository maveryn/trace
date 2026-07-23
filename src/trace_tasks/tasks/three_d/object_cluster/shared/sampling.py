"""Identity-free sampling helpers for object-cluster objectives."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import normalize_positive_weights, weighted_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.three_d.shared.task_support import (
    resolve_axis_variant_for_namespace,
    resolve_support_choice_for_namespace,
)

from .defaults import ANSWER_COUNT_BINS, DEFAULT_ANSWER_COUNT_BIN_WEIGHTS


def uniform_string_probability_map(values: Sequence[str], *, selected: str | None = None) -> Dict[str, float]:
    """Build a probability map over string values, optionally locked to one value."""

    support = tuple(str(value) for value in values)
    if selected is not None:
        return {str(value): (1.0 if str(value) == str(selected) else 0.0) for value in support}
    probability = 1.0 / max(1, len(support))
    return {str(value): float(probability) for value in support}


def one_hot_int_probability_map(values: Sequence[int], *, selected: int) -> Dict[str, float]:
    """Build a deterministic probability map over integer support."""

    return {str(int(value)): (1.0 if int(value) == int(selected) else 0.0) for value in values}


def uniform_int_probability_map(values: Sequence[int]) -> Dict[str, float]:
    """Build a uniform probability map over integer support."""

    support = tuple(int(value) for value in values)
    if not support:
        raise ValueError("cannot build probability map over empty count support")
    probability = 1.0 / float(len(support))
    return {str(int(value)): float(probability) for value in support}


def configured_int(params: Mapping[str, Any], gen_defaults: Mapping[str, Any], key: str, default: int) -> int:
    """Resolve an integer from params first, then task generation defaults."""

    return int(params.get(str(key), group_default(gen_defaults, str(key), int(default))))


def count_bounds(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    minimum_key: str,
    maximum_key: str,
    fallback_minimum: int,
    fallback_maximum: int,
    lower: int,
    upper: int,
) -> Tuple[int, int]:
    """Resolve and clamp one integer answer/count support interval."""

    minimum = configured_int(params, gen_defaults, str(minimum_key), int(fallback_minimum))
    maximum = configured_int(params, gen_defaults, str(maximum_key), int(fallback_maximum))
    minimum = max(int(lower), min(int(upper), int(minimum)))
    maximum = max(int(minimum), min(int(upper), int(maximum)))
    return int(minimum), int(maximum)


def count_probability_map_from_bins(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    support: Sequence[int],
    weights_key: str,
) -> Dict[str, float]:
    """Resolve count probabilities from explicit weights or configured answer bins."""

    support_values = tuple(int(value) for value in support)
    if not support_values:
        raise ValueError("object cluster count support cannot be empty")
    supported_keys = {str(value) for value in support_values}
    raw_count_weights = params.get(str(weights_key), group_default(gen_defaults, str(weights_key), None))
    if isinstance(raw_count_weights, Mapping):
        weights = {
            str(key): float(value)
            for key, value in raw_count_weights.items()
            if str(key) in supported_keys
        }
        probabilities = normalize_positive_weights(weights, default_keys=[str(value) for value in support_values])
        return {str(value): float(probabilities.get(str(value), 0.0)) for value in support_values}

    raw_bin_weights = params.get(
        "answer_count_bin_weights",
        group_default(gen_defaults, "answer_count_bin_weights", DEFAULT_ANSWER_COUNT_BIN_WEIGHTS),
    )
    if not isinstance(raw_bin_weights, Mapping):
        raw_bin_weights = DEFAULT_ANSWER_COUNT_BIN_WEIGHTS
    bin_probabilities = normalize_positive_weights(
        {str(key): float(value) for key, value in raw_bin_weights.items()},
        default_keys=list(ANSWER_COUNT_BINS.keys()),
    )
    unnormalized: Dict[int, float] = {int(value): 0.0 for value in support_values}
    for bin_key, bin_probability in bin_probabilities.items():
        if str(bin_key) not in ANSWER_COUNT_BINS:
            continue
        lower, upper = ANSWER_COUNT_BINS[str(bin_key)]
        bin_support = [value for value in support_values if int(lower) <= int(value) <= int(upper)]
        if not bin_support:
            continue
        per_count_probability = float(bin_probability) / float(len(bin_support))
        for value in bin_support:
            unnormalized[int(value)] += float(per_count_probability)
    total = sum(float(value) for value in unnormalized.values())
    if total <= 0.0:
        return uniform_int_probability_map(support_values)
    return {str(value): float(unnormalized[int(value)] / total) for value in support_values}


def resolve_weighted_count(
    *,
    params: Mapping[str, Any],
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    explicit_key: str,
    weights_key: str,
    minimum: int,
    maximum: int,
    namespace: str,
) -> Tuple[int, Dict[str, float]]:
    """Resolve a count using explicit params or weighted answer-bin sampling."""

    support = tuple(range(int(minimum), int(maximum) + 1))
    if not support:
        raise ValueError(f"{explicit_key} resolved no supported counts")
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        selected = int(explicit)
        if int(selected) not in set(support):
            raise ValueError(f"unsupported {explicit_key}: {selected}")
        return int(selected), one_hot_int_probability_map(support, selected=int(selected))
    probabilities = count_probability_map_from_bins(
        params=params,
        gen_defaults=gen_defaults,
        support=support,
        weights_key=str(weights_key),
    )
    rng = spawn_rng(int(instance_seed), str(namespace))
    selected = int(weighted_choice(rng, probabilities, sort_keys=False))
    return int(selected), {str(value): float(probabilities[str(value)]) for value in support}


def resolve_uniform_count(
    *,
    params: Mapping[str, Any],
    explicit_key: str,
    minimum: int,
    maximum: int,
    instance_seed: int,
    namespace: str,
) -> Tuple[int, Dict[str, float]]:
    """Resolve a uniformly sampled count with optional explicit override."""

    support = tuple(range(int(minimum), int(maximum) + 1))
    if not support:
        raise ValueError(f"{explicit_key} resolved no supported counts")
    explicit = params.get(str(explicit_key))
    if explicit is not None:
        selected = int(explicit)
        if int(selected) not in set(support):
            raise ValueError(f"unsupported {explicit_key}: {selected}")
        return int(selected), one_hot_int_probability_map(support, selected=int(selected))
    rng = spawn_rng(int(instance_seed), str(namespace))
    probabilities = uniform_int_probability_map(support)
    selected = int(weighted_choice(rng, probabilities, sort_keys=False))
    return int(selected), probabilities


def resolve_scene_variant(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    instance_seed: int,
    support: Sequence[str],
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve the scene-grammar variant for one object-cluster instance."""

    return resolve_axis_variant_for_namespace(
        params,
        namespace=str(namespace),
        gen_defaults=gen_defaults,
        instance_seed=int(instance_seed),
        supported_variants=tuple(str(value) for value in support),
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
    )


def resolve_named_choice(
    *,
    params: Mapping[str, Any],
    key: str,
    support: Sequence[str],
    instance_seed: int,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    """Resolve one semantic label from explicit params or deterministic support indexing."""

    support_values = tuple(str(value) for value in support)
    selected, probabilities = resolve_support_choice_for_namespace(
        params=params,
        explicit_key=str(key),
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        support_values=support_values,
    )
    return str(selected), {str(key): float(value) for key, value in probabilities.items()}


def selected_probability_map(values: Sequence[str], selected_values: Sequence[str]) -> Dict[str, float]:
    """Build a probability map for an unordered subset selection."""

    support = tuple(str(value) for value in values)
    selected = {str(value) for value in selected_values}
    probability = 1.0 / max(1, len(selected))
    return {str(value): (float(probability) if str(value) in selected else 0.0) for value in support}


def resolve_string_subset(
    *,
    params: Mapping[str, Any],
    key: str,
    support: Sequence[str],
    instance_seed: int,
    namespace: str,
    count: int,
) -> Tuple[list[str], Dict[str, float]]:
    """Resolve a distinct semantic label subset from explicit params or sampling."""

    support_values = tuple(str(value) for value in support)
    explicit_value = params.get(str(key))
    if explicit_value is not None:
        if isinstance(explicit_value, str):
            selected_values = [value.strip() for value in explicit_value.split(",") if value.strip()]
        else:
            selected_values = [str(value) for value in explicit_value]
        if selected_values:
            if len(selected_values) != len(set(selected_values)):
                raise ValueError(f"{key} contains duplicate values")
            unsupported = [value for value in selected_values if value not in set(support_values)]
            if unsupported:
                raise ValueError(f"unsupported {key} values: {unsupported}")
            if len(selected_values) < int(count):
                raise ValueError(f"{key} requires at least {count} values")
            selected_values = selected_values[: int(count)]
            return list(selected_values), selected_probability_map(support_values, selected_values)

    ordered = list(support_values)
    if int(count) > len(ordered):
        raise ValueError(f"{key} requires at most {len(ordered)} values")
    rng = spawn_rng(int(instance_seed), str(namespace))
    rng.shuffle(ordered)
    selected = [str(value) for value in ordered[: int(count)]]
    return list(selected), selected_probability_map(support_values, selected)


__all__ = [
    "configured_int",
    "count_bounds",
    "one_hot_int_probability_map",
    "resolve_named_choice",
    "resolve_scene_variant",
    "resolve_string_subset",
    "resolve_uniform_count",
    "resolve_weighted_count",
    "selected_probability_map",
    "uniform_int_probability_map",
    "uniform_string_probability_map",
]
