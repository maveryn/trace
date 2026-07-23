"""Sampling-axis helpers for multiseries chart tasks."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Mapping, Tuple

from .....core.sampling import uniform_choice
from .....core.seed import hash64, spawn_rng
from ...shared.label_assets import resolve_chart_category_labels, resolve_chart_entity_labels
from ...shared.labeled_chart_variants import resolve_chart_axis_variant_for_namespace
from .defaults import (
    FAMILY_RANGE_KEYS,
    GEN_DEFAULTS,
    SCENE_NAMESPACE,
    SUPPORTED_CHANGE_DIRECTIONS,
    SUPPORTED_CHANGE_MEASURES,
    SUPPORTED_COMPARISONS,
    SUPPORTED_EXTREMUM_DIRECTIONS,
    SUPPORTED_RATIO_MEASURES,
)
from .state import SUPPORTED_MULTISERIES_CHART_SCENE_VARIANTS


def sample_series_labels(*, count: int, instance_seed: int) -> Tuple[str, ...]:
    """Sample one randomized series-label tuple."""

    if int(count) <= 0:
        raise ValueError("series count must be positive")
    rng = spawn_rng(int(instance_seed), "charts.multiseries.series_labels")
    resolved = resolve_chart_entity_labels(
        rng,
        count=int(count),
        min_chars=2,
        max_chars=4,
        allow_spaces=False,
    )
    return tuple(str(value).title() for value in resolved.labels)



def _sample_distinct_values(
    rng,
    *,
    count: int,
    value_min: int,
    value_max: int,
) -> List[int]:
    """Sample one distinct integer value list within the inclusive bounds."""

    universe = [int(value) for value in range(int(value_min), int(value_max) + 1)]
    if int(count) > len(universe):
        raise ValueError("distinct value sampling requires a larger value range")
    return [int(value) for value in rng.sample(universe, int(count))]


def resolve_change_measure(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
    return resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_CHANGE_MEASURES,
        namespace=f"{SCENE_NAMESPACE}.change_measure",
        explicit_key="change_measure",
        weights_key="change_measure_weights",
        balance_flag_key="balanced_change_measure_sampling",
    )


def resolve_ratio_measure(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
    return resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_RATIO_MEASURES,
        namespace=f"{SCENE_NAMESPACE}.ratio_measure",
        explicit_key="ratio_measure",
        weights_key="ratio_measure_weights",
        balance_flag_key="balanced_ratio_measure_sampling",
    )


def resolve_change_direction(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
    return resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_CHANGE_DIRECTIONS,
        namespace=f"{SCENE_NAMESPACE}.change_direction",
        explicit_key="change_direction",
        weights_key="change_direction_weights",
        balance_flag_key="balanced_change_direction_sampling",
    )


def resolve_extremum_direction(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
    return resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_EXTREMUM_DIRECTIONS,
        namespace=f"{SCENE_NAMESPACE}.extremum_direction",
        explicit_key="extremum_direction",
        weights_key="extremum_direction_weights",
        balance_flag_key="balanced_extremum_direction_sampling",
    )


def resolve_comparison(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
    return resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_COMPARISONS,
        namespace=f"{SCENE_NAMESPACE}.comparison",
        explicit_key="comparison",
        weights_key="comparison_weights",
        balance_flag_key="balanced_comparison_sampling",
    )


def resolve_scene_variant(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
    return resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_MULTISERIES_CHART_SCENE_VARIANTS,
        namespace=f"{SCENE_NAMESPACE}.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
    )


def params_for_variant_family(params: Mapping[str, Any], *, family: str) -> Dict[str, Any]:
    """Apply family-prefixed ranges while still honoring explicit standard keys."""

    resolved = dict(params)
    for key in FAMILY_RANGE_KEYS:
        if key in resolved:
            continue
        prefixed_key = f"{family}_{key}"
        if prefixed_key in resolved:
            resolved[key] = resolved[prefixed_key]
        elif prefixed_key in GEN_DEFAULTS:
            resolved[key] = GEN_DEFAULTS[prefixed_key]
    return resolved


def support_params_for_axis_cycle(
    params: Mapping[str, Any],
    *,
    probabilities: Mapping[str, float],
    supported_values: Tuple[str, ...],
    explicit_key: str,
    weights_key: str,
    balance_flag_key: str,
) -> Dict[str, Any]:
    """Use a per-query-parameter occurrence index for answer-support cycling."""

    support_params = dict(params)
    sampling_index = params.get("_sample_cursor")
    if sampling_index is None:
        return support_params
    if params.get(str(explicit_key)) is not None or params.get(str(weights_key)) is not None:
        return support_params
    enabled = bool(params.get(str(balance_flag_key), GEN_DEFAULTS.get(str(balance_flag_key), True)))
    if not bool(enabled):
        return support_params
    positives = [float(value) for value in probabilities.values() if float(value) > 0.0]
    if len(positives) != len(supported_values):
        return support_params
    if max(positives) - min(positives) > 1e-9:
        return support_params
    support_params["_sample_cursor"] = abs(int(sampling_index)) // max(1, len(supported_values))
    return support_params


def internal_pairwise_variant(comparison: str) -> str:
    if str(comparison) == "greater_than":
        return "series_a_gt_b_count"
    if str(comparison) == "less_than":
        return "series_a_lt_b_count"
    raise ValueError(f"unsupported comparison: {comparison}")


def internal_change_variant(
    *,
    change_measure: str,
    change_direction: str | None,
    extremum_direction: str | None,
) -> str:
    if str(change_measure) == "directional_change":
        if str(change_direction) == "increase":
            return "ranked_largest_increase"
        if str(change_direction) == "decrease":
            return "ranked_largest_decrease"
        raise ValueError(f"unsupported change_direction: {change_direction}")
    if str(change_measure) == "absolute_gap":
        if str(extremum_direction) == "largest":
            return "ranked_largest_gap"
        if str(extremum_direction) == "smallest":
            return "ranked_smallest_gap"
        raise ValueError(f"unsupported extremum_direction: {extremum_direction}")
    raise ValueError(f"unsupported change_measure: {change_measure}")


def internal_ratio_variant(*, ratio_measure: str, extremum_direction: str) -> str:
    if str(ratio_measure) == "series_share":
        if str(extremum_direction) == "largest":
            return "ranked_largest_series_share"
        if str(extremum_direction) == "smallest":
            return "ranked_smallest_series_share"
        raise ValueError(f"unsupported extremum_direction: {extremum_direction}")
    if str(ratio_measure) == "pair_ratio":
        if str(extremum_direction) == "largest":
            return "ranked_largest_pair_ratio"
        if str(extremum_direction) == "smallest":
            return "ranked_smallest_pair_ratio"
        raise ValueError(f"unsupported extremum_direction: {extremum_direction}")
    raise ValueError(f"unsupported ratio_measure: {ratio_measure}")


def _balanced_answer_label_target(
    *,
    namespace: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> str | None:
    sampling_index = params.get("_sample_cursor")
    if sampling_index is None:
        return None
    pool = [
        str(label)
        for label in resolve_chart_category_labels(
            random.Random(73_009 + hash64(0, str(namespace)) % 10_000),
            count=25,
            min_chars=2,
            max_chars=6,
            allow_spaces=False,
        ).labels
    ]
    return str(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{namespace}.answer_label.{abs(int(sampling_index))}"),
            tuple(pool),
        )
    )


def _remap_category_labels(value: Any, mapping: Mapping[str, str]) -> Any:
    if isinstance(value, str):
        return str(mapping.get(str(value), str(value)))
    if isinstance(value, list):
        return [_remap_category_labels(item, mapping) for item in value]
    if isinstance(value, tuple):
        return tuple(_remap_category_labels(item, mapping) for item in value)
    if isinstance(value, dict):
        return {
            str(mapping.get(str(key), str(key))): _remap_category_labels(item, mapping)
            for key, item in value.items()
        }
    return value


def balance_answer_label_for_indexed_probe(
    *,
    namespace: str,
    params: Mapping[str, Any],
    instance_seed: int,
    values_by_category: Mapping[str, Mapping[str, int]],
    answer_label: str,
    trace_extras: Mapping[str, Any],
) -> Tuple[Dict[str, Dict[str, int]], str, Dict[str, Any]]:
    """Relabel categories so indexed probes do not concentrate on one answer letter."""

    normalized_values = {
        str(category): {str(series): int(value) for series, value in series_values.items()}
        for category, series_values in values_by_category.items()
    }
    target_label = _balanced_answer_label_target(
        namespace=str(namespace),
        params=params,
        instance_seed=int(instance_seed),
    )
    if target_label is None or str(target_label) == str(answer_label):
        return normalized_values, str(answer_label), dict(trace_extras)
    category_labels = [str(label) for label in trace_extras.get("category_labels", normalized_values.keys())]
    mapping = {str(label): str(label) for label in category_labels}
    if str(target_label) in mapping:
        mapping[str(target_label)] = str(answer_label)
    mapping[str(answer_label)] = str(target_label)
    remapped_values = {
        str(mapping.get(str(category), str(category))): dict(series_values)
        for category, series_values in normalized_values.items()
    }
    remapped_trace = _remap_category_labels(dict(trace_extras), mapping)
    return remapped_values, str(target_label), dict(remapped_trace)


__all__ = [
    "_sample_distinct_values",
    "balance_answer_label_for_indexed_probe",
    "internal_change_variant",
    "internal_pairwise_variant",
    "internal_ratio_variant",
    "params_for_variant_family",
    "resolve_change_direction",
    "resolve_change_measure",
    "sample_series_labels",
    "resolve_comparison",
    "resolve_extremum_direction",
    "resolve_ratio_measure",
    "resolve_scene_variant",
    "support_params_for_axis_cycle",
]
