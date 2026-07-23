"""Sampling helpers for heatmap chart tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....shared.config_defaults import group_default, resolve_required_int_bounds
from ...shared.balanced_sampling import balanced_int_from_support as _balanced_int
from ...shared.label_assets import resolve_chart_entity_labels
from ...shared.labeled_chart_variants import resolve_chart_axis_variant_for_namespace
from .defaults import (
    SCENE_NAMESPACE,
    GEN_DEFAULTS,
    SUPPORTED_EXTREMUM_DIRECTIONS,
    SUPPORTED_QUERY_AXES,
    SUPPORTED_SCENE_VARIANTS,
    _WEEKDAY_LABELS,
    _condition_support,
)


def _resolve_scene_variant(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
    return resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        namespace=f"{SCENE_NAMESPACE}.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
    )

def _resolve_condition_kind(
    params: Mapping[str, Any],
    *,
    scene_variant: str,
    instance_seed: int,
) -> Tuple[str, Dict[str, float]]:
    support = _condition_support(str(scene_variant))
    return resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=support,
        namespace=f"{SCENE_NAMESPACE}.condition_kind",
        explicit_key="condition_kind",
        weights_key="condition_kind_weights",
        balance_flag_key="balanced_condition_kind_sampling",
    )


def resolve_scene_condition_context(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
) -> Tuple[str, Dict[str, float], str, Dict[str, float]]:
    """Resolve the discrete scene variant and condition axis for condition-based tasks."""

    scene_variant, scene_probabilities = _resolve_scene_variant(params, instance_seed=int(instance_seed))
    condition_params = _decoupled_sampling_params(
        params,
        divisor=2,
        explicit_keys=("condition_kind", "condition_kind_weights"),
    )
    condition_kind, condition_probabilities = _resolve_condition_kind(
        condition_params,
        scene_variant=str(scene_variant),
        instance_seed=int(instance_seed),
    )
    return str(scene_variant), dict(scene_probabilities), str(condition_kind), dict(condition_probabilities)


def _resolve_extremum_direction(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
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


def _resolve_query_axis(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
    return resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_QUERY_AXES,
        namespace=f"{SCENE_NAMESPACE}.query_axis",
        explicit_key="query_axis",
        weights_key="query_axis_weights",
        balance_flag_key="balanced_query_axis_sampling",
    )


def _decoupled_sampling_params(params: Mapping[str, Any], *, divisor: int, explicit_keys: Sequence[str]) -> Dict[str, Any]:
    resolved = dict(params)
    sampling_index = resolved.get("_sample_cursor")
    if sampling_index is None or any(resolved.get(str(key)) is not None for key in explicit_keys):
        return resolved
    resolved["_sample_cursor"] = abs(int(sampling_index)) // max(1, int(divisor))
    return resolved


def require_discrete_heatmap(params: Mapping[str, Any], *, task_label: str) -> None:
    """Reject the continuous colorbar variant for discrete heatmap objectives."""

    if str(params.get("scene_variant") or "") == "continuous_colorbar_heatmap":
        raise ValueError(f"{task_label} heatmap tasks do not support continuous_colorbar_heatmap")


def _resolve_row_column_count(
    params: Mapping[str, Any],
    *,
    scene_variant: str,
    instance_seed: int,
) -> Tuple[int, int, Dict[str, float], Dict[str, float]]:
    """Resolve row/column supports for the selected heatmap visual variant."""

    if str(scene_variant) == "continuous_colorbar_heatmap":
        row_min, row_max = resolve_required_int_bounds(
            params,
            GEN_DEFAULTS,
            min_key="colorbar_row_count_min",
            max_key="colorbar_row_count_max",
            fallback_min=5,
            fallback_max=7,
            context=f"{SCENE_NAMESPACE} continuous colorbar rows",
        )
        col_min, col_max = resolve_required_int_bounds(
            params,
            GEN_DEFAULTS,
            min_key="colorbar_column_count_min",
            max_key="colorbar_column_count_max",
            fallback_min=6,
            fallback_max=8,
            context=f"{SCENE_NAMESPACE} continuous colorbar columns",
        )
        row_support = list(range(int(row_min), int(row_max) + 1))
        col_support = list(range(int(col_min), int(col_max) + 1))
        row_count = _balanced_int(row_support, params=params, instance_seed=int(instance_seed), namespace=f"{SCENE_NAMESPACE}.colorbar_row_count")
        col_count = _balanced_int(col_support, params=params, instance_seed=int(instance_seed), namespace=f"{SCENE_NAMESPACE}.colorbar_column_count")
        return (
            int(row_count),
            int(col_count),
            {str(value): 1.0 / float(len(row_support)) for value in row_support},
            {str(value): 1.0 / float(len(col_support)) for value in col_support},
        )

    if str(scene_variant) == "calendar_heatmap":
        row_min, row_max = resolve_required_int_bounds(
            params,
            GEN_DEFAULTS,
            min_key="calendar_row_count_min",
            max_key="calendar_row_count_max",
            fallback_min=5,
            fallback_max=8,
            context=f"{SCENE_NAMESPACE} calendar rows",
        )
        row_support = list(range(int(row_min), int(row_max) + 1))
        row_count = _balanced_int(
            row_support,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.calendar_row_count",
        )
        return (
            int(row_count),
            7,
            {str(value): 1.0 / float(len(row_support)) for value in row_support},
            {"7": 1.0},
        )

    row_min, row_max = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key="row_count_min",
        max_key="row_count_max",
        fallback_min=6,
        fallback_max=10,
        context=f"{SCENE_NAMESPACE} rows",
    )
    col_min, col_max = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key="column_count_min",
        max_key="column_count_max",
        fallback_min=8,
        fallback_max=12,
        context=f"{SCENE_NAMESPACE} columns",
    )
    row_support = list(range(int(row_min), int(row_max) + 1))
    col_support = list(range(int(col_min), int(col_max) + 1))
    row_count = _balanced_int(row_support, params=params, instance_seed=int(instance_seed), namespace=f"{SCENE_NAMESPACE}.row_count")
    col_count = _balanced_int(col_support, params=params, instance_seed=int(instance_seed), namespace=f"{SCENE_NAMESPACE}.column_count")
    return (
        int(row_count),
        int(col_count),
        {str(value): 1.0 / float(len(row_support)) for value in row_support},
        {str(value): 1.0 / float(len(col_support)) for value in col_support},
    )


def _labels_for_scene(
    *,
    scene_variant: str,
    row_count: int,
    column_count: int,
    rng,
) -> Tuple[List[str], List[str]]:
    if str(scene_variant) == "calendar_heatmap":
        return [f"Week {index + 1}" for index in range(int(row_count))], list(_WEEKDAY_LABELS[: int(column_count)])
    labels = list(
        resolve_chart_entity_labels(
            rng,
            count=int(row_count) + int(column_count),
            min_chars=2,
            max_chars=7,
            allow_spaces=False,
        ).labels
    )
    rows = labels[: int(row_count)]
    columns = labels[int(row_count) : int(row_count) + int(column_count)]
    return [str(label) for label in rows], [str(label) for label in columns]

def _colorbar_ticks(params: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = params.get("colorbar_ticks", group_default(GEN_DEFAULTS, "colorbar_ticks", tuple(range(0, 101, 10))))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = tuple(range(0, 101, 10))
    ticks = tuple(sorted({max(0, min(100, int(value))) for value in raw}))
    return ticks if ticks else tuple(range(0, 101, 10))


def _colorbar_threshold_values(params: Mapping[str, Any]) -> Tuple[int, ...]:
    raw = params.get("colorbar_threshold_values", group_default(GEN_DEFAULTS, "colorbar_threshold_values", (30, 40, 50, 60, 70)))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        raw = (30, 40, 50, 60, 70)
    values = tuple(value for value in sorted({max(5, min(95, int(item))) for item in raw}) if 0 < int(value) < 100)
    return values if values else (30, 40, 50, 60, 70)


def _colorbar_interval_bounds(params: Mapping[str, Any]) -> Tuple[Tuple[int, int], ...]:
    ticks = _colorbar_ticks(params)
    min_width = int(params.get("colorbar_interval_min_width", group_default(GEN_DEFAULTS, "colorbar_interval_min_width", 20)))
    max_width = int(params.get("colorbar_interval_max_width", group_default(GEN_DEFAULTS, "colorbar_interval_max_width", 30)))
    pairs = [
        (int(lower), int(upper))
        for lower in ticks
        for upper in ticks
        if int(lower) < int(upper) and int(min_width) <= int(upper) - int(lower) <= int(max_width)
    ]
    return tuple(pairs) if pairs else ((30, 60), (40, 70), (20, 50))
