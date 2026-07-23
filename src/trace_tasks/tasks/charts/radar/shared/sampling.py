"""Scene-neutral sampling primitives for radar chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Mapping, Sequence
from typing import Any

from .....core.sampling import uniform_choice
from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default, resolve_required_int_bounds
from ...shared.label_assets import (
    resolve_chart_entity_labels,
    resolve_chart_panel_labels,
    validate_chart_label_namespaces,
)

from .defaults import GEN_DEFAULTS, PANEL_LABEL_SUPPORT, profile_palette, resolve_gen_int
from .state import RadarPanel, RadarProfile


@dataclass(frozen=True)
class RadarSmallMultipleFrame:
    value_min: int
    value_max: int
    metrics: tuple[str, ...]
    panel_labels: tuple[str, ...]
    panel_label_meta: dict[str, Any]
    values: dict[str, dict[str, int]]


def choose_index(count: int, params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> int:
    """Sample one zero-based index from a finite support."""

    if int(count) <= 0:
        raise ValueError(f"empty index support for {namespace}")
    return int(
        uniform_choice(
            spawn_rng(int(instance_seed), str(namespace)),
            tuple(range(int(count))),
            sort_keys=True,
        )
    )


def without_sample_cursor(params: Mapping[str, Any]) -> dict[str, Any]:
    derived_params = dict(params)
    derived_params.pop("_sample_cursor", None)
    return derived_params


def balanced_choice(values: Sequence[int], params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> int:
    support = [int(value) for value in values]
    if not support:
        raise ValueError(f"empty support for {namespace}")
    return int(
        uniform_choice(
            spawn_rng(int(instance_seed), str(namespace)),
            support,
            sort_keys=True,
        )
    )


def metric_count(params: Mapping[str, Any], *, min_required: int = 1, instance_seed: int, namespace: str) -> int:
    low, high = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key="metric_count_min",
        max_key="metric_count_max",
        fallback_min=5,
        fallback_max=7,
        context="generation defaults for radar metric count",
    )
    low = max(int(low), int(min_required))
    if int(low) > int(high):
        raise ValueError("metric_count support is too small for requested radar scene")
    return balanced_choice(
        list(range(int(low), int(high) + 1)),
        params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def axis_metric_count(params: Mapping[str, Any], *, min_required: int = 1, instance_seed: int, namespace: str) -> int:
    low = resolve_gen_int(params, "axis_metric_count_min", resolve_gen_int(params, "metric_count_min", 5))
    high = resolve_gen_int(params, "axis_metric_count_max", resolve_gen_int(params, "metric_count_max", 7))
    low = max(int(low), int(min_required))
    if int(low) > int(high):
        raise ValueError("axis metric count support is too small for requested radar scene")
    return balanced_choice(
        list(range(int(low), int(high) + 1)),
        params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def panel_count(params: Mapping[str, Any], *, min_required: int = 1, instance_seed: int, namespace: str) -> int:
    low, high = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key="panel_count_min",
        max_key="panel_count_max",
        fallback_min=5,
        fallback_max=8,
        context="generation defaults for radar panel count",
    )
    low = max(int(low), int(min_required))
    if int(low) > int(high):
        raise ValueError("panel_count support is too small for requested radar scene")
    return balanced_choice(
        list(range(int(low), int(high) + 1)),
        params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def axis_panel_count(params: Mapping[str, Any], *, min_required: int = 1, instance_seed: int, namespace: str) -> int:
    low = resolve_gen_int(params, "axis_panel_count_min", resolve_gen_int(params, "panel_count_min", 5))
    high = resolve_gen_int(params, "axis_panel_count_max", resolve_gen_int(params, "panel_count_max", 8))
    low = max(int(low), int(min_required))
    high = min(len(PANEL_LABEL_SUPPORT), int(high))
    if int(low) > int(high):
        raise ValueError("axis panel count support is too small for requested radar scene")
    return balanced_choice(
        list(range(int(low), int(high) + 1)),
        params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def target_count_support(params: Mapping[str, Any], *, upper: int) -> list[int]:
    low = resolve_gen_int(params, "target_count_min", 1)
    high = resolve_gen_int(params, "target_count_max", 6)
    low = max(1, int(low))
    high = min(int(high), int(upper))
    if int(low) > int(high):
        raise ValueError("target count support is empty")
    return [int(value) for value in range(int(low), int(high) + 1)]


def sample_metrics(count: int, *, instance_seed: int, namespace: str) -> tuple[str, ...]:
    rng = spawn_rng(int(instance_seed), str(namespace))
    labels = resolve_chart_entity_labels(
        rng,
        count=int(count),
        min_chars=2,
        max_chars=8,
        allow_spaces=False,
    ).labels
    return tuple(str(label) for label in labels)


def resolved_label_metadata(resolved: Any) -> dict[str, Any]:
    return {
        "label_variant": str(resolved.label_variant),
        "label_pool_kind": str(resolved.label_pool_kind),
        "label_source_kind": str(resolved.label_source_kind),
        "label_bucket": str(resolved.label_bucket),
        "label_manifest": str(resolved.label_manifest),
        "label_filter": dict(resolved.label_filter),
        "label_bucket_probabilities": dict(resolved.label_bucket_probabilities),
    }


def sample_panel_labels(
    count: int,
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    reserved_labels: Sequence[str],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    resolved = resolve_chart_panel_labels(
        spawn_rng(int(instance_seed), str(namespace)),
        count=int(count),
        min_chars=1,
        max_chars=10,
        allow_spaces=False,
        variant_weights=params.get(
            "panel_label_variant_weights",
            group_default(
                GEN_DEFAULTS,
                "panel_label_variant_weights",
                {
                    "subplot_letters": 1.0,
                    "named_compact": 0.75,
                    "technical_topics": 0.75,
                    "condition_labels": 0.5,
                    "temporal_sequence": 0.25,
                },
            ),
        ),
        reserved_labels=tuple(str(label) for label in reserved_labels),
    )
    collision_check = validate_chart_label_namespaces(
        panel_labels=resolved.labels,
        other_label_groups={"metric_or_profile_labels": tuple(str(label) for label in reserved_labels)},
        context="radar panel labels",
    )
    return tuple(str(label) for label in resolved.labels), {
        "panel_label_resolution": resolved_label_metadata(resolved),
        "panel_label_collision_check": dict(collision_check),
    }


def value_bounds(params: Mapping[str, Any]) -> tuple[int, int]:
    low = resolve_gen_int(params, "value_min", 1)
    high = resolve_gen_int(params, "value_max", 10)
    if int(low) >= int(high):
        raise ValueError("value_min must be lower than value_max")
    return int(low), int(high)


def threshold_value(params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> int:
    low = resolve_gen_int(params, "threshold_min", 4)
    high = resolve_gen_int(params, "threshold_max", 7)
    if int(low) > int(high):
        raise ValueError("threshold support is empty")
    return balanced_choice(
        list(range(int(low), int(high) + 1)),
        params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )


def make_random_panel_values(
    *,
    metrics: Sequence[str],
    panel_labels: Sequence[str],
    value_min: int,
    value_max: int,
    instance_seed: int,
    namespace: str,
) -> dict[str, dict[str, int]]:
    rng = spawn_rng(int(instance_seed), str(namespace))
    return {
        str(panel): {
            str(metric): int(rng.randint(int(value_min), int(value_max)))
            for metric in metrics
        }
        for panel in panel_labels
    }


def shuffled_subset(values: Sequence[str], count: int, rng: Any) -> tuple[str, ...]:
    candidates = [str(value) for value in values]
    rng.shuffle(candidates)
    return tuple(candidates[: int(count)])


def force_panel_threshold_by_metric(
    *,
    values: dict[str, dict[str, int]],
    panel_labels: Sequence[str],
    metric_label: str,
    matching_panel_labels: Sequence[str],
    threshold: int,
    value_min: int,
    value_max: int,
    rng: Any,
) -> None:
    matching = set(str(label) for label in matching_panel_labels)
    for panel in panel_labels:
        if str(panel) in matching:
            values[str(panel)][str(metric_label)] = int(rng.randint(int(threshold) + 1, int(value_max)))
        else:
            values[str(panel)][str(metric_label)] = int(rng.randint(int(value_min), int(threshold)))


def force_metric_threshold_by_panel(
    *,
    values: dict[str, dict[str, int]],
    metrics: Sequence[str],
    panel_label: str,
    matching_metric_labels: Sequence[str],
    threshold: int,
    value_min: int,
    value_max: int,
    rng: Any,
) -> None:
    matching = set(str(label) for label in matching_metric_labels)
    for metric in metrics:
        if str(metric) in matching:
            values[str(panel_label)][str(metric)] = int(rng.randint(int(threshold) + 1, int(value_max)))
        else:
            values[str(panel_label)][str(metric)] = int(rng.randint(int(value_min), int(threshold)))


def sample_small_multiple_frame(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    metric_count: int,
    panel_count: int,
    namespace: str,
    metric_seed_offset: int = 0,
    reserved_profile_labels: Sequence[str] = ("Profile",),
) -> RadarSmallMultipleFrame:
    value_min, value_max = value_bounds(params)
    metrics = sample_metrics(
        int(metric_count),
        instance_seed=int(instance_seed) + int(metric_seed_offset),
        namespace=f"{namespace}.metric_labels",
    )
    panel_labels, panel_label_meta = sample_panel_labels(
        int(panel_count),
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.panel_labels",
        reserved_labels=tuple(metrics) + tuple(str(label) for label in reserved_profile_labels),
    )
    values = make_random_panel_values(
        metrics=metrics,
        panel_labels=panel_labels,
        value_min=int(value_min),
        value_max=int(value_max),
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.base_values",
    )
    return RadarSmallMultipleFrame(
        value_min=int(value_min),
        value_max=int(value_max),
        metrics=tuple(metrics),
        panel_labels=tuple(panel_labels),
        panel_label_meta=dict(panel_label_meta),
        values=values,
    )


def make_single_profile_panels(
    *,
    panel_labels: Sequence[str],
    values_by_panel: Mapping[str, Mapping[str, int]],
    params: Mapping[str, Any],
) -> tuple[RadarPanel, ...]:
    colors = profile_palette(params)
    return tuple(
        RadarPanel(
            panel_label=str(panel),
            profiles=(
                RadarProfile(
                    profile_label="Profile",
                    values={str(metric): int(value) for metric, value in values_by_panel[str(panel)].items()},
                    color_rgb=tuple(colors[index % len(colors)]),
                ),
            ),
        )
        for index, panel in enumerate(panel_labels)
    )
