"""Neutral dataset sampling primitives for the style-legend scene."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.label_assets import resolve_chart_axis_labels, resolve_chart_entity_labels
from trace_tasks.tasks.shared.config_defaults import group_default

from .defaults import (
    GEN_DEFAULTS,
    balanced_choice,
    count_from_range,
    gen_int,
    resolve_legend_position,
    resolve_palette_mode,
)
from .state import (
    LINE_STYLES,
    MARKER_FILLS,
    MARKER_SHAPES,
    RGB,
    SCENE_NAMESPACE,
    SeriesSpec,
    SeriesStyle,
    StyleLegendDataset,
)


@dataclass(frozen=True)
class StyleLegendSampleContext:
    """Scene-wide sampled support used by task-owned style-legend objectives."""

    task_params: dict[str, Any]
    x_count: int
    series_count: int
    labels_x: tuple[str, ...]
    x_label_meta: dict[str, Any]
    labels_series: tuple[str, ...]
    series_label_meta: dict[str, Any]
    palette_mode: str
    palette_mode_probabilities: dict[str, float]
    legend_position: str
    legend_position_probabilities: dict[str, float]
    styles: tuple[SeriesStyle, ...]
    value_min: int
    value_max: int
    series: tuple[SeriesSpec, ...]


def series_labels(params: Mapping[str, Any], *, instance_seed: int, count: int) -> tuple[tuple[str, ...], dict[str, Any]]:
    weights = params.get("series_label_bucket_weights", group_default(GEN_DEFAULTS, "series_label_bucket_weights", None))
    resolved = resolve_chart_entity_labels(
        spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.series_labels"),
        count=int(count),
        min_chars=int(gen_int(params, "series_label_min_chars", 2)),
        max_chars=int(gen_int(params, "series_label_max_chars", 7)),
        allow_spaces=False,
        bucket_weights=weights if isinstance(weights, Mapping) else None,
    )
    return tuple(str(label) for label in resolved.labels), {
        "label_variant": str(resolved.label_variant),
        "label_pool_kind": str(resolved.label_pool_kind),
        "label_source_kind": str(resolved.label_source_kind),
        "label_bucket": str(resolved.label_bucket),
        "label_manifest": str(resolved.label_manifest),
        "label_filter": dict(resolved.label_filter),
        "label_bucket_probabilities": dict(resolved.label_bucket_probabilities),
    }


def x_labels(params: Mapping[str, Any], *, instance_seed: int, count: int) -> tuple[tuple[str, ...], dict[str, Any]]:
    weights = params.get("x_label_bucket_weights", group_default(GEN_DEFAULTS, "x_label_bucket_weights", None))
    resolved = resolve_chart_axis_labels(
        spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.x_labels"),
        count=int(count),
        min_chars=int(gen_int(params, "x_label_min_chars", 2)),
        max_chars=int(gen_int(params, "x_label_max_chars", 6)),
        bucket_weights=weights if isinstance(weights, Mapping) else None,
    )
    return tuple(str(label) for label in resolved.labels), {
        "label_variant": str(resolved.label_variant),
        "label_pool_kind": str(resolved.label_pool_kind),
        "label_source_kind": str(resolved.label_source_kind),
        "label_bucket": str(resolved.label_bucket),
        "label_manifest": str(resolved.label_manifest),
        "label_filter": dict(resolved.label_filter),
        "label_bucket_probabilities": dict(resolved.label_bucket_probabilities),
    }


def colors_for_mode(mode: str, count: int) -> tuple[RGB, ...]:
    if str(mode) == "grayscale":
        base = (32, 48, 64, 82, 104, 126)
        return tuple((value, value, value) for value in base[: int(count)])
    if str(mode) == "muted_color":
        palette = ((64, 92, 134), (132, 89, 71), (78, 121, 94), (116, 92, 137), (148, 124, 68), (72, 124, 134))
        return tuple(palette[index % len(palette)] for index in range(int(count)))
    palette = ((0, 114, 178), (213, 94, 0), (0, 158, 115), (204, 121, 167), (230, 159, 0), (86, 180, 233))
    return tuple(palette[index % len(palette)] for index in range(int(count)))


def styles_for_series(*, count: int, palette_mode: str, instance_seed: int) -> tuple[SeriesStyle, ...]:
    colors = list(colors_for_mode(str(palette_mode), int(count)))
    style_tuples = [
        (line, marker, fill)
        for line in LINE_STYLES
        for marker in MARKER_SHAPES
        for fill in MARKER_FILLS
    ]
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.style_tuples")
    rng.shuffle(style_tuples)
    styles: list[SeriesStyle] = []
    used: set[tuple[str, str, str]] = set()
    for index in range(int(count)):
        line_style, marker_shape, marker_fill = style_tuples[int(index)]
        key = (str(line_style), str(marker_shape), str(marker_fill))
        if key in used:
            raise ValueError("duplicate style tuple")
        used.add(key)
        styles.append(
            SeriesStyle(
                color_rgb=tuple(colors[int(index) % len(colors)]),
                line_style=str(line_style),
                marker_shape=str(marker_shape),
                marker_fill=str(marker_fill),
                line_width_px=2 + (int(index) % 2),
            )
        )
    return tuple(styles)


def random_values(*, count: int, rng: Any, min_value: int, max_value: int) -> list[int]:
    current = int(rng.randint(28, 72))
    values: list[int] = []
    for _ in range(int(count)):
        current += int(rng.randint(-16, 16))
        current = max(int(min_value) + 5, min(int(max_value) - 5, int(current)))
        values.append(int(current))
    return values


def base_series(
    *,
    labels: Sequence[str],
    x_count: int,
    styles: Sequence[SeriesStyle],
    instance_seed: int,
    value_min: int,
    value_max: int,
) -> list[SeriesSpec]:
    series: list[SeriesSpec] = []
    for index, label in enumerate(labels):
        values = random_values(
            count=int(x_count),
            rng=spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.values.s{int(index)}"),
            min_value=int(value_min),
            max_value=int(value_max),
        )
        series.append(
            SeriesSpec(
                series_id=f"s{int(index)}",
                label=str(label),
                values=tuple(int(value) for value in values),
                style=styles[int(index)],
            )
        )
    return series


def replace_series_value(series: SeriesSpec, *, x_index: int, value: int) -> SeriesSpec:
    values = list(series.values)
    values[int(x_index)] = int(value)
    return SeriesSpec(series_id=str(series.series_id), label=str(series.label), values=tuple(values), style=series.style)


def replace_series_values(series: SeriesSpec, values: Sequence[int]) -> SeriesSpec:
    return SeriesSpec(
        series_id=str(series.series_id),
        label=str(series.label),
        values=tuple(int(value) for value in values),
        style=series.style,
    )


def common_setup(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    min_series_count: int = 4,
) -> tuple[int, int, tuple[str, ...], dict[str, Any], tuple[str, ...], dict[str, Any], str, dict[str, float], str, dict[str, float], tuple[SeriesStyle, ...]]:
    """Sample the scene-wide support shared by all objective-owned tasks."""

    x_count = count_from_range(
        params,
        min_key="style_legend_x_count_min",
        max_key="style_legend_x_count_max",
        fallback_min=5,
        fallback_max=9,
        instance_seed=int(instance_seed),
        namespace="x_count",
    )
    series_count = count_from_range(
        params,
        min_key="style_legend_series_count_min",
        max_key="style_legend_series_count_max",
        fallback_min=4,
        fallback_max=6,
        instance_seed=int(instance_seed),
        namespace="series_count",
    )
    series_count = max(int(min_series_count), int(series_count))
    sampled_x_labels, x_meta = x_labels(params, instance_seed=int(instance_seed), count=int(x_count))
    labels, label_meta = series_labels(params, instance_seed=int(instance_seed), count=int(series_count))
    palette_mode, palette_probs = resolve_palette_mode(params, instance_seed=int(instance_seed))
    legend_position, legend_probs = resolve_legend_position(params, instance_seed=int(instance_seed))
    styles = styles_for_series(count=int(series_count), palette_mode=str(palette_mode), instance_seed=int(instance_seed))
    return (
        int(x_count),
        int(series_count),
        tuple(sampled_x_labels),
        dict(x_meta),
        tuple(labels),
        dict(label_meta),
        str(palette_mode),
        dict(palette_probs),
        str(legend_position),
        dict(legend_probs),
        tuple(styles),
    )


def sample_context(params: Mapping[str, Any], *, instance_seed: int) -> StyleLegendSampleContext:
    """Sample scene support and base series without applying task objective constraints."""

    task_params = dict(params)
    (
        x_count,
        series_count,
        labels_x,
        meta_x,
        labels_series,
        meta_series,
        palette_mode,
        palette_probs,
        legend_position,
        legend_probs,
        styles,
    ) = common_setup(task_params, instance_seed=int(instance_seed))
    value_min = int(gen_int(task_params, "style_legend_value_min", 0))
    value_max = int(gen_int(task_params, "style_legend_value_max", 100))
    if int(value_min) >= int(value_max):
        raise ValueError("style_legend_value_min must be lower than style_legend_value_max")
    series = base_series(
        labels=labels_series,
        x_count=int(x_count),
        styles=styles,
        instance_seed=int(instance_seed),
        value_min=int(value_min),
        value_max=int(value_max),
    )
    return StyleLegendSampleContext(
        task_params=dict(task_params),
        x_count=int(x_count),
        series_count=int(series_count),
        labels_x=tuple(labels_x),
        x_label_meta=dict(meta_x),
        labels_series=tuple(labels_series),
        series_label_meta=dict(meta_series),
        palette_mode=str(palette_mode),
        palette_mode_probabilities=dict(palette_probs),
        legend_position=str(legend_position),
        legend_position_probabilities=dict(legend_probs),
        styles=tuple(styles),
        value_min=int(value_min),
        value_max=int(value_max),
        series=tuple(series),
    )


def dataset_from_context(
    context: StyleLegendSampleContext,
    *,
    series: Sequence[SeriesSpec],
    target_x_index: int,
    threshold_value: int | None = None,
    pair_series_ids: tuple[str, str] = (),
) -> StyleLegendDataset:
    return package_dataset(
        x_labels_value=context.labels_x,
        x_label_meta=context.x_label_meta,
        series=series,
        series_label_meta=context.series_label_meta,
        target_x_index=int(target_x_index),
        threshold_value=threshold_value,
        pair_series_ids=pair_series_ids,
        palette_mode=str(context.palette_mode),
        palette_mode_probabilities=context.palette_mode_probabilities,
        legend_position=str(context.legend_position),
        legend_position_probabilities=context.legend_position_probabilities,
    )


def package_dataset(
    *,
    x_labels_value: Sequence[str],
    x_label_meta: Mapping[str, Any],
    series: Sequence[SeriesSpec],
    series_label_meta: Mapping[str, Any],
    target_x_index: int,
    threshold_value: int | None,
    pair_series_ids: tuple[str, str] = (),
    palette_mode: str,
    palette_mode_probabilities: Mapping[str, float],
    legend_position: str,
    legend_position_probabilities: Mapping[str, float],
) -> StyleLegendDataset:
    return StyleLegendDataset(
        x_labels=tuple(str(label) for label in x_labels_value),
        x_label_meta=dict(x_label_meta),
        series=tuple(series),
        series_label_meta=dict(series_label_meta),
        target_x_index=int(target_x_index),
        threshold_value=None if threshold_value is None else int(threshold_value),
        pair_series_ids=tuple(str(value) for value in pair_series_ids),  # type: ignore[arg-type]
        palette_mode=str(palette_mode),
        palette_mode_probabilities={str(key): float(value) for key, value in palette_mode_probabilities.items()},
        legend_position=str(legend_position),
        legend_position_probabilities={str(key): float(value) for key, value in legend_position_probabilities.items()},
    )


def style_support_trace(series: Sequence[SeriesSpec]) -> list[dict[str, Any]]:
    return [
        {
            "series_id": str(item.series_id),
            "label": str(item.label),
            "values": [int(value) for value in item.values],
            "style": {
                "color_rgb": [int(channel) for channel in item.style.color_rgb],
                "line_style": str(item.style.line_style),
                "marker_shape": str(item.style.marker_shape),
                "marker_fill": str(item.style.marker_fill),
                "line_width_px": int(item.style.line_width_px),
            },
        }
        for item in series
    ]
