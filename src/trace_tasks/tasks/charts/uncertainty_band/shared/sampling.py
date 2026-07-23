"""Neutral sampling helpers for uncertainty-band chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import sample_without_replacement, uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.label_assets import resolve_chart_axis_labels, resolve_chart_entity_labels
from trace_tasks.tasks.shared.color_distance import sample_color_palette_with_distance_constraints
from trace_tasks.tasks.shared.config_defaults import resolve_required_int_bounds

from .defaults import GEN_DEFAULTS, RENDER_DEFAULTS, generation_default, rendering_default
from .state import RGB, SCENE_NAMESPACE


@dataclass(frozen=True)
class BaseBandSample:
    x_count: int
    x_count_probabilities: Dict[str, float]
    x_labels: Tuple[str, ...]
    x_label_meta: Dict[str, Any]
    series_labels: Tuple[str, str]
    series_label_meta: Dict[str, Any]
    colors: Tuple[RGB, RGB]
    title: str


def probability_map(values: Sequence[int | str]) -> Dict[str, float]:
    support = tuple(str(value) for value in values)
    if not support:
        return {}
    weight = 1.0 / float(len(support))
    return {str(value): float(weight) for value in support}


def choose_from_values(
    params: Mapping[str, Any],
    *,
    values: Sequence[int | str],
    instance_seed: int,
    namespace: str,
) -> int | str:
    """Sample one value uniformly from an explicit support."""

    del params
    candidates = tuple(values)
    if not candidates:
        raise ValueError(f"empty support for {namespace}")
    rng = spawn_rng(int(instance_seed), str(namespace))
    return uniform_choice(rng, candidates)


def resolve_x_count(params: Mapping[str, Any], *, instance_seed: int) -> tuple[int, Dict[str, float]]:
    low, high = resolve_required_int_bounds(
        params,
        GEN_DEFAULTS,
        min_key="x_count_min",
        max_key="x_count_max",
        fallback_min=6,
        fallback_max=9,
        context="generation defaults for uncertainty_band",
    )
    support = tuple(range(int(low), int(high) + 1))
    return (
        int(
            choose_from_values(
                params,
                values=support,
                instance_seed=int(instance_seed),
                namespace=f"{SCENE_NAMESPACE}.x_count",
            )
        ),
        probability_map(support),
    )


def resolve_colors(params: Mapping[str, Any], *, instance_seed: int) -> tuple[RGB, RGB]:
    configured = params.get("series_palette_rgb", rendering_default("series_palette_rgb", None))
    if isinstance(configured, Sequence) and not isinstance(configured, (str, bytes)):
        colors: list[RGB] = []
        for raw in configured:
            if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)) and len(raw) >= 3:
                colors.append(tuple(max(0, min(255, int(channel))) for channel in raw[:3]))  # type: ignore[arg-type]
        if len(colors) >= 2:
            sampled = sample_without_replacement(
                spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.palette"),
                tuple(colors),
                2,
            )
            return tuple(sampled[0]), tuple(sampled[1])
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.palette")
    palette = sample_color_palette_with_distance_constraints(
        rng,
        palette_size=2,
        channel_min=20,
        channel_max=210,
        anchor_colors=((255, 255, 255), (20, 20, 20)),
        min_distance=50.0,
        distance_space="lab",
    )
    return tuple(palette[0]), tuple(palette[1])


def interval_from_mid_width(midpoint: int, width: int) -> tuple[int, int, int]:
    width = max(4, int(width))
    lower = int(midpoint) - int(width // 2)
    upper = int(lower) + int(width)
    if lower < 0:
        upper -= int(lower)
        lower = 0
    if upper > 100:
        lower -= int(upper) - 100
        upper = 100
    midpoint = int(round((int(lower) + int(upper)) / 2.0))
    return int(lower), int(midpoint), int(upper)


def sample_x_labels(*, count: int, instance_seed: int) -> tuple[tuple[str, ...], Dict[str, Any]]:
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.x_labels")
    resolved = resolve_chart_axis_labels(
        rng,
        count=int(count),
        min_chars=2,
        max_chars=6,
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


def sample_series_labels(*, instance_seed: int) -> tuple[tuple[str, str], Dict[str, Any]]:
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.series_labels")
    resolved = resolve_chart_entity_labels(rng, count=2, min_chars=3, max_chars=8, allow_spaces=False)
    labels = tuple(str(label) for label in resolved.labels)
    return (labels[0], labels[1]), {
        "label_source_kind": str(resolved.label_source_kind),
        "label_bucket": str(resolved.label_bucket),
        "label_manifest": str(resolved.label_manifest),
        "label_filter": dict(resolved.label_filter),
    }


def sample_base_band_scene(params: Mapping[str, Any], *, instance_seed: int) -> BaseBandSample:
    x_count, x_count_probabilities = resolve_x_count(params, instance_seed=int(instance_seed))
    x_labels, x_label_meta = sample_x_labels(count=int(x_count), instance_seed=int(instance_seed))
    series_labels, series_label_meta = sample_series_labels(instance_seed=int(instance_seed))
    colors = resolve_colors(params, instance_seed=int(instance_seed))
    title_options = params.get("title_options", rendering_default("title_options", ("Uncertainty Bands",)))
    if not isinstance(title_options, Sequence) or isinstance(title_options, (str, bytes)) or not title_options:
        title_options = ("Uncertainty Bands",)
    title = uniform_choice(
        spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.title"),
        tuple(str(option) for option in title_options),
    )
    return BaseBandSample(
        x_count=int(x_count),
        x_count_probabilities=dict(x_count_probabilities),
        x_labels=tuple(str(label) for label in x_labels),
        x_label_meta=dict(x_label_meta),
        series_labels=(str(series_labels[0]), str(series_labels[1])),
        series_label_meta=dict(series_label_meta),
        colors=colors,
        title=str(title),
    )


__all__ = [
    "BaseBandSample",
    "choose_from_values",
    "generation_default",
    "interval_from_mid_width",
    "probability_map",
    "sample_base_band_scene",
]
