"""Neutral data sampling primitives for error-bar series scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.errorbar_series.shared.defaults import (
    GEN_DEFAULTS,
    as_rgb,
    choose_from_values,
    config_context_key,
    group_generation_default,
    group_render_default,
    resolve_count,
)
from trace_tasks.tasks.charts.errorbar_series.shared.state import (
    RGB,
    SCENE_NAMESPACE,
    SUPPORTED_SCENE_VARIANTS,
    ErrorbarSeries,
)
from trace_tasks.tasks.charts.shared.label_assets import resolve_chart_axis_labels, resolve_chart_entity_labels
from trace_tasks.tasks.charts.shared.labeled_chart_variants import resolve_chart_axis_variant


@dataclass(frozen=True)
class ErrorbarBaseScene:
    """Scene-level sampled values shared by all error-bar objectives."""

    x_count: int
    series_count: int
    x_labels: Tuple[str, ...]
    x_label_meta: Dict[str, Any]
    series_labels: Tuple[str, ...]
    series_label_meta: Dict[str, Any]
    scene_variant: str
    scene_variant_probabilities: Dict[str, float]
    title: str


def resolve_scene_variant(params: Mapping[str, Any], *, instance_seed: int) -> tuple[str, dict[str, float]]:
    """Sample a rendering variant without depending on public task identity."""

    return resolve_chart_axis_variant(
        params=params,
        gen_defaults=GEN_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        **{config_context_key(): SCENE_NAMESPACE},
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
        axis_namespace="scene_variant",
    )


def sample_labels(
    params: Mapping[str, Any],
    *,
    x_count: int,
    series_count: int,
    instance_seed: int,
) -> tuple[tuple[str, ...], dict[str, Any], tuple[str, ...], dict[str, Any]]:
    """Sample x-axis and series labels from the chart label asset pools."""

    label_rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.labels")
    x_bucket_weights = params.get("x_label_bucket_weights", group_generation_default("x_label_bucket_weights", None))
    x_resolved = resolve_chart_axis_labels(
        label_rng,
        count=int(x_count),
        min_chars=int(params.get("x_label_min_chars", group_generation_default("x_label_min_chars", 2))),
        max_chars=int(params.get("x_label_max_chars", group_generation_default("x_label_max_chars", 6))),
        bucket_weights=x_bucket_weights if isinstance(x_bucket_weights, Mapping) else None,
    )
    series_resolved = resolve_chart_entity_labels(
        label_rng,
        count=int(series_count),
        min_chars=2,
        max_chars=8,
        allow_spaces=False,
    )
    return (
        tuple(str(label) for label in x_resolved.labels),
        {
            "label_source_kind": str(x_resolved.label_source_kind),
            "label_bucket": str(x_resolved.label_bucket),
            "label_manifest": str(x_resolved.label_manifest),
            "label_filter": dict(x_resolved.label_filter),
            "label_bucket_probabilities": dict(x_resolved.label_bucket_probabilities),
        },
        tuple(str(label) for label in series_resolved.labels),
        {
            "label_source_kind": str(series_resolved.label_source_kind),
            "label_bucket": str(series_resolved.label_bucket),
            "label_manifest": str(series_resolved.label_manifest),
            "label_filter": dict(series_resolved.label_filter),
            "label_bucket_probabilities": dict(series_resolved.label_bucket_probabilities),
        },
    )


def sample_title(params: Mapping[str, Any], *, instance_seed: int) -> str:
    """Sample one optional chart title from rendering defaults."""

    raw_titles = params.get("title_options", group_render_default("title_options", ()))
    titles = tuple(str(value) for value in raw_titles) if isinstance(raw_titles, Sequence) and not isinstance(raw_titles, (str, bytes)) else ()
    if not titles:
        titles = ("Error-Bar Series", "Scientific Estimate Profile", "Point Estimate Ranges")
    return str(
        choose_from_values(
            params,
            values=titles,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.title",
        )
    )


def sample_base_scene(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    series_mode: str,
) -> ErrorbarBaseScene:
    """Sample labels, counts, title, and render variant for one objective."""

    x_count = resolve_count(
        params,
        min_key="x_count_min",
        max_key="x_count_max",
        fallback_min=5,
        fallback_max=8,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.x_count",
    )
    if str(series_mode) == "overlap":
        series_count = resolve_count(
            params,
            min_key="overlap_series_count_min",
            max_key="overlap_series_count_max",
            fallback_min=3,
            fallback_max=4,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.overlap_series_count",
        )
    else:
        series_count = resolve_count(
            params,
            min_key="series_count_min",
            max_key="series_count_max",
            fallback_min=2,
            fallback_max=4,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.series_count",
        )
    x_labels, x_meta, series_labels, series_meta = sample_labels(
        params,
        x_count=int(x_count),
        series_count=int(series_count),
        instance_seed=int(instance_seed),
    )
    scene_variant, scene_probs = resolve_scene_variant(params, instance_seed=int(instance_seed))
    return ErrorbarBaseScene(
        x_count=int(x_count),
        series_count=int(series_count),
        x_labels=tuple(x_labels),
        x_label_meta=dict(x_meta),
        series_labels=tuple(series_labels),
        series_label_meta=dict(series_meta),
        scene_variant=str(scene_variant),
        scene_variant_probabilities=dict(scene_probs),
        title=sample_title(params, instance_seed=int(instance_seed)),
    )


def palette(params: Mapping[str, Any]) -> Tuple[RGB, ...]:
    """Resolve the scene palette used for all series."""

    default_colors: tuple[RGB, ...] = (
        (47, 111, 196),
        (216, 104, 62),
        (42, 147, 112),
        (126, 93, 190),
        (202, 138, 4),
    )
    raw_palette = params.get("series_palette_rgb", group_render_default("series_palette_rgb", ()))
    colors: List[RGB] = []
    if isinstance(raw_palette, Sequence) and not isinstance(raw_palette, (str, bytes)):
        for item in raw_palette:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 3:
                colors.append(as_rgb(item, (0, 0, 0)))
    if not colors:
        colors = list(default_colors)
    for fallback in default_colors:
        if len(colors) >= len(default_colors):
            break
        if fallback not in colors:
            colors.append(fallback)
    return tuple(colors)


def random_interval(rng: Any, *, low_min: int = 10, high_max: int = 90) -> tuple[int, int, int]:
    """Sample one lower/mid/upper interval triple."""

    mid = int(rng.randint(int(low_min) + 10, int(high_max) - 10))
    err_low = int(rng.randint(5, 13))
    err_high = int(rng.randint(5, 13))
    lower = max(0, int(mid) - int(err_low))
    upper = min(100, int(mid) + int(err_high))
    return int(lower), int(mid), int(upper)


def random_series_triples(rng: Any, *, x_count: int) -> list[tuple[int, int, int]]:
    """Sample one random interval triple per x position."""

    return [random_interval(rng) for _ in range(int(x_count))]


def make_series(
    *,
    series_id: str,
    label: str,
    color_rgb: RGB,
    triples: Sequence[tuple[int, int, int]],
) -> ErrorbarSeries:
    """Build one immutable series record from interval triples."""

    return ErrorbarSeries(
        series_id=str(series_id),
        label=str(label),
        color_rgb=tuple(int(value) for value in color_rgb),
        lower_values=tuple(int(triple[0]) for triple in triples),
        mid_values=tuple(int(triple[1]) for triple in triples),
        upper_values=tuple(int(triple[2]) for triple in triples),
    )


__all__ = [
    "ErrorbarBaseScene",
    "make_series",
    "palette",
    "random_interval",
    "random_series_triples",
    "resolve_scene_variant",
    "sample_base_scene",
    "sample_labels",
    "sample_title",
]
