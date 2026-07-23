"""Neutral sampling primitives for contour-density chart scenes."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default, resolve_required_int_bounds
from trace_tasks.tasks.charts.shared.label_assets import resolve_chart_entity_labels
from trace_tasks.tasks.charts.shared.labeled_chart_variants import resolve_chart_axis_variant
from trace_tasks.tasks.charts.shared.visual_defaults import coerce_rgb
from trace_tasks.tasks.charts.contour_density.shared.defaults import (
    CONFIG_CONTEXT_KEY,
    GENERATION_DEFAULTS,
    RENDER_DEFAULTS,
    SCENE_NAMESPACE,
    SUPPORTED_SCENE_VARIANTS,
)
from trace_tasks.tasks.charts.contour_density.shared.state import RGB, Region, Reference


def as_rgb(value: Any, fallback: RGB) -> RGB:
    return coerce_rgb(value, fallback)


def bbox(values: Sequence[float]) -> List[float]:
    return [round(float(value), 3) for value in values]


def balanced_choice(values: Sequence[Any], params: Mapping[str, Any], *, instance_seed: int, namespace: str) -> Any:
    support = list(values)
    if not support:
        raise ValueError(f"empty support for {namespace}")
    return uniform_choice(
        spawn_rng(int(instance_seed), str(namespace)),
        support,
        sort_keys=True,
    )


def resolve_semantic_axis(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    supported: Sequence[str],
    explicit_key: str,
    weights_key: str,
    balance_key: str,
    namespace: str,
) -> Tuple[str, Dict[str, float]]:
    return resolve_chart_axis_variant(
        params=params,
        gen_defaults=GENERATION_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=tuple(str(value) for value in supported),
        **{CONFIG_CONTEXT_KEY: SCENE_NAMESPACE},
        explicit_key=str(explicit_key),
        weights_key=str(weights_key),
        balance_flag_key=str(balance_key),
        axis_namespace=str(namespace),
    )


def palette(params: Mapping[str, Any]) -> Tuple[RGB, ...]:
    raw = params.get("region_palette_rgb", RENDER_DEFAULTS.get("region_palette_rgb", ()))
    colors: List[RGB] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes)) and len(item) >= 3:
                colors.append(as_rgb(item, (0, 0, 0)))
    return tuple(colors) if colors else (
        (37, 99, 180),
        (210, 83, 73),
        (59, 145, 99),
        (143, 91, 184),
        (216, 145, 48),
        (65, 157, 183),
        (188, 88, 135),
    )


def region_labels(count: int, *, instance_seed: int) -> Tuple[str, ...]:
    labels = resolve_chart_entity_labels(
        spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.labels"),
        count=int(count),
        min_chars=2,
        max_chars=8,
        allow_spaces=False,
    ).labels
    return tuple(str(label) for label in labels)


def density_level_from_density(density: float) -> int:
    return max(1, min(5, int(round(float(density) * 5.0))))


def density_from_level(level: int) -> float:
    return 0.20 + (0.15 * max(1, min(5, int(level))))


def region_count(params: Mapping[str, Any], *, instance_seed: int) -> int:
    low, high = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="region_count_min",
        max_key="region_count_max",
        fallback_min=5,
        fallback_max=7,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    return int(
        balanced_choice(
            list(range(int(low), int(high) + 1)),
            params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.region_count",
        )
    )


def scene_variant(params: Mapping[str, Any], *, instance_seed: int) -> Tuple[str, Dict[str, float]]:
    return resolve_semantic_axis(
        params,
        instance_seed=int(instance_seed),
        supported=SUPPORTED_SCENE_VARIANTS,
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_key="balanced_scene_variant_sampling",
        namespace="scene_variant",
    )


def base_centers(count: int, *, rng: Any) -> List[Tuple[float, float]]:
    candidates = [
        (18.0, 24.0),
        (50.0, 18.0),
        (80.0, 28.0),
        (24.0, 72.0),
        (58.0, 56.0),
        (82.0, 76.0),
        (42.0, 84.0),
    ]
    rng.shuffle(candidates)
    centers: List[Tuple[float, float]] = []
    for x_value, y_value in candidates[: int(count)]:
        centers.append((float(x_value) + rng.uniform(-4.0, 4.0), float(y_value) + rng.uniform(-4.0, 4.0)))
    return centers


def fit_regions_within_unit_bounds(regions: Sequence[Region], *, draw_scale: float = 1.9) -> Tuple[Region, ...]:
    """Shift region centers so the rendered contour footprint stays in the plot."""

    fitted: List[Region] = []
    for region in regions:
        margin_x = min(49.0, max(0.0, float(region.radius_x) * float(draw_scale)))
        margin_y = min(49.0, max(0.0, float(region.radius_y) * float(draw_scale)))
        center_x = min(100.0 - margin_x, max(margin_x, float(region.center_x)))
        center_y = min(100.0 - margin_y, max(margin_y, float(region.center_y)))
        fitted.append(replace(region, center_x=float(center_x), center_y=float(center_y)))
    return tuple(fitted)


def build_regions(
    *,
    count: int,
    labels: Sequence[str],
    option_labels: Sequence[str],
    densities: Sequence[float],
    density_levels: Sequence[int] | None = None,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> Tuple[Region, ...]:
    rng = spawn_rng(int(instance_seed), str(namespace))
    centers = base_centers(int(count), rng=rng)
    colors = palette(params)
    regions: List[Region] = []
    for index in range(int(count)):
        density = float(densities[int(index)])
        density_level = (
            max(1, min(5, int(density_levels[int(index)])))
            if density_levels is not None and int(index) < len(density_levels)
            else density_level_from_density(float(density))
        )
        regions.append(
            Region(
                region_id=f"region_{index}",
                label=str(labels[int(index)]),
                option_label=str(option_labels[int(index)]) if int(index) < len(option_labels) else "",
                center_x=float(centers[int(index)][0]),
                center_y=float(centers[int(index)][1]),
                radius_x=float(rng.uniform(7.5, 12.0)),
                radius_y=float(rng.uniform(6.5, 11.5)),
                density=float(density),
                density_level=int(density_level),
                color_rgb=tuple(colors[int(index) % len(colors)]),
            )
        )
    return fit_regions_within_unit_bounds(regions)


def distance_to_reference(region: Region, reference: Reference) -> float:
    if str(reference.kind) == "point":
        return math.hypot(float(region.center_x) - float(reference.x_value), float(region.center_y) - float(reference.y_value))
    if str(reference.kind) == "vertical_line":
        return abs(float(region.center_x) - float(reference.x_value))
    if str(reference.kind) == "horizontal_line":
        return abs(float(region.center_y) - float(reference.y_value))
    raise ValueError(f"unsupported reference kind: {reference.kind}")


__all__ = [
    "bbox",
    "balanced_choice",
    "build_regions",
    "density_from_level",
    "density_level_from_density",
    "distance_to_reference",
    "fit_regions_within_unit_bounds",
    "region_count",
    "region_labels",
    "resolve_semantic_axis",
    "scene_variant",
]
