"""Sampling helpers for scatter-point chart scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.label_assets import ResolvedChartLabels, resolve_chart_category_labels
from trace_tasks.tasks.charts.shared.visual_defaults import coerce_rgb

from .defaults import GEN_DEFAULTS, RENDER_DEFAULTS, gen_float, gen_int, gen_sequence, group_default
from .state import Category, Dataset, MARKER_SHAPES, Point, Query, RGB, SCENE_NAMESPACE


def as_rgb(value: Any, fallback: RGB) -> RGB:
    return coerce_rgb(value, fallback)


def sample_count(rng: Any, *, low: int, high: int) -> int:
    low = int(low)
    high = max(int(low), int(high))
    return int(rng.randint(low, high))


def sample_threshold(rng: Any, params: Mapping[str, Any]) -> int:
    values = [int(value) for value in gen_sequence(params, "threshold_values", (20, 30, 40, 50, 60, 70, 80))]
    if not values:
        values = [20, 30, 40, 50, 60, 70, 80]
    return int(rng.choice(values))


def coord_on_side(rng: Any, *, threshold: float, direction: str, match: bool, margin: float = 4.0) -> float:
    threshold = float(threshold)
    if str(direction) == "above":
        low, high = (threshold + margin, 96.0) if bool(match) else (4.0, threshold - margin)
    else:
        low, high = (4.0, threshold - margin) if bool(match) else (threshold + margin, 96.0)
    if low > high:
        low, high = min(low, high), max(low, high)
    return round(float(rng.uniform(float(low), float(high))), 3)


def palette(params: Mapping[str, Any]) -> tuple[RGB, ...]:
    raw = params.get("scatter_points_category_palette_rgb", RENDER_DEFAULTS.get("scatter_points_category_palette_rgb", ()))
    colors: list[RGB] = []
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            colors.append(as_rgb(item, (42, 96, 176)))
    if len(colors) >= 6:
        return tuple(colors)
    return (
        (42, 96, 176),
        (210, 83, 72),
        (52, 144, 92),
        (138, 82, 178),
        (215, 137, 42),
        (44, 148, 169),
        (184, 73, 124),
    )


def label_metadata(label_resolution: ResolvedChartLabels | None) -> dict[str, Any]:
    if label_resolution is None:
        return {}
    return {
        "labels": list(label_resolution.labels),
        "label_variant": str(label_resolution.label_variant),
        "label_pool_kind": str(label_resolution.label_pool_kind),
        "label_source_kind": str(label_resolution.label_source_kind),
        "label_bucket": str(label_resolution.label_bucket),
        "label_manifest": str(label_resolution.label_manifest),
        "label_filter": dict(label_resolution.label_filter),
        "label_bucket_probabilities": dict(label_resolution.label_bucket_probabilities),
    }


def resolve_category_labels(rng: Any, params: Mapping[str, Any], count: int) -> ResolvedChartLabels:
    weights = params.get("category_label_bucket_weights", GEN_DEFAULTS.get("category_label_bucket_weights"))
    return resolve_chart_category_labels(
        rng,
        count=int(count),
        min_chars=int(gen_int(params, "category_label_min_chars", 3)),
        max_chars=int(gen_int(params, "category_label_max_chars", 12)),
        allow_spaces=bool(params.get("category_label_allow_spaces", group_default(GEN_DEFAULTS, "category_label_allow_spaces", True))),
        bucket_weights=weights if isinstance(weights, Mapping) else None,
    )


def new_point(
    *,
    point_id: str,
    x_value: float,
    y_value: float,
    category_label: str,
    color_rgb: RGB,
    marker_shape: str,
) -> Point:
    return Point(
        point_id=str(point_id),
        x_value=round(float(x_value), 3),
        y_value=round(float(y_value), 3),
        category_label=str(category_label),
        color_rgb=tuple(int(channel) for channel in color_rgb),
        marker_shape=str(marker_shape),
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def build_axis_threshold_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    threshold_axis: str,
    threshold_direction: str,
) -> Dataset:
    """Sample an uncategorized scatter plot with a controlled threshold count."""

    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.axis_threshold")
    point_count = sample_count(
        rng,
        low=gen_int(params, "scatter_points_count_min", 28),
        high=gen_int(params, "scatter_points_count_max", 54),
    )
    threshold_value = sample_threshold(rng, params)
    answer_low = max(1, gen_int(params, "axis_threshold_answer_min", 4))
    answer_high = min(point_count - 1, gen_int(params, "axis_threshold_answer_max", 18))
    answer_count = sample_count(rng, low=answer_low, high=max(answer_low, answer_high))
    matching_indices = set(rng.sample(range(point_count), k=int(answer_count)))
    color_rgb = as_rgb(params.get("plain_point_rgb", RENDER_DEFAULTS.get("plain_point_rgb")), (48, 99, 176))
    marker_shape = str(params.get("plain_marker_shape", group_default(GEN_DEFAULTS, "plain_marker_shape", "circle")))

    points: list[Point] = []
    annotation_point_ids: list[str] = []
    for index in range(point_count):
        point_id = f"P{index + 1:02d}"
        matches = index in matching_indices
        queried_value = coord_on_side(rng, threshold=float(threshold_value), direction=str(threshold_direction), match=matches)
        other_value = round(float(rng.uniform(4.0, 96.0)), 3)
        if str(threshold_axis) == "x":
            point = new_point(
                point_id=point_id,
                x_value=queried_value,
                y_value=other_value,
                category_label="",
                color_rgb=color_rgb,
                marker_shape=marker_shape,
            )
        else:
            point = new_point(
                point_id=point_id,
                x_value=other_value,
                y_value=queried_value,
                category_label="",
                color_rgb=color_rgb,
                marker_shape=marker_shape,
            )
        points.append(point)
        if matches:
            annotation_point_ids.append(str(point_id))

    trace = {
        "threshold_axis": str(threshold_axis),
        "threshold_direction": str(threshold_direction),
        "threshold_value": int(threshold_value),
        "target_answer_count": int(answer_count),
    }
    return Dataset(
        scene_variant="plain_scatter",
        points=tuple(points),
        categories=(),
        query=Query(
            answer=int(answer_count),
            answer_type="integer",
            annotation_point_ids=tuple(annotation_point_ids),
            trace=trace,
        ),
    )


def build_category_mean_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    mean_axis: str,
    mean_extremum: str,
) -> Dataset:
    """Sample categorized point clouds with one category having the extremal mean."""

    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.category_mean")
    category_count = sample_count(
        rng,
        low=gen_int(params, "scatter_points_category_count_min", 4),
        high=gen_int(params, "scatter_points_category_count_max", 6),
    )
    points_per_category = sample_count(
        rng,
        low=gen_int(params, "scatter_points_per_category_min", 6),
        high=gen_int(params, "scatter_points_per_category_max", 10),
    )
    label_resolution = resolve_category_labels(rng, params, int(category_count))
    labels = tuple(str(label) for label in label_resolution.labels)
    colors = palette(params)
    target_index = int(rng.randrange(int(category_count)))
    target_label = str(labels[target_index])
    margin_min = max(3.0, gen_float(params, "mean_extremum_margin_min", 5.0))
    margin_max = max(margin_min, gen_float(params, "mean_extremum_margin_max", 9.0))
    target_center = float(rng.uniform(60.0, 82.0) if str(mean_extremum) == "largest" else rng.uniform(18.0, 40.0))
    close_competitor_index = int(rng.choice([index for index in range(int(category_count)) if index != int(target_index)]))
    close_gap = float(rng.uniform(float(margin_min), float(margin_max)))

    points: list[Point] = []
    categories: list[Category] = []
    annotation_point_ids: list[str] = []
    means: dict[str, dict[str, float]] = {}

    for category_index, label in enumerate(labels):
        is_target = int(category_index) == int(target_index)
        is_close_competitor = int(category_index) == int(close_competitor_index)
        marker_shape = MARKER_SHAPES[int(category_index)]
        color_rgb = colors[int(category_index)]
        point_ids: list[str] = []
        if is_target:
            queried_center = float(target_center)
        elif is_close_competitor:
            queried_center = (
                float(target_center - close_gap)
                if str(mean_extremum) == "largest"
                else float(target_center + close_gap)
            )
        elif str(mean_extremum) == "largest":
            queried_center = float(rng.uniform(10.0, max(10.0, target_center - margin_max - 2.0)))
        else:
            queried_center = float(rng.uniform(min(90.0, target_center + margin_max + 2.0), 90.0))
        other_center = float(rng.uniform(18.0, 82.0))
        xs: list[float] = []
        ys: list[float] = []
        for point_index in range(int(points_per_category)):
            point_id = f"C{category_index + 1:02d}P{point_index + 1:02d}"
            queried_value = _clamp(float(rng.gauss(queried_center, 4.0)), 4.0, 96.0)
            other_value = _clamp(float(rng.gauss(other_center, 12.0)), 4.0, 96.0)
            if str(mean_axis) == "x":
                x_value, y_value = queried_value, other_value
            else:
                x_value, y_value = other_value, queried_value
            point = new_point(
                point_id=point_id,
                x_value=x_value,
                y_value=y_value,
                category_label=str(label),
                color_rgb=color_rgb,
                marker_shape=marker_shape,
            )
            points.append(point)
            point_ids.append(str(point_id))
            xs.append(float(point.x_value))
            ys.append(float(point.y_value))
            if is_target:
                annotation_point_ids.append(str(point_id))
        categories.append(Category(label=str(label), color_rgb=color_rgb, marker_shape=marker_shape, point_ids=tuple(point_ids)))
        means[str(label)] = {
            "x": round(float(sum(xs) / len(xs)), 4),
            "y": round(float(sum(ys) / len(ys)), 4),
        }

    target_mean = float(means[target_label][str(mean_axis)])
    other_means = [float(values[str(mean_axis)]) for label, values in means.items() if str(label) != target_label]
    actual_margin = target_mean - max(other_means) if str(mean_extremum) == "largest" else min(other_means) - target_mean
    trace = {
        "mean_axis": str(mean_axis),
        "mean_extremum": str(mean_extremum),
        "target_category_label": str(target_label),
        "close_competitor_label": str(labels[int(close_competitor_index)]),
        "sampled_center_gap": round(float(close_gap), 4),
        "category_means": dict(means),
        "mean_margin": round(float(actual_margin), 4),
        "label_resolution": label_metadata(label_resolution),
    }
    return Dataset(
        scene_variant="categorized_scatter",
        points=tuple(points),
        categories=tuple(categories),
        query=Query(
            answer=str(target_label),
            answer_type="string",
            annotation_point_ids=tuple(annotation_point_ids),
            trace=trace,
        ),
        label_resolution=label_resolution,
    )


def build_category_threshold_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    threshold_axis: str,
    threshold_direction: str,
) -> Dataset:
    """Sample categorized points with a controlled threshold count for one category."""

    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.category_threshold")
    category_count = sample_count(
        rng,
        low=gen_int(params, "scatter_points_category_count_min", 4),
        high=gen_int(params, "scatter_points_category_count_max", 6),
    )
    points_per_category = sample_count(
        rng,
        low=gen_int(params, "category_threshold_points_per_category_min", 7),
        high=gen_int(params, "category_threshold_points_per_category_max", 12),
    )
    threshold_value = sample_threshold(rng, params)
    label_resolution = resolve_category_labels(rng, params, int(category_count))
    labels = tuple(str(label) for label in label_resolution.labels)
    colors = palette(params)
    target_index = int(rng.randrange(int(category_count)))
    target_label = str(labels[target_index])
    answer_low = max(1, gen_int(params, "category_threshold_answer_min", 2))
    answer_high = min(int(points_per_category) - 1, gen_int(params, "category_threshold_answer_max", 9))
    answer_count = sample_count(rng, low=answer_low, high=max(answer_low, answer_high))
    target_matching_indices = set(rng.sample(range(int(points_per_category)), k=int(answer_count)))

    points: list[Point] = []
    categories: list[Category] = []
    annotation_point_ids: list[str] = []
    match_counts_by_category: dict[str, int] = {}
    for category_index, label in enumerate(labels):
        marker_shape = MARKER_SHAPES[int(category_index)]
        color_rgb = colors[int(category_index)]
        point_ids: list[str] = []
        category_match_count = 0
        for point_index in range(int(points_per_category)):
            point_id = f"C{category_index + 1:02d}P{point_index + 1:02d}"
            matches = int(point_index) in target_matching_indices if int(category_index) == int(target_index) else bool(rng.random() < 0.42)
            queried_value = coord_on_side(rng, threshold=float(threshold_value), direction=str(threshold_direction), match=matches)
            other_value = round(float(rng.uniform(4.0, 96.0)), 3)
            if str(threshold_axis) == "x":
                x_value, y_value = queried_value, other_value
            else:
                x_value, y_value = other_value, queried_value
            point = new_point(
                point_id=point_id,
                x_value=x_value,
                y_value=y_value,
                category_label=str(label),
                color_rgb=color_rgb,
                marker_shape=marker_shape,
            )
            points.append(point)
            point_ids.append(str(point_id))
            if matches:
                category_match_count += 1
                if int(category_index) == int(target_index):
                    annotation_point_ids.append(str(point_id))
        categories.append(Category(label=str(label), color_rgb=color_rgb, marker_shape=marker_shape, point_ids=tuple(point_ids)))
        match_counts_by_category[str(label)] = int(category_match_count)

    trace = {
        "threshold_axis": str(threshold_axis),
        "threshold_direction": str(threshold_direction),
        "threshold_value": int(threshold_value),
        "target_category_label": str(target_label),
        "match_counts_by_category": dict(match_counts_by_category),
        "label_resolution": label_metadata(label_resolution),
    }
    return Dataset(
        scene_variant="categorized_scatter",
        points=tuple(points),
        categories=tuple(categories),
        query=Query(
            answer=int(answer_count),
            answer_type="integer",
            annotation_point_ids=tuple(annotation_point_ids),
            trace=trace,
        ),
        label_resolution=label_resolution,
    )


__all__ = [
    "as_rgb",
    "build_axis_threshold_dataset",
    "build_category_mean_dataset",
    "build_category_threshold_dataset",
    "coord_on_side",
    "label_metadata",
    "new_point",
    "palette",
    "resolve_category_labels",
    "sample_count",
    "sample_threshold",
]
