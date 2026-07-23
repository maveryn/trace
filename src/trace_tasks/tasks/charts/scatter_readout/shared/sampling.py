"""Sampling primitives for scatter-readout chart scenes."""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.label_assets import resolve_chart_entity_labels
from trace_tasks.tasks.shared.unanswerable import choose_missing_label

from .defaults import gen_int, render_sequence
from .state import (
    MARKER_SHAPES,
    MONTH_LABELS,
    SCENE_NAMESPACE,
    SCENE_VARIANT,
    Point,
    RGB,
    SceneDataset,
    Series,
)


def _as_rgb(value: Any, fallback: RGB) -> RGB:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) < 3:
        return tuple(int(channel) for channel in fallback)
    return tuple(max(0, min(255, int(channel))) for channel in value[:3])  # type: ignore[index]


def palette(params: Mapping[str, Any]) -> tuple[RGB, ...]:
    raw_values = render_sequence(params, "series_palette_rgb", ())
    colors: list[RGB] = []
    for item in raw_values:
        colors.append(_as_rgb(item, (48, 105, 180)))
    if len(colors) >= 5:
        return tuple(colors[:5])
    return (
        (45, 103, 178),
        (211, 86, 70),
        (55, 145, 92),
        (133, 86, 175),
        (213, 136, 41),
    )


def sample_x_axis(*, params: Mapping[str, Any], instance_seed: int, x_count: int) -> tuple[str, tuple[str, ...]]:
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.x_axis")
    axis_kinds = ("year", "month", "period")
    axis_kind = str(params.get("x_axis_kind", axis_kinds[rng.randint(0, len(axis_kinds) - 1)]))
    if axis_kind == "month":
        return "Month", tuple(MONTH_LABELS[: int(x_count)])
    if axis_kind == "period":
        start = rng.randint(1, 9)
        return "Time", tuple(f"T{int(start) + index}" for index in range(int(x_count)))
    start_year = rng.randint(2010, 2027 - int(x_count))
    return "Year", tuple(str(int(start_year) + index) for index in range(int(x_count)))


def sample_series_labels(*, series_count: int, instance_seed: int) -> tuple[str, ...]:
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.series_labels")
    labels = resolve_chart_entity_labels(
        rng,
        count=int(series_count),
        min_chars=2,
        max_chars=7,
        allow_spaces=False,
    ).labels
    return tuple(str(value) for value in labels)


def sample_y_matrix(
    *,
    series_count: int,
    x_count: int,
    params: Mapping[str, Any],
    instance_seed: int,
) -> list[list[int]]:
    value_min = gen_int(params, "readout_y_value_min", 10)
    value_max = gen_int(params, "readout_y_value_max", 90)
    min_gap_at_x = gen_int(params, "readout_min_series_gap_at_x", 6)
    pool = list(range(int(value_min), int(value_max) + 1))
    if len(pool) < int(x_count):
        raise ValueError("readout y-value range is too small for unique per-series values")
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.y_values")
    rows: list[list[int]] = []
    for _ in range(500):
        rows = []
        for _series_index in range(int(series_count)):
            shuffled = list(pool)
            rng.shuffle(shuffled)
            rows.append([int(value) for value in shuffled[: int(x_count)]])
        valid = True
        for x_index in range(int(x_count)):
            column = sorted(row[int(x_index)] for row in rows)
            if any(int(b) - int(a) < int(min_gap_at_x) for a, b in zip(column, column[1:])):
                valid = False
                break
        if valid:
            return rows
    return rows


def build_base_dataset(*, params: Mapping[str, Any], instance_seed: int) -> SceneDataset:
    """Sample the shared scatter-readout scene without deciding a public objective."""

    series_min = gen_int(params, "readout_series_count_min", 3)
    series_max = gen_int(params, "readout_series_count_max", 5)
    x_count_min = gen_int(params, "readout_x_count_min", 6)
    x_count_max = gen_int(params, "readout_x_count_max", 10)
    count_rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.counts")
    series_count = max(3, min(5, int(count_rng.randint(int(series_min), int(series_max)))))
    x_count = max(5, min(10, int(count_rng.randint(int(x_count_min), int(x_count_max)))))

    x_axis_title, x_labels = sample_x_axis(params=params, instance_seed=int(instance_seed), x_count=int(x_count))
    series_labels = sample_series_labels(series_count=int(series_count), instance_seed=int(instance_seed))
    y_values = sample_y_matrix(
        series_count=int(series_count),
        x_count=int(x_count),
        params=params,
        instance_seed=int(instance_seed),
    )
    colors = palette(params)

    series_items: list[Series] = []
    for series_index, label in enumerate(series_labels):
        points = tuple(
            Point(
                point_id=f"s{int(series_index)}_x{int(x_index)}",
                series_label=str(label),
                x_label=str(x_labels[int(x_index)]),
                x_index=int(x_index),
                y_value=int(y_values[int(series_index)][int(x_index)]),
            )
            for x_index in range(int(x_count))
        )
        series_items.append(
            Series(
                label=str(label),
                color_rgb=tuple(colors[int(series_index)]),
                marker_shape=str(MARKER_SHAPES[int(series_index)]),
                points=points,
            )
        )

    return SceneDataset(
        scene_variant=SCENE_VARIANT,
        x_axis_title=str(x_axis_title),
        y_axis_title="Value",
        x_labels=tuple(str(value) for value in x_labels),
        series=tuple(series_items),
    )


def point_by_id(series: Sequence[Series], point_id: str) -> Point:
    for series_item in series:
        for point in series_item.points:
            if str(point.point_id) == str(point_id):
                return point
    raise ValueError(f"unknown point id: {point_id}")


def matching_x_point(series: Series, x_label: str) -> Point:
    matches = [point for point in series.points if str(point.x_label) == str(x_label)]
    if len(matches) != 1:
        raise ValueError(f"expected one point for x-axis label {x_label!r}")
    return matches[0]


def select_series_point(
    *,
    dataset: SceneDataset,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[Series, Point]:
    """Choose one visible series point from a scatter-readout dataset."""

    target_series = uniform_choice(
        spawn_rng(int(instance_seed), f"{namespace}.series"),
        tuple(dataset.series),
    )
    target_point = uniform_choice(
        spawn_rng(int(instance_seed), f"{namespace}.point"),
        tuple(target_series.points),
    )
    return target_series, target_point


def select_x_index(
    *,
    dataset: SceneDataset,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> int:
    """Choose one visible x-axis position from a scatter-readout dataset."""

    return int(
        uniform_choice(
            spawn_rng(int(instance_seed), str(namespace)),
            tuple(range(len(dataset.x_labels))),
            sort_keys=True,
        )
    )


def select_x_extreme_point(
    *,
    dataset: SceneDataset,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    extremum: str,
) -> tuple[str, Point]:
    """Choose one x-axis position and the highest/lowest point in that column."""

    x_index = select_x_index(dataset=dataset, params=params, instance_seed=int(instance_seed), namespace=str(namespace))
    x_label = str(dataset.x_labels[x_index])
    points = [series_item.points[x_index] for series_item in dataset.series]
    if str(extremum) == "highest":
        return x_label, max(points, key=lambda point: (int(point.y_value), str(point.series_label)))
    if str(extremum) == "lowest":
        return x_label, min(points, key=lambda point: (int(point.y_value), str(point.series_label)))
    raise ValueError(f"unsupported x-column extremum: {extremum}")


def select_series_point_pair(
    *,
    dataset: SceneDataset,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> tuple[Series, Point, Series, Point]:
    """Choose an anchor point and same-x comparison point for two-series readout objectives."""

    target_series = uniform_choice(
        spawn_rng(int(instance_seed), f"{namespace}.series"),
        tuple(dataset.series),
    )
    target_point = uniform_choice(
        spawn_rng(int(instance_seed), f"{namespace}.point"),
        tuple(target_series.points),
    )
    comparison_candidates = [series for series in dataset.series if str(series.label) != str(target_series.label)]
    comparison_series = uniform_choice(
        spawn_rng(int(instance_seed), f"{namespace}.comparison_series"),
        tuple(comparison_candidates),
    )
    comparison_point = matching_x_point(comparison_series, str(target_point.x_label))
    return target_series, target_point, comparison_series, comparison_point


def missing_series_label(*, visible_labels: Sequence[str], instance_seed: int) -> str:
    candidates = resolve_chart_entity_labels(
        spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.missing_series_candidates"),
        count=max(12, len(visible_labels) + 6),
        min_chars=2,
        max_chars=7,
        allow_spaces=False,
    ).labels
    return str(
        choose_missing_label(
            visible_labels=visible_labels,
            candidate_labels=candidates,
            fallback_prefix="Series ",
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.missing_series",
        )
    )


__all__ = [
    "build_base_dataset",
    "matching_x_point",
    "missing_series_label",
    "palette",
    "point_by_id",
    "sample_series_labels",
    "sample_x_axis",
    "select_series_point",
    "select_series_point_pair",
    "select_x_extreme_point",
    "select_x_index",
    "sample_y_matrix",
]
