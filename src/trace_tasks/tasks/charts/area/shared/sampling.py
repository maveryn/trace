"""Sampling helpers for the area chart scene."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.area.shared.defaults import (
    GENERATION_DEFAULTS,
    SCENE_NAMESPACE,
    generation_bounds,
    scene_default,
)
from trace_tasks.tasks.charts.shared.label_assets import resolve_chart_entity_labels
from trace_tasks.tasks.charts.shared.label_assets import sample_chart_labels


def sample_point_count(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
) -> tuple[int, tuple[int, int]]:
    point_min, point_max = generation_bounds(
        params,
        "point_count_min",
        "point_count_max",
        7,
        10,
    )
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.point_count")
    return int(rng.randint(int(point_min), int(point_max))), (int(point_min), int(point_max))


def sample_x_labels(
    *,
    count: int,
    instance_seed: int,
) -> tuple[str, ...]:
    return tuple(
        str(label)
        for label in sample_chart_labels(
            count=int(count),
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.x_labels:{int(count)}",
        )
    )


def sample_interval(
    *,
    point_count: int,
    params: Mapping[str, Any],
    instance_seed: int,
) -> tuple[int, int, tuple[int, int]]:
    span_min, span_max = generation_bounds(
        params,
        "interval_span_min",
        "interval_span_max",
        3,
        5,
    )
    span_max = min(int(span_max), max(1, int(point_count) - 1))
    span_min = min(int(span_min), int(span_max))
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.interval")
    span = int(rng.randint(int(span_min), int(span_max)))
    start = int(rng.randint(0, int(point_count) - int(span) - 1))
    return int(start), int(start + span), (int(span_min), int(span_max))


def sample_single_values(
    *,
    point_count: int,
    params: Mapping[str, Any],
    instance_seed: int,
) -> tuple[int, ...]:
    value_min, value_max = generation_bounds(
        params,
        "single_value_min",
        "single_value_max",
        6,
        42,
    )
    even_values = [value for value in range(int(value_min), int(value_max) + 1) if int(value) % 2 == 0]
    if not even_values:
        raise ValueError("single area values require at least one even value")
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.single_values")
    return tuple(int(even_values[rng.randrange(len(even_values))]) for _ in range(int(point_count)))


def sample_categories(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
) -> tuple[tuple[str, ...], tuple[int, int]]:
    cat_min, cat_max = generation_bounds(
        params,
        "category_count_min",
        "category_count_max",
        3,
        4,
    )
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.categories")
    count = int(rng.randint(int(cat_min), int(cat_max)))
    label_max_chars = int(scene_default(params, GENERATION_DEFAULTS, "category_label_max_chars", 6))
    labels = resolve_chart_entity_labels(
        rng,
        count=int(count),
        min_chars=2,
        max_chars=int(label_max_chars),
        allow_spaces=False,
    ).labels
    return tuple(str(label) for label in labels), (int(cat_min), int(cat_max))


def sample_stacked_values(
    *,
    category_count: int,
    point_count: int,
    params: Mapping[str, Any],
    instance_seed: int,
) -> tuple[tuple[int, ...], ...]:
    value_min, value_max = generation_bounds(
        params,
        "stacked_value_min",
        "stacked_value_max",
        4,
        16,
    )
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.stacked_values")
    return tuple(
        tuple(int(rng.randint(int(value_min), int(value_max))) for _ in range(int(point_count)))
        for _ in range(int(category_count))
    )


def trapezoid_interval_area(values: Sequence[int], start_index: int, end_index: int) -> int:
    total = 0
    for index in range(int(start_index), int(end_index)):
        total += (int(values[index]) + int(values[index + 1])) // 2
    return int(total)
