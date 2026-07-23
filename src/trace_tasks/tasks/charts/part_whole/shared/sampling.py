"""Identity-free sampling primitives for part-whole chart tasks."""

from __future__ import annotations

import colorsys
from typing import Any, Mapping, Sequence

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.label_assets import resolve_chart_category_labels
from trace_tasks.tasks.charts.shared.labeled_chart_composition import sample_composition_with_sum

from .defaults import (
    GEN_DEFAULTS,
    ORDER_DIRECTIONS,
    SAMPLING_NAMESPACE,
    balanced_int,
    configured_int_values,
    format_quoted,
    resolve_count_bounds,
)
from .state import CategorySpec, TransferQuery


def palette(count: int, *, instance_seed: int) -> tuple[tuple[int, int, int], ...]:
    """Return a shuffled high-separation categorical palette."""

    colors: list[tuple[int, int, int]] = []
    for index in range(int(count)):
        hue = (0.08 + (float(index) * 0.61803398875)) % 1.0
        lightness = 0.50 + (0.08 if int(index) % 2 else 0.0)
        red, green, blue = colorsys.hls_to_rgb(float(hue), float(lightness), 0.64)
        colors.append((int(round(red * 255)), int(round(green * 255)), int(round(blue * 255))))
    rng = spawn_rng(int(instance_seed), f"{SAMPLING_NAMESPACE}.palette")
    rng.shuffle(colors)
    return tuple((int(red), int(green), int(blue)) for red, green, blue in colors[: int(count)])


def sample_categories(
    *,
    category_count: int,
    value_min: int,
    value_max: int,
    instance_seed: int,
) -> tuple[CategorySpec, ...]:
    """Sample category labels, shares summing to 100, and colors."""

    rng = spawn_rng(int(instance_seed), f"{SAMPLING_NAMESPACE}.categories")
    labels = list(
        resolve_chart_category_labels(
            rng,
            count=int(category_count),
            min_chars=2,
            max_chars=8,
            allow_spaces=False,
        ).labels
    )
    value_rng = spawn_rng(int(instance_seed), f"{SAMPLING_NAMESPACE}.values")
    values = sample_composition_with_sum(
        value_rng,
        target_sum=100,
        count=int(category_count),
        value_min=int(value_min),
        value_max=int(value_max),
    )
    value_rng.shuffle(values)
    colors = palette(int(category_count), instance_seed=int(instance_seed))
    return tuple(
        CategorySpec(label=str(label), value=int(value), color_rgb=tuple(colors[index]))
        for index, (label, value) in enumerate(zip(labels, values))
    )


def sample_categories_from_values(
    values: Sequence[int],
    *,
    instance_seed: int,
    namespace_suffix: str,
) -> tuple[CategorySpec, ...]:
    """Create categories around preselected share values."""

    rng = spawn_rng(int(instance_seed), f"{SAMPLING_NAMESPACE}.{namespace_suffix}.categories")
    labels = list(
        resolve_chart_category_labels(
            rng,
            count=len(values),
            min_chars=2,
            max_chars=8,
            allow_spaces=False,
        ).labels
    )
    colors = palette(len(values), instance_seed=int(instance_seed))
    return tuple(
        CategorySpec(label=str(label), value=int(value), color_rgb=tuple(colors[index]))
        for index, (label, value) in enumerate(zip(labels, values))
    )


def value_bounds(params: Mapping[str, Any]) -> tuple[int, int]:
    """Resolve compact integer share bounds used by active part-whole tasks."""

    return resolve_count_bounds(
        params,
        min_key="compact_value_min",
        max_key="compact_value_max",
        fallback_min=2,
        fallback_max=40,
    )


def sample_category_count(
    *,
    params: Mapping[str, Any],
    count_params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
    even_only: bool = False,
) -> tuple[int, tuple[int, int]]:
    """Resolve and sample a category count support."""

    category_min, category_max = resolve_count_bounds(
        params,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
    )
    support = [int(value) for value in range(int(category_min), int(category_max) + 1)]
    if bool(even_only):
        support = [int(value) for value in support if int(value) % 2 == 0]
    if not support:
        raise ValueError("part-whole category count support is empty")
    return (
        balanced_int(
            support,
            params=count_params,
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        ),
        (int(category_min), int(category_max)),
    )


def base_extras(
    categories: Sequence[CategorySpec],
    *,
    category_count_range: tuple[int, int],
    value_min: int,
    value_max: int,
) -> dict[str, Any]:
    """Build common symbolic fields shared by all part-whole objectives."""

    labels = tuple(str(category.label) for category in categories)
    values_by_label = {str(category.label): int(category.value) for category in categories}
    return {
        "category_count": int(len(categories)),
        "category_count_range": [int(category_count_range[0]), int(category_count_range[1])],
        "value_min": int(value_min),
        "value_max": int(value_max),
        "category_values": {str(label): int(values_by_label[str(label)]) for label in labels},
        "chart_order_labels": [str(label) for label in labels],
        "table_order_labels": [str(label) for label in sorted(labels)],
    }


def sample_chart_order_span(
    categories: Sequence[CategorySpec],
    *,
    direction: str,
    params: Mapping[str, Any],
    count_params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
    require_share_multiple_of_five: bool = False,
) -> tuple[tuple[CategorySpec, ...], dict[str, Any]]:
    """Select a non-full contiguous circular span under one direction."""

    if str(direction) not in ORDER_DIRECTIONS:
        raise ValueError("unsupported chart-order direction")
    span_min, span_max = resolve_count_bounds(
        params,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
    )
    feasible_span_min = max(2, int(span_min))
    feasible_span_max = min(int(span_max), len(categories) - 1)
    if int(feasible_span_min) > int(feasible_span_max):
        raise ValueError("part-whole span requires enough categories for a non-full circular run")
    step = 1 if str(direction) == "clockwise" else -1
    options: list[tuple[int, int, tuple[int, ...], int]] = []
    for span_count in range(int(feasible_span_min), int(feasible_span_max) + 1):
        for start_index in range(0, len(categories)):
            selected_indices = tuple(
                (int(start_index) + (int(step) * int(offset))) % len(categories)
                for offset in range(int(span_count))
            )
            selected_share = int(sum(int(categories[int(index)].value) for index in selected_indices))
            if bool(require_share_multiple_of_five) and int(selected_share) % 5 != 0:
                continue
            options.append((int(span_count), int(start_index), tuple(int(index) for index in selected_indices), int(selected_share)))
    if not options:
        raise ValueError("could not construct a feasible circular span")
    option_index = balanced_int(
        range(0, len(options)),
        params=count_params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.span",
    )
    span_count, start_index, selected_indices, selected_share = options[int(option_index)]
    selected = tuple(categories[int(index)] for index in selected_indices)
    return selected, {
        "start_category": str(selected[0].label),
        "end_category": str(selected[-1].label),
        "start_index": int(start_index),
        "end_index": int(selected_indices[-1]),
        "selected_indices": [int(index) for index in selected_indices],
        "span_count": int(span_count),
        "span_count_range": [int(feasible_span_min), int(feasible_span_max)],
        "chart_order_direction": str(direction),
        "selected_share_value": int(selected_share),
    }


def sample_subset_denominator(
    *,
    category_count: int,
    share_min: int,
    share_max: int,
    params: Mapping[str, Any],
    count_params: Mapping[str, Any],
    instance_seed: int,
) -> tuple[tuple[CategorySpec, ...], tuple[CategorySpec, ...], dict[str, Any]]:
    """Construct a subset-denominator percentage with exact integer answer."""

    subset_min, subset_max = resolve_count_bounds(
        params,
        min_key="subset_denominator_size_min",
        max_key="subset_denominator_size_max",
        fallback_min=2,
        fallback_max=4,
    )
    subset_sizes = list(range(max(2, int(subset_min)), min(int(subset_max), int(category_count) - 1) + 1))
    if not subset_sizes:
        raise ValueError("subset-denominator task requires a non-full subset")
    answer_values = configured_int_values(
        params,
        key="subset_denominator_answer_values",
        fallback=(20, 25, 30, 40, 50, 60, 70, 75, 80),
    )
    feasible: list[tuple[int, int, int, int, int]] = []
    for subset_size in subset_sizes:
        outside_count = int(category_count) - int(subset_size)
        subset_total_min = max(int(subset_size) * int(share_min), 12)
        subset_total_max = min(int(subset_size) * int(share_max), 100 - (int(outside_count) * int(share_min)))
        for answer_percent in answer_values:
            if not 1 <= int(answer_percent) <= 99:
                continue
            for subset_total in range(int(subset_total_min), int(subset_total_max) + 1):
                product = int(subset_total) * int(answer_percent)
                if product % 100 != 0:
                    continue
                target_value = int(product // 100)
                subset_remainder = int(subset_total) - int(target_value)
                outside_total = 100 - int(subset_total)
                if not int(share_min) <= int(target_value) <= int(share_max):
                    continue
                if not (int(subset_size) - 1) * int(share_min) <= int(subset_remainder) <= (int(subset_size) - 1) * int(share_max):
                    continue
                if not int(outside_count) * int(share_min) <= int(outside_total) <= int(outside_count) * int(share_max):
                    continue
                feasible.append((int(answer_percent), int(subset_size), int(subset_total), int(target_value), int(outside_total)))
    if not feasible:
        raise ValueError("could not construct subset-denominator values with configured bounds")
    available_answers = sorted({int(item[0]) for item in feasible})
    answer_percent = balanced_int(
        available_answers,
        params=count_params,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.subset.answer_percent",
    )
    answer_candidates = [item for item in feasible if int(item[0]) == int(answer_percent)]
    subset_size = balanced_int(
        sorted({int(item[1]) for item in answer_candidates}),
        params=count_params,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.subset.size",
    )
    size_candidates = [item for item in answer_candidates if int(item[1]) == int(subset_size)]
    subset_total = balanced_int(
        sorted({int(item[2]) for item in size_candidates}),
        params=count_params,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.subset.total",
    )
    target_value = int(int(subset_total) * int(answer_percent) // 100)
    outside_total = int(100 - int(subset_total))
    subset_remainder = int(subset_total) - int(target_value)
    outside_count = int(category_count) - int(subset_size)
    subset_rng = spawn_rng(int(instance_seed), f"{SAMPLING_NAMESPACE}.subset.rest")
    outside_rng = spawn_rng(int(instance_seed), f"{SAMPLING_NAMESPACE}.subset.outside")
    subset_rest = sample_composition_with_sum(
        subset_rng,
        target_sum=int(subset_remainder),
        count=int(subset_size) - 1,
        value_min=int(share_min),
        value_max=int(share_max),
    )
    outside_values = sample_composition_with_sum(
        outside_rng,
        target_sum=int(outside_total),
        count=int(outside_count),
        value_min=int(share_min),
        value_max=int(share_max),
    )
    entries: list[tuple[str, int]] = [("target", int(target_value))]
    entries.extend(("subset", int(value)) for value in subset_rest)
    entries.extend(("outside", int(value)) for value in outside_values)
    layout_rng = spawn_rng(int(instance_seed), f"{SAMPLING_NAMESPACE}.subset.layout")
    layout_rng.shuffle(entries)
    categories = sample_categories_from_values(
        [int(value) for _, value in entries],
        instance_seed=int(instance_seed),
        namespace_suffix="subset",
    )
    subset_indices = tuple(index for index, (role, _) in enumerate(entries) if str(role) in {"target", "subset"})
    target_index = next(index for index, (role, _) in enumerate(entries) if str(role) == "target")
    selected = tuple(categories[int(index)] for index in subset_indices)
    target_category = categories[int(target_index)]
    return tuple(categories), tuple(selected), {
        "target_category": str(target_category.label),
        "target_index": int(target_index),
        "target_share_value": int(target_category.value),
        "category_list": [str(category.label) for category in selected],
        "subset_category_list": [str(category.label) for category in selected],
        "subset_category_list_text": format_quoted([str(category.label) for category in selected]),
        "selected_indices": [int(index) for index in subset_indices],
        "subset_size": int(len(selected)),
        "subset_size_range": [int(min(subset_sizes)), int(max(subset_sizes))],
        "selected_share_value": int(subset_total),
        "subset_share_total": int(subset_total),
        "subset_denominator_percent": int(answer_percent),
        "calculation": "compute_category_share_within_named_subset_denominator",
    }


def sample_adjacent_transfer(
    categories: Sequence[CategorySpec],
    *,
    direction: str,
    params: Mapping[str, Any],
    count_params: Mapping[str, Any],
    instance_seed: int,
) -> TransferQuery:
    """Select a source and adjacent target under one circular direction."""

    if str(direction) not in ORDER_DIRECTIONS:
        raise ValueError("unsupported adjacent-transfer direction")
    delta_min, delta_max = resolve_count_bounds(
        params,
        min_key="counterfactual_transfer_delta_min",
        max_key="counterfactual_transfer_delta_max",
        fallback_min=3,
        fallback_max=10,
    )
    options: list[tuple[CategorySpec, CategorySpec, int, int, int]] = []
    step = 1 if str(direction) == "clockwise" else -1
    for source_index, source in enumerate(categories):
        target_index = (int(source_index) + int(step)) % len(categories)
        target = categories[int(target_index)]
        for delta in range(int(delta_min), int(delta_max) + 1):
            if int(source.value) - int(delta) >= 1:
                options.append((source, target, int(source_index), int(target_index), int(delta)))
    if not options:
        raise ValueError("could not construct adjacent transfer-gap query")
    feasible_deltas = sorted({int(option[4]) for option in options})
    delta = balanced_int(
        feasible_deltas,
        params=count_params,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.adjacent_transfer.delta",
    )
    candidates = [option for option in options if int(option[4]) == int(delta)]
    option_index = balanced_int(
        range(0, len(candidates)),
        params=count_params,
        instance_seed=int(instance_seed),
        namespace=f"{SAMPLING_NAMESPACE}.adjacent_transfer.source",
    )
    source, target, source_index, target_index, delta = candidates[int(option_index)]
    return TransferQuery(
        source=source,
        target=target,
        delta=int(delta),
        extras={
            "source_order_direction": str(direction),
            "source_index": int(source_index),
            "target_index": int(target_index),
            "source_selection_rule": "explicit_category_in_chart_order",
            "target_selection_rule": "adjacent_category_in_chart_order",
            "calculation": "adjacent_chart_order_absolute_gap_after_share_transfer",
        },
    )
