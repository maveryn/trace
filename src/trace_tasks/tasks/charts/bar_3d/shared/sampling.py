"""Sampling and semantic selection primitives for the 3D bar-grid scene."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

from .....core.sampling import uniform_choice_with_probabilities
from .....core.seed import spawn_rng
from ....shared.color_distance import sample_color_palette_with_distance_constraints
from ....shared.config_defaults import group_default, resolve_required_int_bounds
from ...shared.label_assets import resolve_chart_entity_labels
from ...shared.label_assets import sample_chart_labels
from .defaults import _DEFAULT_PALETTE, _GEN_DEFAULTS, _RENDER_DEFAULTS, _as_rgb
from .state import (
    SCENE_NAMESPACE,
    _BarCell,
    _Dataset,
    _Selection,
)


GridValues = Tuple[Tuple[int, ...], ...]
GridRanges = Dict[str, Any]
SelectionFactory = Callable[[tuple[str, ...], tuple[str, ...], GridValues], _Selection]


def _sample_palette(*, instance_seed: int, count: int) -> tuple[tuple[int, int, int], ...]:
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.palette")
    raw_palette = _RENDER_DEFAULTS.get("series_palette_rgb")
    if isinstance(raw_palette, Sequence) and not isinstance(raw_palette, (str, bytes)) and len(raw_palette) >= int(count):
        palette = [_as_rgb(item, _DEFAULT_PALETTE[index % len(_DEFAULT_PALETTE)]) for index, item in enumerate(raw_palette)]
        rng.shuffle(palette)
        return tuple(palette[: int(count)])
    palette = sample_color_palette_with_distance_constraints(
        rng,
        palette_size=int(count),
        channel_min=35,
        channel_max=218,
        anchor_colors=((255, 255, 255), (248, 248, 248)),
        min_distance=44.0,
        distance_space="lab",
    )
    return tuple(tuple(int(channel) for channel in color) for color in palette)


def _sample_x_labels(*, count: int, instance_seed: int) -> tuple[str, ...]:
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.x_labels")
    if rng.random() < 0.58:
        start = int(rng.randint(2014, 2026 - int(count)))
        return tuple(str(start + index) for index in range(int(count)))
    return tuple(
        str(label)
        for label in sample_chart_labels(
            count=int(count),
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.labels.x.{int(count)}",
        )
    )


def _sample_series_labels(*, count: int, instance_seed: int) -> tuple[str, ...]:
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.series_labels")
    labels = resolve_chart_entity_labels(
        rng,
        count=int(count),
        min_chars=2,
        max_chars=7,
        allow_spaces=False,
    ).labels
    return tuple(str(label) for label in labels)


def _int_sequence_default(params: Mapping[str, Any], key: str, fallback: Sequence[int]) -> tuple[int, ...]:
    raw = params.get(str(key), group_default(_GEN_DEFAULTS, str(key), list(fallback)))
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
        return tuple(int(value) for value in fallback)
    values = tuple(int(value) for value in raw)
    return values if values else tuple(int(value) for value in fallback)


def condition_answer_support(params: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(
        int(value)
        for value in _int_sequence_default(params, "condition_count_answer_support", (1, 2, 3, 4, 5))
        if int(value) > 0
    )


def condition_axis_params(params: Mapping[str, Any], *, axis: str) -> dict[str, Any]:
    """Return grid-size defaults for threshold counts along one semantic axis."""

    if str(axis) == "category":
        defaults = {
            "condition_category_count_min": 3,
            "condition_category_count_max": 4,
            "condition_series_count_min": 6,
            "condition_series_count_max": 6,
        }
    elif str(axis) == "series":
        defaults = {
            "condition_category_count_min": 6,
            "condition_category_count_max": 6,
            "condition_series_count_min": 3,
            "condition_series_count_max": 4,
        }
    else:
        raise ValueError(f"unknown condition axis {axis!r}")
    return {**defaults, **dict(params)}


def interval_scope_params(params: Mapping[str, Any]) -> dict[str, Any]:
    """Return defaults for contiguous category intervals in series-scope totals."""

    return {
        "interval_category_count_min": 3,
        "interval_category_count_max": 4,
        **dict(params),
    }


def sample_pairwise_target_count(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    max_category_count: int,
) -> int:
    eligible_counts = [
        int(count)
        for count in condition_answer_support(params)
        if 1 <= int(count) < int(max_category_count)
    ]
    if not eligible_counts:
        eligible_counts = [1]
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.pairwise_target_count")
    return int(eligible_counts[int(rng.randrange(len(eligible_counts)))])


def sample_condition_target_count(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    axis_size: int,
    namespace: str,
) -> tuple[int, dict[str, float]]:
    support = tuple(
        int(count)
        for count in condition_answer_support(params)
        if 1 <= int(count) < int(axis_size)
    )
    if not support:
        raise ValueError("condition count answer support has no nontrivial values for the selected axis")
    selected, probabilities = uniform_choice_with_probabilities(
        spawn_rng(int(instance_seed), str(namespace)),
        support,
        sort_keys=True,
    )
    return int(selected), dict(probabilities)


def pairwise_target_max_category_count(params: Mapping[str, Any]) -> int:
    _, x_max = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key="condition_category_count_min",
        max_key="condition_category_count_max",
        fallback_min=4,
        fallback_max=6,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    return max(1, min(int(x_max), 8))


def sample_grid(
    params: Mapping[str, Any],
    *,
    condition_scope: bool,
    pairwise_target_count: int | None,
    instance_seed: int,
) -> tuple[tuple[str, ...], tuple[str, ...], GridValues, GridRanges]:
    """Sample one bounded 3D bar grid while preserving calibrated count and max-bar constraints."""

    category_min_key = "condition_category_count_min" if bool(condition_scope) else "category_count_min"
    category_max_key = "condition_category_count_max" if bool(condition_scope) else "category_count_max"
    series_min_key = "condition_series_count_min" if bool(condition_scope) else "series_count_min"
    series_max_key = "condition_series_count_max" if bool(condition_scope) else "series_count_max"
    x_min, x_max = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key=category_min_key,
        max_key=category_max_key,
        fallback_min=4,
        fallback_max=6,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    series_min, series_max = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key=series_min_key,
        max_key=series_max_key,
        fallback_min=3,
        fallback_max=5,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    value_min, value_max = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key="value_min",
        max_key="value_max",
        fallback_min=6,
        fallback_max=36,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    x_max = max(int(x_min), min(int(x_max), 8))
    series_max = max(int(series_min), int(series_max))
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.grid")
    max_bar_count = max(1, int(params.get("max_bar_count", group_default(_GEN_DEFAULTS, "max_bar_count", 24))))
    valid_grid_sizes = [
        (int(x_count), int(series_count))
        for x_count in range(int(x_min), int(x_max) + 1)
        for series_count in range(int(series_min), int(series_max) + 1)
        if int(x_count) * int(series_count) <= int(max_bar_count)
        and (pairwise_target_count is None or int(x_count) > int(pairwise_target_count))
    ]
    if not valid_grid_sizes:
        raise ValueError("3D bar grid size constraints leave no valid category/series pairs")
    x_count, series_count = valid_grid_sizes[int(rng.randrange(len(valid_grid_sizes)))]
    x_labels = _sample_x_labels(count=int(x_count), instance_seed=int(instance_seed))
    series_labels = _sample_series_labels(count=int(series_count), instance_seed=int(instance_seed))
    values_grid = [
        [int(rng.randint(int(value_min), int(value_max))) for _ in range(int(series_count))]
        for _ in range(int(x_count))
    ]
    if pairwise_target_count is not None:
        control_rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.pairwise_control_pair")
        series_order = list(range(int(series_count)))
        control_rng.shuffle(series_order)
        series_a, series_b = int(series_order[0]), int(series_order[1])
        x_order = list(range(int(x_count)))
        control_rng.shuffle(x_order)
        winning_x = set(int(index) for index in x_order[: int(pairwise_target_count)])
        for x_index in range(int(x_count)):
            low = int(control_rng.randint(int(value_min), max(int(value_min), int(value_max) - 1)))
            high = int(control_rng.randint(int(low) + 1, int(value_max))) if int(low) < int(value_max) else int(value_max)
            if int(x_index) in winning_x:
                values_grid[int(x_index)][int(series_a)] = int(high)
                values_grid[int(x_index)][int(series_b)] = int(low)
            else:
                values_grid[int(x_index)][int(series_a)] = int(low)
                values_grid[int(x_index)][int(series_b)] = int(high)
    values = tuple(tuple(int(value) for value in row) for row in values_grid)
    ranges = {
        "category_count_range": [int(x_min), int(x_max)],
        "series_count_range": [int(series_min), int(series_max)],
        "max_bar_count": int(max_bar_count),
        "value_range": [int(value_min), int(value_max)],
    }
    return tuple(x_labels), tuple(series_labels), values, dict(ranges)


def choose_interval(*, x_count: int, params: Mapping[str, Any], instance_seed: int) -> tuple[int, int, tuple[int, int]]:
    span_min, span_max = resolve_required_int_bounds(
        params,
        _GEN_DEFAULTS,
        min_key="interval_category_count_min",
        max_key="interval_category_count_max",
        fallback_min=2,
        fallback_max=4,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    span_max = min(int(span_max), int(x_count))
    span_min = min(int(span_min), int(span_max))
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.interval")
    span_count = int(rng.randint(int(span_min), int(span_max)))
    start = int(rng.randint(0, int(x_count) - int(span_count)))
    return int(start), int(start + span_count - 1), (int(span_min), int(span_max))


def valid_threshold(
    values: Sequence[int],
    *,
    want_at_least: bool,
    rng: Any,
    target_counts: Sequence[int] | None = None,
) -> tuple[int, int] | None:
    if len(values) < 2:
        return None
    lower = min(int(value) for value in values)
    upper = max(int(value) for value in values)
    candidates_by_count: dict[int, list[int]] = defaultdict(list)
    for threshold in range(int(lower), int(upper) + 1):
        if bool(want_at_least):
            count = sum(1 for value in values if int(value) >= int(threshold))
        else:
            count = sum(1 for value in values if int(value) < int(threshold))
        if 1 <= int(count) <= len(values) - 1:
            candidates_by_count[int(count)].append(int(threshold))
    if not candidates_by_count:
        return None
    preferred_counts = [
        int(count)
        for count in (target_counts or ())
        if int(count) in candidates_by_count
    ]
    available_counts = preferred_counts or sorted(int(count) for count in candidates_by_count.keys())
    target_count = int(available_counts[int(rng.randrange(len(available_counts)))])
    thresholds = candidates_by_count[int(target_count)]
    threshold = int(thresholds[int(rng.randrange(len(thresholds)))])
    return int(threshold), int(target_count)


def _axis_size(*, axis: str, x_labels: Sequence[str], series_labels: Sequence[str]) -> int:
    if str(axis) == "category":
        return len(x_labels)
    if str(axis) == "series":
        return len(series_labels)
    raise ValueError(f"unknown 3D bar axis {axis!r}")


def _axis_label(*, axis: str, x_labels: Sequence[str], series_labels: Sequence[str], index: int) -> str:
    return str(x_labels[int(index)] if str(axis) == "category" else series_labels[int(index)])


def _axis_total(*, axis: str, values: GridValues, index: int) -> int:
    if str(axis) == "category":
        return int(sum(int(value) for value in values[int(index)]))
    return int(sum(int(row[int(index)]) for row in values))


def _axis_annotation_ids(*, axis: str, x_count: int, series_count: int, index: int) -> tuple[str, ...]:
    if str(axis) == "category":
        return tuple(bar_id(int(index), series_index) for series_index in range(int(series_count)))
    return tuple(bar_id(x_index, int(index)) for x_index in range(int(x_count)))


def select_axis_total(
    *,
    axis: str,
    x_labels: tuple[str, ...],
    series_labels: tuple[str, ...],
    values: GridValues,
    instance_seed: int,
) -> _Selection:
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.{axis}.total")
    selected_index = int(rng.randrange(_axis_size(axis=str(axis), x_labels=x_labels, series_labels=series_labels)))
    trace_key = "category" if str(axis) == "category" else "series"
    selected_values = (
        [int(values[selected_index][series_index]) for series_index in range(len(series_labels))]
        if str(axis) == "category"
        else [int(values[x_index][selected_index]) for x_index in range(len(x_labels))]
    )
    return _Selection(
        answer=_axis_total(axis=str(axis), values=values, index=int(selected_index)),
        annotation_bar_ids=_axis_annotation_ids(
            axis=str(axis),
            x_count=len(x_labels),
            series_count=len(series_labels),
            index=int(selected_index),
        ),
        trace={
            f"{trace_key}_label": _axis_label(
                axis=str(axis),
                x_labels=x_labels,
                series_labels=series_labels,
                index=int(selected_index),
            ),
            f"{trace_key}_index": int(selected_index),
            "selected_values": selected_values,
        },
    )


def select_axis_total_gap(
    *,
    axis: str,
    x_labels: tuple[str, ...],
    series_labels: tuple[str, ...],
    values: GridValues,
    instance_seed: int,
) -> _Selection:
    """Bind a two-axis total gap and preserve operand grouping for annotation.

    The returned keyed groups mirror the visual operands: category gaps group
    by series label, while series gaps group by category label.
    """

    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.{axis}.total_gap")
    count = _axis_size(axis=str(axis), x_labels=x_labels, series_labels=series_labels)
    totals = [_axis_total(axis=str(axis), values=values, index=index) for index in range(int(count))]
    pairs = [(left, right) for left in range(int(count)) for right in range(left + 1, int(count)) if totals[left] != totals[right]]
    if not pairs:
        raise ValueError(f"{axis} totals are not distinct enough for a gap selection")
    left_index, right_index = pairs[int(rng.randrange(len(pairs)))]
    trace_key = "category" if str(axis) == "category" else "series"
    if str(axis) == "category":
        annotation_groups = {
            str(x_labels[left_index]): tuple(
                bar_id(int(left_index), int(series_index))
                for series_index in range(len(series_labels))
            ),
            str(x_labels[right_index]): tuple(
                bar_id(int(right_index), int(series_index))
                for series_index in range(len(series_labels))
            ),
        }
    elif str(axis) == "series":
        annotation_groups = {
            str(series_labels[left_index]): tuple(
                bar_id(int(x_index), int(left_index))
                for x_index in range(len(x_labels))
            ),
            str(series_labels[right_index]): tuple(
                bar_id(int(x_index), int(right_index))
                for x_index in range(len(x_labels))
            ),
        }
    else:
        raise ValueError(f"unknown 3D bar axis {axis!r}")
    annotation_ids = tuple(
        str(bar_id_value)
        for bar_ids in annotation_groups.values()
        for bar_id_value in bar_ids
    )
    return _Selection(
        answer=abs(int(totals[left_index]) - int(totals[right_index])),
        annotation_bar_ids=annotation_ids,
        annotation_kind="point_set_map",
        annotation_bar_id_groups=annotation_groups,
        trace={
            f"{trace_key}_label_a": _axis_label(axis=str(axis), x_labels=x_labels, series_labels=series_labels, index=int(left_index)),
            f"{trace_key}_label_b": _axis_label(axis=str(axis), x_labels=x_labels, series_labels=series_labels, index=int(right_index)),
            f"{trace_key}_index_a": int(left_index),
            f"{trace_key}_index_b": int(right_index),
            f"{trace_key}_total_a": int(totals[left_index]),
            f"{trace_key}_total_b": int(totals[right_index]),
        },
    )


def select_category_extremum_gap(
    *,
    x_labels: tuple[str, ...],
    series_labels: tuple[str, ...],
    values: GridValues,
    instance_seed: int,
) -> _Selection:
    """Select one category with unique extrema and bind highest/lowest witnesses."""

    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.category.extremum_gap")
    valid_categories = []
    for x_index in range(len(x_labels)):
        row_values = [int(value) for value in values[x_index]]
        max_value = max(row_values)
        min_value = min(row_values)
        if row_values.count(max_value) == 1 and row_values.count(min_value) == 1 and max_value > min_value:
            valid_categories.append(x_index)
    if not valid_categories:
        raise ValueError("category values are not distinct enough for an extremum gap")
    x_index = int(valid_categories[int(rng.randrange(len(valid_categories)))])
    category_values = [int(values[x_index][series_index]) for series_index in range(len(series_labels))]
    max_value = int(max(category_values))
    min_value = int(min(category_values))
    max_series_index = int(category_values.index(max_value))
    min_series_index = int(category_values.index(min_value))
    max_bar_id = bar_id(int(x_index), int(max_series_index))
    min_bar_id = bar_id(int(x_index), int(min_series_index))
    return _Selection(
        answer=int(max_value - min_value),
        annotation_bar_ids=(max_bar_id, min_bar_id),
        annotation_kind="point_map",
        annotation_bar_id_groups={
            "highest": (max_bar_id,),
            "lowest": (min_bar_id,),
        },
        trace={
            "category_label": str(x_labels[x_index]),
            "category_index": int(x_index),
            "category_values": list(category_values),
            "max_value": max_value,
            "min_value": min_value,
            "max_series_label": str(series_labels[max_series_index]),
            "min_series_label": str(series_labels[min_series_index]),
            "max_series_index": max_series_index,
            "min_series_index": min_series_index,
        },
    )


def select_axis_threshold_count(
    *,
    axis: str,
    comparison_phrase: str,
    want_at_least: bool,
    x_labels: tuple[str, ...],
    series_labels: tuple[str, ...],
    values: GridValues,
    params: Mapping[str, Any],
    instance_seed: int,
) -> _Selection:
    """Bind one threshold-count objective over a chosen semantic axis.

    The caller chooses category-vs-series and comparison direction; this helper
    only samples a nontrivial threshold/count pair and returns checked bars.
    """

    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.{axis}.threshold_count")
    selected_index = int(rng.randrange(_axis_size(axis=str(axis), x_labels=x_labels, series_labels=series_labels)))
    selected_values = (
        [int(values[selected_index][series_index]) for series_index in range(len(series_labels))]
        if str(axis) == "category"
        else [int(values[x_index][selected_index]) for x_index in range(len(x_labels))]
    )
    target_count, target_count_probabilities = sample_condition_target_count(
        params,
        instance_seed=int(instance_seed),
        axis_size=len(selected_values),
        namespace=f"{SCENE_NAMESPACE}.{axis}.threshold_count.target_count",
    )
    threshold_info = valid_threshold(
        selected_values,
        want_at_least=bool(want_at_least),
        rng=rng,
        target_counts=(int(target_count),),
    )
    if threshold_info is None:
        raise ValueError(f"{axis} threshold selection has no nontrivial count")
    threshold, answer = threshold_info
    trace_key = "category" if str(axis) == "category" else "series"
    matching_offsets = [
        int(index)
        for index, value in enumerate(selected_values)
        if (int(value) >= int(threshold) if bool(want_at_least) else int(value) < int(threshold))
    ]
    if str(axis) == "category":
        matched_bar_ids = tuple(bar_id(int(selected_index), series_index) for series_index in matching_offsets)
    elif str(axis) == "series":
        matched_bar_ids = tuple(bar_id(x_index, int(selected_index)) for x_index in matching_offsets)
    else:
        raise ValueError(f"unknown 3D bar axis {axis!r}")
    checked_bar_ids = _axis_annotation_ids(
        axis=str(axis),
        x_count=len(x_labels),
        series_count=len(series_labels),
        index=int(selected_index),
    )
    if int(answer) != len(matched_bar_ids):
        raise ValueError("threshold annotation bars do not match the sampled answer count")
    return _Selection(
        answer=int(answer),
        annotation_bar_ids=matched_bar_ids,
        trace={
            f"{trace_key}_label": _axis_label(
                axis=str(axis),
                x_labels=x_labels,
                series_labels=series_labels,
                index=int(selected_index),
            ),
            f"{trace_key}_index": int(selected_index),
            "comparison_phrase": str(comparison_phrase),
            "threshold": int(threshold),
            "target_count": int(target_count),
            "target_count_probabilities": dict(target_count_probabilities),
            "selected_values": list(selected_values),
            "checked_bar_ids": [str(bar_id_value) for bar_id_value in checked_bar_ids],
            "matched_bar_ids": [str(bar_id_value) for bar_id_value in matched_bar_ids],
        },
    )


def select_pairwise_series_greater_count(
    *,
    x_labels: tuple[str, ...],
    series_labels: tuple[str, ...],
    values: GridValues,
    target_count: int,
    instance_seed: int,
) -> _Selection:
    """Bind a pair of series whose greater-than count matches the target."""

    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.series_pairwise_greater_count")
    pairs_by_count: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for series_a in range(len(series_labels)):
        for series_b in range(len(series_labels)):
            if series_a == series_b:
                continue
            count = sum(
                1
                for x_index in range(len(x_labels))
                if int(values[x_index][series_a]) > int(values[x_index][series_b])
            )
            if 1 <= int(count) <= int(len(x_labels)) - 1:
                pairs_by_count[int(count)].append((int(series_a), int(series_b)))
    if int(target_count) not in pairs_by_count:
        raise ValueError("series comparison target count is unavailable")
    pairs = pairs_by_count[int(target_count)]
    series_a, series_b = pairs[int(rng.randrange(len(pairs)))]
    matched_x_indices = [
        int(x_index)
        for x_index in range(len(x_labels))
        if int(values[x_index][series_a]) > int(values[x_index][series_b])
    ]
    annotation_pairs = tuple(
        (
            bar_id(int(x_index), int(series_a)),
            bar_id(int(x_index), int(series_b)),
        )
        for x_index in matched_x_indices
    )
    annotation_ids = tuple(str(bar_id_value) for pair in annotation_pairs for bar_id_value in pair)
    checked_bar_ids = (
        _axis_annotation_ids(axis="series", x_count=len(x_labels), series_count=len(series_labels), index=int(series_a))
        + _axis_annotation_ids(axis="series", x_count=len(x_labels), series_count=len(series_labels), index=int(series_b))
    )
    if int(target_count) != len(annotation_pairs):
        raise ValueError("pairwise annotation pairs do not match the sampled answer count")
    return _Selection(
        answer=int(target_count),
        annotation_bar_ids=annotation_ids,
        annotation_kind="segment_set",
        annotation_bar_id_pairs=annotation_pairs,
        trace={
            "series_label_a": str(series_labels[series_a]),
            "series_label_b": str(series_labels[series_b]),
            "series_index_a": int(series_a),
            "series_index_b": int(series_b),
            "target_count": int(target_count),
            "selected_values_a": [int(values[x_index][series_a]) for x_index in range(len(x_labels))],
            "selected_values_b": [int(values[x_index][series_b]) for x_index in range(len(x_labels))],
            "matched_category_labels": [str(x_labels[x_index]) for x_index in matched_x_indices],
            "matched_category_indices": [int(x_index) for x_index in matched_x_indices],
            "checked_bar_ids": [str(bar_id_value) for bar_id_value in checked_bar_ids],
            "matched_bar_ids": [str(bar_id_value) for bar_id_value in annotation_ids],
        },
    )


def select_series_interval_total(
    *,
    x_labels: tuple[str, ...],
    series_labels: tuple[str, ...],
    values: GridValues,
    params: Mapping[str, Any],
    instance_seed: int,
) -> _Selection:
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.series_interval_total")
    series_index = int(rng.randrange(len(series_labels)))
    start_index, end_index, span_range = choose_interval(x_count=len(x_labels), params=params, instance_seed=int(instance_seed))
    selected_values = [int(values[x_index][series_index]) for x_index in range(int(start_index), int(end_index) + 1)]
    return _Selection(
        answer=int(sum(selected_values)),
        annotation_bar_ids=tuple(bar_id(x_index, series_index) for x_index in range(int(start_index), int(end_index) + 1)),
        trace={
            "series_label": str(series_labels[series_index]),
            "series_index": int(series_index),
            "start_category_label": str(x_labels[start_index]),
            "end_category_label": str(x_labels[end_index]),
            "start_category_index": int(start_index),
            "end_category_index": int(end_index),
            "interval_category_count": int(end_index) - int(start_index) + 1,
            "interval_category_count_range": list(span_range),
            "selected_values": selected_values,
        },
    )


def select_series_scoped_total(
    *,
    interval_scope: bool,
    x_labels: tuple[str, ...],
    series_labels: tuple[str, ...],
    values: GridValues,
    params: Mapping[str, Any],
    instance_seed: int,
) -> _Selection:
    """Bind either all-category or contiguous-interval total for one series."""

    if bool(interval_scope):
        return select_series_interval_total(
            x_labels=x_labels,
            series_labels=series_labels,
            values=values,
            params=params,
            instance_seed=int(instance_seed),
        )
    return select_axis_total(
        axis="series",
        x_labels=x_labels,
        series_labels=series_labels,
        values=values,
        instance_seed=int(instance_seed),
    )


def bar_id(x_index: int, series_index: int) -> str:
    return f"bar_{int(x_index)}_{int(series_index)}"


def make_dataset(
    *,
    x_labels: Sequence[str],
    series_labels: Sequence[str],
    values: Sequence[Sequence[int]],
    selection: _Selection,
    instance_seed: int,
) -> _Dataset:
    palette = _sample_palette(instance_seed=int(instance_seed), count=len(series_labels))
    bars: list[_BarCell] = []
    for x_index, x_label in enumerate(x_labels):
        for series_index, series_label in enumerate(series_labels):
            bars.append(
                _BarCell(
                    bar_id=bar_id(int(x_index), int(series_index)),
                    x_label=str(x_label),
                    series_label=str(series_label),
                    x_index=int(x_index),
                    series_index=int(series_index),
                    value=int(values[x_index][series_index]),
                    color_rgb=tuple(int(channel) for channel in palette[int(series_index) % len(palette)]),
                )
            )
    return _Dataset(
        x_labels=tuple(str(label) for label in x_labels),
        series_labels=tuple(str(label) for label in series_labels),
        bars=tuple(bars),
        selection=selection,
    )


def sample_dataset_with_selection(
    *,
    params: Mapping[str, Any],
    condition_scope: bool,
    pairwise_target_count: int | None,
    instance_seed: int,
    select: SelectionFactory,
) -> tuple[_Dataset, GridRanges, dict[str, Any]]:
    """Sample a grid, let a public task's selector bind the answer, then package bars."""

    x_labels, series_labels, values, ranges = sample_grid(
        params,
        condition_scope=bool(condition_scope),
        pairwise_target_count=pairwise_target_count,
        instance_seed=int(instance_seed),
    )
    selection = select(x_labels, series_labels, values)
    dataset = make_dataset(
        x_labels=x_labels,
        series_labels=series_labels,
        values=values,
        selection=selection,
        instance_seed=int(instance_seed),
    )
    return dataset, dict(ranges), dict(dataset.selection.trace)
