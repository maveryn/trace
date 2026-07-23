"""Neutral sampling primitives for curve-panel chart tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from ....shared.config_defaults import group_default, resolve_required_int_bounds
from ...shared.label_assets import (
    resolve_chart_entity_labels,
    resolve_chart_panel_labels,
    validate_chart_label_namespaces,
)
from .defaults import (
    GENERATION_DEFAULTS,
    SCENE_NAMESPACE,
    balanced_choice,
    generation_int,
    method_count,
    method_count_max,
    palette,
    panel_answer_index_support,
    panel_count,
    without_sample_cursor,
)
from .state import Curve, Intersection, Panel, RGB, ThresholdCrossing


@dataclass(frozen=True)
class CurvePanelBaseSample:
    """Neutral shared axes, labels, colors, and baseline values for one objective."""

    x_values: Tuple[int, ...]
    y_min: int
    y_max: int
    panel_labels: Tuple[str, ...]
    method_labels: Tuple[str, ...]
    panel_label_meta: Dict[str, Any]
    colors: Tuple[RGB, ...]
    values: Dict[str, Dict[str, List[int]]]
    answer_panel_index: int
    answer_panel: str
    non_answer_params: Mapping[str, Any]


def point_id(panel_label: str, method_label: str, x_value: int) -> str:
    """Return the rendered marker id for one panel/method/x tuple."""

    return f"{str(panel_label)}|{str(method_label)}|{int(x_value)}"


def intersection_id(
    panel_label: str, method_a_label: str, method_b_label: str, index: int
) -> str:
    """Return the rendered intersection id for one pairwise crossing."""

    return f"{str(panel_label)}|{str(method_a_label)}|{str(method_b_label)}|intersection:{int(index)}"


def threshold_crossing_id(panel_label: str, method_label: str, index: int) -> str:
    """Return the rendered threshold-crossing id for one curve crossing."""

    return f"{str(panel_label)}|{str(method_label)}|threshold_crossing:{int(index)}"


def method_labels_for_seed(*, count: int, instance_seed: int) -> Tuple[str, ...]:
    """Sample visible curve/method labels."""

    labels = resolve_chart_entity_labels(
        spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.method_labels"),
        count=int(count),
        min_chars=2,
        max_chars=7,
        allow_spaces=False,
    ).labels
    return tuple(str(label) for label in labels)


def _resolved_label_metadata(resolved: Any) -> Dict[str, Any]:
    return {
        "label_variant": str(resolved.label_variant),
        "label_pool_kind": str(resolved.label_pool_kind),
        "label_source_kind": str(resolved.label_source_kind),
        "label_bucket": str(resolved.label_bucket),
        "label_manifest": str(resolved.label_manifest),
        "label_filter": dict(resolved.label_filter),
        "label_bucket_probabilities": dict(resolved.label_bucket_probabilities),
    }


def _has_panel_prefix_label(labels: Sequence[str]) -> bool:
    """Return whether any visible subplot label reads like a panel prefix."""

    return any(str(label).strip().lower().startswith("panel") for label in labels)


def panel_labels_for_seed(
    *,
    count: int,
    params: Mapping[str, Any],
    instance_seed: int,
    reserved_labels: Sequence[str] = (),
) -> Tuple[Tuple[str, ...], Dict[str, Any]]:
    """Sample visible panel labels without colliding with method labels."""

    variant_weights = params.get(
        "panel_label_variant_weights",
        group_default(
            GENERATION_DEFAULTS,
            "panel_label_variant_weights",
            {
                "subplot_letters": 1.0,
                "technical_topics": 0.75,
                "condition_labels": 0.5,
                "named_compact": 0.5,
                "temporal_sequence": 0.25,
            },
        ),
    )
    resolved = None
    for attempt in range(12):
        rng_namespace = (
            f"{SCENE_NAMESPACE}.panel_labels"
            if int(attempt) == 0
            else f"{SCENE_NAMESPACE}.panel_labels.retry.{int(attempt)}"
        )
        candidate = resolve_chart_panel_labels(
            spawn_rng(int(instance_seed), rng_namespace),
            count=int(count),
            min_chars=1,
            max_chars=10,
            allow_spaces=False,
            variant_weights=variant_weights,
            reserved_labels=tuple(str(label) for label in reserved_labels),
        )
        if not _has_panel_prefix_label(candidate.labels):
            resolved = candidate
            break
    if resolved is None:
        resolved = resolve_chart_panel_labels(
            spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.panel_labels.fallback"),
            count=int(count),
            min_chars=1,
            max_chars=10,
            allow_spaces=False,
            variant_weights={"subplot_letters": 1.0},
            reserved_labels=tuple(str(label) for label in reserved_labels),
        )
    collision_check = validate_chart_label_namespaces(
        panel_labels=resolved.labels,
        other_label_groups={
            "method_labels": tuple(str(label) for label in reserved_labels)
        },
        context="scientific curve-panel labels",
    )
    return tuple(str(label) for label in resolved.labels), {
        "panel_label_resolution": _resolved_label_metadata(resolved),
        "panel_label_collision_check": dict(collision_check),
    }


def x_values(
    params: Mapping[str, Any], *, instance_seed: int, min_required: int = 4
) -> Tuple[int, ...]:
    """Sample evenly spaced numeric x-axis values."""

    low, high = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="x_tick_count_min",
        max_key="x_tick_count_max",
        fallback_min=6,
        fallback_max=12,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    low = max(4, int(min_required), int(low))
    high = max(int(low), int(high))
    count = int(
        balanced_choice(
            list(range(int(low), int(high) + 1)),
            params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.x_tick_count",
        )
    )
    step_min = int(generation_int(params, "x_step_min", 5))
    step_max = int(generation_int(params, "x_step_max", 20))
    span_min = int(generation_int(params, "x_span_min", 50))
    span_max = int(generation_int(params, "x_span_max", 100))
    if int(step_min) > int(step_max):
        raise ValueError("x_step_min must be <= x_step_max")
    if int(span_min) > int(span_max):
        raise ValueError("x_span_min must be <= x_span_max")
    step_support = [
        int(step)
        for step in range(max(1, int(step_min)), int(step_max) + 1)
        if int(span_min) <= int(step) * max(1, int(count) - 1) <= int(span_max)
    ]
    if not step_support:
        raise ValueError("x step/span support is empty for sampled x_tick_count")
    step = int(
        balanced_choice(
            step_support,
            params,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.x_step:{int(count)}",
        )
    )
    return tuple(int(index) * int(step) for index in range(int(count)))


def threshold_for_x_values(
    params: Mapping[str, Any], *, instance_seed: int, x_axis_values: Sequence[int]
) -> int:
    """Sample a visible y-threshold value."""

    raw = params.get(
        "threshold_values", GENERATION_DEFAULTS.get("threshold_values", (45, 55, 65))
    )
    support: Tuple[int, ...] = (45, 55, 65)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        values = tuple(int(value) for value in raw if 10 <= int(value) <= 90)
        if values:
            support = values
    return int(
        balanced_choice(
            support,
            without_sample_cursor(params),
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.threshold:{len(x_axis_values)}:{int(max(x_axis_values)) if x_axis_values else 0}",
        )
    )


def random_curve_values(
    *, rng: Any, count: int, value_min: int, value_max: int
) -> List[int]:
    """Sample a gently varying curve."""

    current = int(rng.randint(28, 72))
    values: List[int] = []
    for _ in range(int(count)):
        current += int(rng.randint(-15, 15))
        current = max(int(value_min) + 8, min(int(value_max) - 8, int(current)))
        values.append(int(current))
    return values


def make_random_panels(
    *,
    panel_labels: Sequence[str],
    method_labels: Sequence[str],
    x_count: int,
    instance_seed: int,
    namespace: str,
    value_min: int,
    value_max: int,
) -> Dict[str, Dict[str, List[int]]]:
    """Build random baseline curves for every panel and method."""

    values: Dict[str, Dict[str, List[int]]] = {}
    for panel in panel_labels:
        values[str(panel)] = {}
        for method in method_labels:
            rng = spawn_rng(
                int(instance_seed), f"{str(namespace)}:{str(panel)}:{str(method)}"
            )
            values[str(panel)][str(method)] = random_curve_values(
                rng=rng,
                count=int(x_count),
                value_min=int(value_min),
                value_max=int(value_max),
            )
    return values


def choose_method_label(
    *,
    method_labels: Sequence[str],
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> str:
    """Sample one method label without advancing task-level answer balancing."""

    return str(
        balanced_choice(
            tuple(str(label) for label in method_labels),
            without_sample_cursor(params),
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
    )


def base_curve_panel_sample(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    min_x_tick_count: int = 4,
    min_panel_count: int = 1,
    min_method_count: int = 1,
) -> CurvePanelBaseSample:
    """Sample neutral curve-panel state before an objective imposes constraints."""

    answer_panel_index = int(
        balanced_choice(
            panel_answer_index_support(params),
            params,
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.answer",
        )
    )
    non_answer_params = without_sample_cursor(params)
    (
        sampled_x_values,
        y_min,
        y_max,
        _panel_total,
        panel_labels,
        method_labels,
        panel_label_meta,
    ) = common_axes(
        params,
        instance_seed=int(instance_seed),
        min_x_tick_count=int(min_x_tick_count),
        min_panel_count=max(int(min_panel_count), int(answer_panel_index) + 1),
        min_method_count=int(min_method_count),
    )
    values = make_random_panels(
        panel_labels=panel_labels,
        method_labels=method_labels,
        x_count=len(sampled_x_values),
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.values",
        value_min=int(y_min),
        value_max=int(y_max),
    )
    return CurvePanelBaseSample(
        x_values=tuple(sampled_x_values),
        y_min=int(y_min),
        y_max=int(y_max),
        panel_labels=tuple(panel_labels),
        method_labels=tuple(method_labels),
        panel_label_meta=dict(panel_label_meta),
        colors=tuple(palette(params)),
        values=values,
        answer_panel_index=int(answer_panel_index),
        answer_panel=str(panel_labels[int(answer_panel_index)]),
        non_answer_params=non_answer_params,
    )


def panels_from_values(
    *,
    values_by_panel_method: Mapping[str, Mapping[str, Sequence[int]]],
    panel_labels: Sequence[str],
    method_labels: Sequence[str],
    colors: Sequence[RGB],
    omitted_panel_methods: Mapping[str, Sequence[str]] | None = None,
) -> Tuple[Panel, ...]:
    """Convert sampled values into renderable panels."""

    omitted = {
        str(panel): {str(method) for method in methods}
        for panel, methods in dict(omitted_panel_methods or {}).items()
    }
    panels: List[Panel] = []
    for panel in panel_labels:
        curves: List[Curve] = []
        for index, method in enumerate(method_labels):
            if str(method) in omitted.get(str(panel), set()):
                continue
            curves.append(
                Curve(
                    method_label=str(method),
                    values=tuple(
                        int(value)
                        for value in values_by_panel_method[str(panel)][str(method)]
                    ),
                    color_rgb=tuple(colors[int(index) % len(colors)]),
                )
            )
        panels.append(Panel(panel_label=str(panel), curves=tuple(curves)))
    return tuple(panels)


def common_axes(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    min_x_tick_count: int = 4,
    min_panel_count: int = 1,
    min_method_count: int = 1,
) -> Tuple[
    Tuple[int, ...], int, int, int, Tuple[str, ...], Tuple[str, ...], Dict[str, Any]
]:
    """Sample shared x/y bounds, panel labels, and method labels."""

    non_answer_params = without_sample_cursor(params)
    panel_total = panel_count(
        non_answer_params,
        instance_seed=int(instance_seed),
        min_required=int(min_panel_count),
    )
    method_total = method_count(
        non_answer_params,
        instance_seed=int(instance_seed),
        min_required=int(min_method_count),
    )
    sampled_x_values = x_values(
        non_answer_params,
        instance_seed=int(instance_seed),
        min_required=int(min_x_tick_count),
    )
    y_min = int(generation_int(params, "y_value_min", 0))
    y_max = int(generation_int(params, "y_value_max", 100))
    if int(y_min) >= int(y_max):
        raise ValueError("y_value_min must be lower than y_value_max")
    sampled_method_labels = method_labels_for_seed(
        count=int(method_total), instance_seed=int(instance_seed)
    )
    sampled_panel_labels, panel_label_meta = panel_labels_for_seed(
        count=int(panel_total),
        params=params,
        instance_seed=int(instance_seed),
        reserved_labels=sampled_method_labels,
    )
    return (
        tuple(sampled_x_values),
        int(y_min),
        int(y_max),
        int(panel_total),
        tuple(sampled_panel_labels),
        tuple(sampled_method_labels),
        dict(panel_label_meta),
    )


def threshold_panel_context(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
    min_x_tick_count: int = 5,
) -> Tuple[
    Tuple[int, ...],
    int,
    int,
    Tuple[str, ...],
    Tuple[str, ...],
    Dict[str, Any],
    Tuple[RGB, ...],
    Any,
    str,
    int,
    Dict[str, Dict[str, List[int]]],
    Mapping[str, Any],
]:
    """Sample a subplot, threshold, and base values for threshold objectives."""

    non_answer_params = without_sample_cursor(params)
    (
        sampled_x_values,
        y_min,
        y_max,
        _panel_total,
        panel_labels,
        method_labels,
        panel_label_meta,
    ) = common_axes(
        params, instance_seed=int(instance_seed), min_x_tick_count=int(min_x_tick_count)
    )
    colors = palette(params)
    rng = spawn_rng(int(instance_seed), str(namespace))
    query_panel = str(
        balanced_choice(
            panel_labels,
            non_answer_params,
            instance_seed=int(instance_seed),
            namespace=f"{namespace}.panel",
        )
    )
    threshold = threshold_for_x_values(
        params, instance_seed=int(instance_seed), x_axis_values=sampled_x_values
    )
    values = make_random_panels(
        panel_labels=panel_labels,
        method_labels=method_labels,
        x_count=len(sampled_x_values),
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.values",
        value_min=int(y_min),
        value_max=int(y_max),
    )
    return (
        tuple(sampled_x_values),
        int(y_min),
        int(y_max),
        tuple(panel_labels),
        tuple(method_labels),
        dict(panel_label_meta),
        tuple(colors),
        rng,
        str(query_panel),
        int(threshold),
        values,
        non_answer_params,
    )


def side_value(
    *,
    rng: Any,
    threshold: int,
    side: str,
    y_min: int,
    y_max: int,
    margin_min: int = 7,
    margin_max: int = 28,
) -> int:
    """Sample a value constrained to one side of a threshold."""

    margin = int(rng.randint(int(margin_min), int(margin_max)))
    if str(side) == "above":
        return max(int(y_min) + 5, min(int(y_max) - 5, int(threshold) + int(margin)))
    if str(side) == "below":
        return max(int(y_min) + 5, min(int(y_max) - 5, int(threshold) - int(margin)))
    raise ValueError(f"unsupported threshold side: {side}")


def make_single_threshold_crossing_curve(
    *,
    rng: Any,
    x_count: int,
    threshold: int,
    direction: str,
    crossing_index: int,
    y_min: int,
    y_max: int,
) -> List[int]:
    """Build a curve with exactly one threshold crossing."""

    if int(x_count) < 3:
        raise ValueError("x_count must be at least 3 for threshold crossing curves")
    crossing_index = max(1, min(int(x_count) - 1, int(crossing_index)))
    before_side = "below" if str(direction) == "upward" else "above"
    after_side = "above" if str(direction) == "upward" else "below"
    values: List[int] = []
    for index in range(int(x_count)):
        side = before_side if int(index) < int(crossing_index) else after_side
        values.append(
            side_value(
                rng=rng,
                threshold=int(threshold),
                side=str(side),
                y_min=int(y_min),
                y_max=int(y_max),
                margin_min=10,
                margin_max=30,
            )
        )
    return values


def threshold_crossing_points(
    *,
    panel_label: str,
    method_label: str,
    x_axis_values: Sequence[int],
    values: Sequence[int],
    threshold: int,
    direction: str,
) -> Tuple[ThresholdCrossing, ...]:
    """Compute interpolated threshold crossing points for one curve."""

    crossings: List[ThresholdCrossing] = []
    for index in range(len(x_axis_values) - 1):
        y0 = float(values[int(index)])
        y1 = float(values[int(index) + 1])
        requested_crossing = (
            str(direction) == "upward" and y0 < float(threshold) < y1
        ) or (str(direction) == "downward" and y0 > float(threshold) > y1)
        if not requested_crossing:
            continue
        x0 = float(x_axis_values[int(index)])
        x1 = float(x_axis_values[int(index) + 1])
        t = (float(threshold) - y0) / (y1 - y0)
        x_value = x0 + (t * (x1 - x0))
        crossings.append(
            ThresholdCrossing(
                crossing_id=threshold_crossing_id(
                    str(panel_label), str(method_label), len(crossings)
                ),
                panel_label=str(panel_label),
                method_label=str(method_label),
                x_value=float(x_value),
                y_value=float(threshold),
                direction=str(direction),
            )
        )
    return tuple(crossings)


def replace_curve_interval(
    curve_values: List[int],
    *,
    start_index: int,
    end_index: int,
    start_value: int,
    end_value: int,
    rng: Any,
    y_min: int,
    y_max: int,
) -> None:
    """Replace one curve span with a smooth interval from start to end."""

    curve_values[int(start_index)] = int(start_value)
    curve_values[int(end_index)] = int(end_value)
    span = max(1, int(end_index) - int(start_index))
    for index in range(int(start_index) + 1, int(end_index)):
        alpha = float(index - int(start_index)) / float(span)
        base = (1.0 - alpha) * float(start_value) + alpha * float(end_value)
        curve_values[int(index)] = max(
            int(y_min) + 5, min(int(y_max) - 5, int(round(base + rng.randint(-4, 4))))
        )


def build_intersection_curves(
    *,
    x_axis_values: Sequence[int],
    target_count: int,
    instance_seed: int,
    namespace: str,
) -> Tuple[List[int], List[int], List[Intersection]]:
    """Build two curves with a target number of sign changes."""

    rng = spawn_rng(int(instance_seed), str(namespace))
    interval_count = len(x_axis_values) - 1
    change_positions = list(range(1, int(interval_count) + 1))
    rng.shuffle(change_positions)
    change_set = set(
        int(position) for position in change_positions[: int(target_count)]
    )
    signs: List[int] = []
    sign = 1
    for index in range(len(x_axis_values)):
        if int(index) > 0 and int(index) in change_set:
            sign *= -1
        signs.append(int(sign))

    method_a: List[int] = []
    method_b: List[int] = []
    for index, sign_value in enumerate(signs):
        base = int(50 + rng.randint(-7, 7))
        magnitude = int(12 + rng.randint(0, 14))
        method_a.append(max(8, min(92, int(base) + (int(sign_value) * int(magnitude)))))
        method_b.append(max(8, min(92, int(base) - (int(sign_value) * int(magnitude)))))
    return method_a, method_b, []


def intersection_points(
    *,
    panel_label: str,
    method_a_label: str,
    method_b_label: str,
    x_axis_values: Sequence[int],
    values_a: Sequence[int],
    values_b: Sequence[int],
) -> Tuple[Intersection, ...]:
    """Compute interpolated intersections between two curves."""

    intersections: List[Intersection] = []
    for index in range(len(x_axis_values) - 1):
        d0 = float(values_a[int(index)] - values_b[int(index)])
        d1 = float(values_a[int(index) + 1] - values_b[int(index) + 1])
        if d0 == 0.0 or d1 == 0.0 or (d0 > 0) == (d1 > 0):
            continue
        t = abs(d0) / (abs(d0) + abs(d1))
        x0 = float(x_axis_values[int(index)])
        x1 = float(x_axis_values[int(index) + 1])
        y0 = float(values_a[int(index)])
        y1 = float(values_a[int(index) + 1])
        x_value = x0 + (t * (x1 - x0))
        y_value = y0 + (t * (y1 - y0))
        intersections.append(
            Intersection(
                intersection_id=intersection_id(
                    str(panel_label),
                    str(method_a_label),
                    str(method_b_label),
                    len(intersections),
                ),
                panel_label=str(panel_label),
                method_a_label=str(method_a_label),
                method_b_label=str(method_b_label),
                x_value=float(x_value),
                y_value=float(y_value),
            )
        )
    return tuple(intersections)


__all__ = [
    "balanced_choice",
    "base_curve_panel_sample",
    "build_intersection_curves",
    "choose_method_label",
    "common_axes",
    "CurvePanelBaseSample",
    "generation_int",
    "intersection_points",
    "make_random_panels",
    "make_single_threshold_crossing_curve",
    "method_count",
    "method_count_max",
    "method_labels_for_seed",
    "palette",
    "panel_answer_index_support",
    "panel_count",
    "panel_labels_for_seed",
    "panels_from_values",
    "point_id",
    "replace_curve_interval",
    "side_value",
    "threshold_crossing_points",
    "threshold_for_x_values",
    "without_sample_cursor",
    "x_values",
]
