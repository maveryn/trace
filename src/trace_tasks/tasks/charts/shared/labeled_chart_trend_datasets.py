"""Trend-structure, threshold-crossing, and interval-change datasets for labeled charts."""

from __future__ import annotations

from itertools import product
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from ....core.seed import spawn_rng
from ....core.sampling import uniform_choice
from ...shared.config_defaults import group_default
from .label_assets import sample_chart_labels
from .labeled_chart_defaults import LabeledChartDefaults
from .labeled_chart_values import (
    balanced_choice_from_values,
    resolve_mark_count_bounds,
    resolve_value_bounds,
    sorted_labels,
)
from .labeled_chart_variants import PIE_LIKE_SCENE_VARIANTS, SceneVariant
from .labeled_chart_sampling import (
    _sample_values_from_pool,
    choose_mark_count,
)

def _trend_signs_for_values(values: Sequence[int]) -> List[int]:
    """Return the +/- step-sign sequence for one ordered value list."""

    resolved_values = [int(value) for value in values]
    if len(resolved_values) < 2:
        raise ValueError("trend analysis requires at least two values")
    signs: List[int] = []
    for left, right in zip(resolved_values[:-1], resolved_values[1:]):
        delta = int(right) - int(left)
        if int(delta) == 0:
            raise ValueError("trend analysis requires strictly ordered adjacent values")
        signs.append(1 if int(delta) > 0 else -1)
    return [int(value) for value in signs]

def _turning_point_indices(signs: Sequence[int], *, positive_then_negative: bool) -> List[int]:
    """Return interior point indices for local peaks or troughs."""

    resolved_signs = [int(value) for value in signs]
    indices: List[int] = []
    for index in range(len(resolved_signs) - 1):
        left = int(resolved_signs[index])
        right = int(resolved_signs[index + 1])
        if bool(positive_then_negative):
            if int(left) > 0 and int(right) < 0:
                indices.append(int(index) + 1)
        else:
            if int(left) < 0 and int(right) > 0:
                indices.append(int(index) + 1)
    return [int(value) for value in indices]

def _unique_longest_run_point_indices(signs: Sequence[int], *, direction: int) -> List[int] | None:
    """Return the unique longest monotone run as point indices, or `None` on ties."""

    resolved_signs = [int(value) for value in signs]
    target_sign = 1 if int(direction) > 0 else -1
    runs: List[List[int]] = []
    index = 0
    while int(index) < len(resolved_signs):
        if int(resolved_signs[index]) != int(target_sign):
            index += 1
            continue
        run_start = int(index)
        while int(index) + 1 < len(resolved_signs) and int(resolved_signs[int(index) + 1]) == int(target_sign):
            index += 1
        run_end = int(index) + 1
        runs.append(list(range(int(run_start), int(run_end) + 1)))
        index += 1
    if not runs:
        return None
    max_length = max(len(run) for run in runs)
    winners = [list(run) for run in runs if len(run) == int(max_length)]
    if len(winners) != 1:
        return None
    return [int(value) for value in winners[0]]

def _summarize_trend_variant_from_signs(
    *,
    trend_variant: str,
    signs: Sequence[int],
) -> Tuple[int, List[int], Dict[str, Any]] | None:
    """Resolve one trend-query answer from a sign sequence."""

    resolved_signs = [int(value) for value in signs]
    if str(trend_variant) == "peak_count":
        indices = _turning_point_indices(resolved_signs, positive_then_negative=True)
        return int(len(indices)), [int(value) for value in indices], {"turning_kind": "peak"}
    if str(trend_variant) == "trough_count":
        indices = _turning_point_indices(resolved_signs, positive_then_negative=False)
        return int(len(indices)), [int(value) for value in indices], {"turning_kind": "trough"}
    if str(trend_variant) == "longest_increasing_streak":
        indices = _unique_longest_run_point_indices(resolved_signs, direction=1)
        if indices is None:
            return None
        return (
            int(len(indices)),
            [int(value) for value in indices],
            {"streak_direction": "increasing"},
        )
    if str(trend_variant) == "longest_decreasing_streak":
        indices = _unique_longest_run_point_indices(resolved_signs, direction=-1)
        if indices is None:
            return None
        return (
            int(len(indices)),
            [int(value) for value in indices],
            {"streak_direction": "decreasing"},
        )
    raise ValueError(f"unsupported trend_variant: {trend_variant}")

def _build_values_from_trend_signs(
    *,
    signs: Sequence[int],
    value_min: int,
    value_max: int,
    instance_seed: int,
    namespace: str,
) -> List[int]:
    """Construct one bounded integer value list that realizes the requested sign pattern."""

    resolved_signs = [int(value) for value in signs]
    prefix_values = [0]
    current = 0
    for sign in resolved_signs:
        current += 2 * int(sign)
        prefix_values.append(int(current))
    min_prefix = min(prefix_values)
    max_prefix = max(prefix_values)
    feasible_bases = [
        int(base)
        for base in range(int(value_min) - int(min_prefix), int(value_max) - int(max_prefix) + 1)
    ]
    if not feasible_bases:
        raise ValueError("no feasible base value for requested trend sign sequence")
    base_value = balanced_choice_from_values(
        feasible_bases,
        params={},
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    return [int(base_value) + int(prefix) for prefix in prefix_values]

def _trend_sign_value_span(signs: Sequence[int]) -> int:
    """Return the numeric span required by the fixed step trend construction."""

    current = 0
    prefix_values = [0]
    for sign in signs:
        current += 2 * int(sign)
        prefix_values.append(int(current))
    return int(max(prefix_values) - min(prefix_values))

def build_trend_structure_dataset_for_variant(
    *,
    trend_variant: str,
    scene_variant: SceneVariant,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: LabeledChartDefaults,
    task_id: str,
) -> Tuple[List[int], int, List[str], Dict[str, Any]]:
    """Construct one ordered labeled-chart dataset for trend-structure queries."""

    del scene_variant
    value_min, value_max = resolve_value_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
        instance_seed=int(instance_seed),
    )
    mark_count_min, mark_count_max = resolve_mark_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
    )
    candidate_mark_counts = [
        int(count)
        for count in range(int(mark_count_min), int(mark_count_max) + 1)
        if int(count) >= 3
    ]
    explicit_mark_count = params.get("mark_count")
    if explicit_mark_count is not None:
        candidate_mark_counts = [int(count) for count in candidate_mark_counts if int(count) == int(explicit_mark_count)]
    if not candidate_mark_counts:
        raise ValueError("trend structure tasks require at least three ordered marks")

    feasible_by_answer: Dict[int, List[Tuple[int, Tuple[int, ...], List[int], Dict[str, Any]]]] = {}
    for mark_count in candidate_mark_counts:
        sign_length = int(mark_count) - 1
        for sign_sequence in product((-1, 1), repeat=int(sign_length)):
            if _trend_sign_value_span(sign_sequence) > int(value_max) - int(value_min):
                continue
            resolved = _summarize_trend_variant_from_signs(
                trend_variant=str(trend_variant),
                signs=sign_sequence,
            )
            if resolved is None:
                continue
            answer_value, annotation_indices, metric_extras = resolved
            feasible_by_answer.setdefault(int(answer_value), []).append(
                (
                    int(mark_count),
                    tuple(int(value) for value in sign_sequence),
                    [int(value) for value in annotation_indices],
                    dict(metric_extras),
                )
            )
    if not feasible_by_answer:
        raise ValueError(f"no feasible trend support for variant={trend_variant}")

    if str(trend_variant) in {"peak_count", "trough_count"}:
        default_answer_min, default_answer_max = 0, max(int(value) for value in feasible_by_answer)
    elif str(trend_variant) in {"longest_increasing_streak", "longest_decreasing_streak"}:
        default_answer_min, default_answer_max = 2, max(int(value) for value in feasible_by_answer)
    else:
        raise ValueError(f"unsupported trend_variant: {trend_variant}")

    explicit_min = params.get("target_answer_min")
    explicit_max = params.get("target_answer_max")
    supported_answer_min = int(default_answer_min if explicit_min is None else explicit_min)
    supported_answer_max = int(default_answer_max if explicit_max is None else explicit_max)
    if int(supported_answer_min) > int(supported_answer_max):
        raise ValueError("target_answer_min must be <= target_answer_max")

    answer_candidates = [
        int(answer)
        for answer in sorted(feasible_by_answer.keys())
        if int(supported_answer_min) <= int(answer) <= int(supported_answer_max)
    ]
    if not answer_candidates:
        raise ValueError("no feasible target answers for requested trend answer range")
    target_rng = spawn_rng(int(instance_seed), f"{task_id}.target_answer:{str(trend_variant)}")
    target_answer = int(uniform_choice(target_rng, answer_candidates, sort_keys=True))
    candidate_sequences = list(feasible_by_answer[int(target_answer)])
    sequence_rng = spawn_rng(
        int(instance_seed),
        f"{task_id}.sequence:{str(trend_variant)}:{int(target_answer)}",
    )
    mark_count, sign_sequence, annotation_indices, metric_extras = sequence_rng.choice(candidate_sequences)

    values = _build_values_from_trend_signs(
        signs=sign_sequence,
        value_min=int(value_min),
        value_max=int(value_max),
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.values:{str(trend_variant)}:{int(target_answer)}",
    )
    labels = list(
        sample_chart_labels(
            count=int(mark_count),
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.labels:{str(trend_variant)}:{int(mark_count)}",
        )
    )
    ordered_annotation_labels = [str(labels[int(index)]) for index in annotation_indices]
    annotation_labels = sorted_labels(ordered_annotation_labels)

    trace_extras: Dict[str, Any] = {
        "value_min": int(value_min),
        "value_max": int(value_max),
        "target_answer_range": [int(supported_answer_min), int(supported_answer_max)],
        "mark_count_range": [int(mark_count_min), int(mark_count_max)],
        "target_answer": int(target_answer),
        "mark_count": int(mark_count),
        "labels": [str(label) for label in labels],
        "values_by_label": {str(label): int(value) for label, value in zip(labels, values)},
        "annotation_labels": list(annotation_labels),
        "ordered_annotation_labels": [str(label) for label in ordered_annotation_labels],
        "annotation_point_indices": [int(value) for value in annotation_indices],
        "step_signs": [int(value) for value in sign_sequence],
        "step_directions": ["up" if int(value) > 0 else "down" for value in sign_sequence],
        **dict(metric_extras),
    }
    return [int(value) for value in values], int(target_answer), annotation_labels, trace_extras

def _resolve_trend_threshold_bounds(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    value_min: int,
    value_max: int,
) -> Tuple[int, int]:
    """Resolve inclusive threshold bounds for ordered threshold-crossing charts."""

    axis_span = int(value_max) - int(value_min)
    configured_margin = params.get("threshold_edge_margin", group_default(gen_defaults, "threshold_edge_margin", None))
    if configured_margin is None:
        edge_margin = max(1, min(5, int(axis_span) // 4))
    else:
        edge_margin = max(1, int(configured_margin))
    fallback_min = int(value_min) + int(edge_margin)
    fallback_max = int(value_max) - int(edge_margin)
    value_window_raw = params.get("value_window_enabled", group_default(gen_defaults, "value_window_enabled", False))
    value_window_enabled = bool(value_window_raw)
    if isinstance(value_window_raw, str):
        value_window_enabled = str(value_window_raw).strip().lower() in {"1", "true", "yes", "on"}
    if bool(value_window_enabled) and "threshold_min" not in params and "threshold_max" not in params:
        threshold_min = int(fallback_min)
        threshold_max = int(fallback_max)
    else:
        threshold_min = int(params.get("threshold_min", group_default(gen_defaults, "threshold_min", fallback_min)))
        threshold_max = int(params.get("threshold_max", group_default(gen_defaults, "threshold_max", fallback_max)))
    threshold_min = max(int(threshold_min), int(value_min) + 1)
    threshold_max = min(int(threshold_max), int(value_max) - 1)
    if int(threshold_min) > int(threshold_max):
        raise ValueError("threshold bounds leave no feasible support")
    return int(threshold_min), int(threshold_max)

def _crossing_comparison_for_variant(crossing_variant: str) -> str:
    """Return the strict comparison semantics for one threshold-crossing variant."""

    if str(crossing_variant) in {"first_crosses_above_threshold", "linear_projection_crosses_above"}:
        return "greater_than"
    if str(crossing_variant) in {"first_crosses_below_threshold", "linear_projection_crosses_below"}:
        return "less_than"
    raise ValueError(f"unsupported threshold-crossing variant: {crossing_variant}")

def _build_direct_threshold_crossing_values(
    *,
    crossing_variant: str,
    mark_count: int,
    target_index: int,
    threshold: int,
    value_min: int,
    value_max: int,
    instance_seed: int,
    task_id: str,
) -> List[int]:
    """Build one sequence where the first threshold crossing occurs at target_index."""

    rng = spawn_rng(
        int(instance_seed),
        f"{task_id}.direct_values:{str(crossing_variant)}:{int(mark_count)}:{int(target_index)}:{int(threshold)}",
    )
    comparison = _crossing_comparison_for_variant(str(crossing_variant))
    if str(comparison) == "greater_than":
        low_pool = [int(value) for value in range(int(value_min), int(threshold) + 1)]
        high_pool = [int(value) for value in range(int(threshold) + 1, int(value_max) + 1)]
    else:
        low_pool = [int(value) for value in range(int(value_min), int(threshold))]
        high_pool = [int(value) for value in range(int(threshold), int(value_max) + 1)]
    if not low_pool or not high_pool:
        raise ValueError("threshold leaves no feasible crossing values")

    values: List[int] = []
    if str(comparison) == "greater_than":
        values.extend(_sample_values_from_pool(rng, count=int(target_index), pool=low_pool))
        values.append(int(high_pool[int(rng.randint(0, len(high_pool) - 1))]))
        values.extend(
            _sample_values_from_pool(
                rng,
                count=int(mark_count) - int(target_index) - 1,
                pool=high_pool,
            )
        )
    else:
        values.extend(_sample_values_from_pool(rng, count=int(target_index), pool=high_pool))
        values.append(int(low_pool[int(rng.randint(0, len(low_pool) - 1))]))
        values.extend(
            _sample_values_from_pool(
                rng,
                count=int(mark_count) - int(target_index) - 1,
                pool=low_pool,
            )
        )
    return [int(value) for value in values]

def _projection_count_bounds(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
) -> Tuple[int, int, int, int]:
    """Resolve observed and projected support for projected threshold crossings."""

    observed_min = int(params.get("observed_count_min", group_default(gen_defaults, "observed_count_min", 5)))
    observed_max = int(params.get("observed_count_max", group_default(gen_defaults, "observed_count_max", 8)))
    projection_min = int(params.get("projection_count_min", group_default(gen_defaults, "projection_count_min", 3)))
    projection_max = int(params.get("projection_count_max", group_default(gen_defaults, "projection_count_max", 5)))
    if int(observed_min) > int(observed_max):
        raise ValueError("observed_count_min must be <= observed_count_max")
    if int(projection_min) > int(projection_max):
        raise ValueError("projection_count_min must be <= projection_count_max")
    if int(observed_min) < 2:
        raise ValueError("projection variants require at least two observed marks")
    if int(projection_min) < 1:
        raise ValueError("projection variants require at least one projected mark")
    return int(observed_min), int(observed_max), int(projection_min), int(projection_max)

def _decouple_sample_cursor_after_query_id(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Return params whose sampling index advances within each selected query id."""

    explicit_index = params.get("_sample_cursor")
    if explicit_index is None:
        return params
    weights = gen_defaults.get("query_id_weights", {})
    if not isinstance(weights, Mapping):
        return params
    variant_count = 0
    for key, value in weights.items():
        if float(value) <= 0.0:
            continue
        try:
            _crossing_comparison_for_variant(str(key))
        except ValueError:
            continue
        variant_count += 1
    if int(variant_count) <= 1:
        return params
    decoupled = dict(params)
    decoupled["_sample_cursor"] = abs(int(explicit_index)) // int(variant_count)
    return decoupled

def _seeded_choice_from_values(
    values: Sequence[int],
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> int:
    """Select one feasible support value using seeded RNG."""

    ordered = [int(value) for value in values]
    if not ordered:
        raise ValueError(f"no feasible values for {namespace}")
    rng = spawn_rng(int(instance_seed), str(namespace))
    return int(uniform_choice(rng, ordered, sort_keys=True))

def _build_projected_threshold_crossing_values(
    *,
    crossing_variant: str,
    observed_count: int,
    projection_count: int,
    target_step: int,
    threshold: int,
    value_min: int,
    value_max: int,
    instance_seed: int,
    task_id: str,
) -> Tuple[List[int], int, int]:
    """Build one linear observed series plus projected values crossing at target_step."""

    rng = spawn_rng(
        int(instance_seed),
        f"{task_id}.projected_values:{str(crossing_variant)}:{int(observed_count)}:{int(projection_count)}:{int(target_step)}:{int(threshold)}",
    )
    comparison = _crossing_comparison_for_variant(str(crossing_variant))
    delta_min = 1
    delta_max = max(2, min(12, int(value_max) - int(value_min)))
    feasible_pairs: List[Tuple[int, int, List[int]]] = []
    for delta in range(int(delta_min), int(delta_max) + 1):
        for overshoot in range(1, int(delta) + 1):
            delta = int(delta)
            overshoot = int(overshoot)
            if str(comparison) == "greater_than":
                last_observed = int(threshold) - (int(target_step) * int(delta)) + int(overshoot)
                projected = [int(last_observed) + (int(step) * int(delta)) for step in range(1, int(projection_count) + 1)]
                observed = [
                    int(last_observed) - (int(observed_count) - 1 - int(index)) * int(delta)
                    for index in range(int(observed_count))
                ]
            else:
                last_observed = int(threshold) + (int(target_step) * int(delta)) - int(overshoot)
                projected = [int(last_observed) - (int(step) * int(delta)) for step in range(1, int(projection_count) + 1)]
                observed = [
                    int(last_observed) + (int(observed_count) - 1 - int(index)) * int(delta)
                    for index in range(int(observed_count))
                ]
            values = [int(value) for value in observed] + [int(value) for value in projected]
            if any(int(value) < int(value_min) or int(value) > int(value_max) for value in values):
                continue
            before_projected = projected[: max(0, int(target_step) - 1)]
            if str(comparison) == "greater_than":
                if any(int(value) > int(threshold) for value in values[: int(observed_count)]):
                    continue
                if any(int(value) > int(threshold) for value in before_projected):
                    continue
                if int(projected[int(target_step) - 1]) <= int(threshold):
                    continue
            else:
                if any(int(value) < int(threshold) for value in values[: int(observed_count)]):
                    continue
                if any(int(value) < int(threshold) for value in before_projected):
                    continue
                if int(projected[int(target_step) - 1]) >= int(threshold):
                    continue
            feasible_pairs.append((int(delta), int(overshoot), [int(value) for value in values]))
    if not feasible_pairs:
        raise RuntimeError(f"unable to construct projected threshold crossing for {crossing_variant}")
    delta, overshoot, values = feasible_pairs[int(rng.randint(0, len(feasible_pairs) - 1))]
    return [int(value) for value in values], int(delta), int(overshoot)

def build_trend_threshold_crossing_dataset_for_variant(
    *,
    crossing_variant: str,
    scene_variant: SceneVariant,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: LabeledChartDefaults,
    task_id: str,
) -> Tuple[List[int], str, List[str], Dict[str, Any]]:
    """Construct one ordered labeled-chart dataset for threshold-crossing queries."""

    if str(scene_variant) in PIE_LIKE_SCENE_VARIANTS or str(scene_variant) in {"radar", "scatter", "horizontal_bar"}:
        raise ValueError(f"unsupported threshold-crossing scene_variant: {scene_variant}")

    value_min, value_max = resolve_value_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
        instance_seed=int(instance_seed),
    )
    threshold_min, threshold_max = _resolve_trend_threshold_bounds(
        params,
        gen_defaults=gen_defaults,
        value_min=int(value_min),
        value_max=int(value_max),
    )
    threshold = balanced_choice_from_values(
        [int(value) for value in range(int(threshold_min), int(threshold_max) + 1)],
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.threshold:{str(crossing_variant)}",
    )

    comparison = _crossing_comparison_for_variant(str(crossing_variant))
    is_projection = str(crossing_variant).startswith("linear_projection_")
    if not bool(is_projection):
        mark_count_min, mark_count_max = resolve_mark_count_bounds(
            params,
            gen_defaults=gen_defaults,
            defaults=defaults,
            task_id=task_id,
        )
        feasible_counts = [int(count) for count in range(int(mark_count_min), int(mark_count_max) + 1) if int(count) >= 5]
        mark_count = choose_mark_count(
            feasible_counts,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.mark_count:{str(crossing_variant)}",
        )
        index_min = int(params.get("crossing_index_min", group_default(gen_defaults, "crossing_index_min", 2)))
        index_max = int(params.get("crossing_index_max", group_default(gen_defaults, "crossing_index_max", int(mark_count) - 2)))
        index_min = max(1, int(index_min))
        index_max = min(int(index_max), int(mark_count) - 2)
        if int(index_min) > int(index_max):
            raise ValueError("no feasible crossing index support")
        crossing_index = balanced_choice_from_values(
            [int(value) for value in range(int(index_min), int(index_max) + 1)],
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.crossing_index:{str(crossing_variant)}:{int(mark_count)}",
        )
        labels = list(
            sample_chart_labels(
                count=int(mark_count),
                instance_seed=int(instance_seed),
                namespace=f"{task_id}.labels:{str(crossing_variant)}:{int(mark_count)}",
            )
        )
        values = _build_direct_threshold_crossing_values(
            crossing_variant=str(crossing_variant),
            mark_count=int(mark_count),
            target_index=int(crossing_index),
            threshold=int(threshold),
            value_min=int(value_min),
            value_max=int(value_max),
            instance_seed=int(instance_seed),
            task_id=str(task_id),
        )
        answer_label = str(labels[int(crossing_index)])
        ordered_annotation_labels = [str(label) for label in labels[: int(crossing_index) + 1]]
        trace_extras = {
            "value_min": int(value_min),
            "value_max": int(value_max),
            "threshold": int(threshold),
            "threshold_range": [int(threshold_min), int(threshold_max)],
            "comparison": str(comparison),
            "mark_count_range": [int(mark_count_min), int(mark_count_max)],
            "mark_count": int(mark_count),
            "labels": [str(label) for label in labels],
            "values_by_label": {str(label): int(value) for label, value in zip(labels, values)},
            "answer_label": str(answer_label),
            "answer_index": int(crossing_index),
            "crossing_index": int(crossing_index),
            "crossing_label": str(answer_label),
            "pre_crossing_label": str(labels[int(crossing_index) - 1]) if int(crossing_index) > 0 else "",
            "ordered_annotation_labels": [str(label) for label in ordered_annotation_labels],
            "observed_labels": [str(label) for label in labels],
            "projected_labels": [],
            "point_kind_by_label": {str(label): "observed" for label in labels},
            "projection_delta": None,
            "projection_steps_to_cross": None,
        }
        return [int(value) for value in values], str(answer_label), sorted_labels(ordered_annotation_labels), trace_extras

    observed_min, observed_max, projection_min, projection_max = _projection_count_bounds(
        params,
        gen_defaults=gen_defaults,
    )
    support_params = _decouple_sample_cursor_after_query_id(params, gen_defaults=gen_defaults)
    observed_count = _seeded_choice_from_values(
        [int(value) for value in range(int(observed_min), int(observed_max) + 1)],
        params=support_params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.observed_count:{str(crossing_variant)}",
    )
    projection_count = balanced_choice_from_values(
        [int(value) for value in range(int(projection_min), int(projection_max) + 1)],
        params=support_params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.projection_count:{str(crossing_variant)}",
    )
    step_min = int(params.get("projection_step_min", group_default(gen_defaults, "projection_step_min", 1)))
    step_max = int(params.get("projection_step_max", group_default(gen_defaults, "projection_step_max", int(projection_count))))
    step_min = max(1, int(step_min))
    step_max = min(int(step_max), int(projection_count))
    if int(step_min) > int(step_max):
        raise ValueError("no feasible projection step support")
    target_step = balanced_choice_from_values(
        [int(value) for value in range(int(step_min), int(step_max) + 1)],
        params=support_params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.projection_step:{str(crossing_variant)}:{int(projection_count)}",
    )
    total_count = int(observed_count) + int(projection_count)
    labels = list(
        sample_chart_labels(
            count=int(total_count),
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.labels:projected:{str(crossing_variant)}:{int(total_count)}",
        )
    )
    values, projection_delta, overshoot = _build_projected_threshold_crossing_values(
        crossing_variant=str(crossing_variant),
        observed_count=int(observed_count),
        projection_count=int(projection_count),
        target_step=int(target_step),
        threshold=int(threshold),
        value_min=int(value_min),
        value_max=int(value_max),
        instance_seed=int(instance_seed),
        task_id=str(task_id),
    )
    answer_index = int(observed_count) + int(target_step) - 1
    answer_label = str(labels[int(answer_index)])
    observed_labels = [str(label) for label in labels[: int(observed_count)]]
    projected_labels = [str(label) for label in labels[int(observed_count):]]
    ordered_annotation_labels = [str(labels[int(observed_count) - 2]), str(labels[int(observed_count) - 1])] + [
        str(label) for label in labels[int(observed_count): int(answer_index) + 1]
    ]
    trace_extras = {
        "value_min": int(value_min),
        "value_max": int(value_max),
        "threshold": int(threshold),
        "threshold_range": [int(threshold_min), int(threshold_max)],
        "comparison": str(comparison),
        "mark_count_range": [int(observed_min) + int(projection_min), int(observed_max) + int(projection_max)],
        "mark_count": int(total_count),
        "observed_count": int(observed_count),
        "observed_count_range": [int(observed_min), int(observed_max)],
        "projection_count": int(projection_count),
        "projection_count_range": [int(projection_min), int(projection_max)],
        "projection_step_range": [int(step_min), int(step_max)],
        "labels": [str(label) for label in labels],
        "values_by_label": {str(label): int(value) for label, value in zip(labels, values)},
        "answer_label": str(answer_label),
        "answer_index": int(answer_index),
        "crossing_index": int(answer_index),
        "crossing_label": str(answer_label),
        "pre_crossing_label": str(labels[int(answer_index) - 1]) if int(answer_index) > 0 else "",
        "ordered_annotation_labels": [str(label) for label in ordered_annotation_labels],
        "observed_labels": [str(label) for label in observed_labels],
        "projected_labels": [str(label) for label in projected_labels],
        "point_kind_by_label": {
            **{str(label): "observed" for label in observed_labels},
            **{str(label): "projected" for label in projected_labels},
        },
        "projection_delta": int(projection_delta),
        "projection_overshoot": int(overshoot),
        "projection_steps_to_cross": int(target_step),
    }
    return [int(value) for value in values], str(answer_label), sorted_labels(ordered_annotation_labels), trace_extras

def _signed_change_support(
    *,
    value_min: int,
    value_max: int,
    min_abs_change: int,
    max_abs_change: int,
) -> List[int]:
    """Return feasible nonzero signed endpoint changes."""

    max_possible = int(value_max) - int(value_min)
    lower = max(1, int(min_abs_change))
    upper = min(int(max_abs_change), int(max_possible))
    if int(lower) > int(upper):
        return []
    return [int(value) for value in range(-int(upper), -int(lower) + 1)] + [
        int(value) for value in range(int(lower), int(upper) + 1)
    ]

def _start_values_for_delta(
    *,
    delta: int,
    value_min: int,
    value_max: int,
) -> List[int]:
    """Return feasible starting values for one signed endpoint delta."""

    return [
        int(value)
        for value in range(int(value_min), int(value_max) + 1)
        if int(value_min) <= int(value) + int(delta) <= int(value_max)
    ]

def _percent_change_support(
    *,
    value_min: int,
    value_max: int,
    percent_min: int,
    percent_max: int,
    percent_step: int,
) -> Dict[int, List[Tuple[int, int]]]:
    """Return feasible integer-percent endpoint transitions."""

    step = max(1, int(percent_step))
    support: Dict[int, List[Tuple[int, int]]] = {}
    for percent in range(int(percent_min), int(percent_max) + 1, int(step)):
        if int(percent) == 0 or int(percent) <= -100:
            continue
        multiplier = int(100) + int(percent)
        transitions: List[Tuple[int, int]] = []
        for start_value in range(int(value_min), int(value_max) + 1):
            numerator = int(start_value) * int(multiplier)
            if int(numerator) % 100 != 0:
                continue
            end_value = int(numerator // 100)
            if int(value_min) <= int(end_value) <= int(value_max) and int(end_value) != int(start_value):
                transitions.append((int(start_value), int(end_value)))
        if transitions:
            support[int(percent)] = list(transitions)
    return support

def build_trend_interval_change_dataset_for_variant(
    *,
    interval_variant: str,
    scene_variant: SceneVariant,
    params: Mapping[str, Any],
    instance_seed: int,
    gen_defaults: Mapping[str, Any],
    defaults: LabeledChartDefaults,
    task_id: str,
) -> Tuple[List[int], int, List[str], Dict[str, Any]]:
    """Construct one ordered labeled-chart dataset for interval-change value queries."""

    if str(scene_variant) in PIE_LIKE_SCENE_VARIANTS or str(scene_variant) in {"radar", "scatter"}:
        raise ValueError(f"unsupported interval-change scene_variant: {scene_variant}")

    value_min, value_max = resolve_value_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
        instance_seed=int(instance_seed),
    )
    mark_count_min, mark_count_max = resolve_mark_count_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=defaults,
        task_id=task_id,
    )
    feasible_counts = [int(count) for count in range(int(mark_count_min), int(mark_count_max) + 1) if int(count) >= 4]
    mark_count = choose_mark_count(
        feasible_counts,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.mark_count:{str(interval_variant)}",
    )

    gap_min = int(params.get("interval_gap_min", group_default(gen_defaults, "interval_gap_min", 2)))
    gap_max = int(params.get("interval_gap_max", group_default(gen_defaults, "interval_gap_max", 6)))
    gap_min = max(1, int(gap_min))
    gap_max = min(int(gap_max), int(mark_count) - 1)
    if int(gap_min) > int(gap_max):
        raise ValueError("no feasible interval gap support")
    gap = balanced_choice_from_values(
        [int(value) for value in range(int(gap_min), int(gap_max) + 1)],
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.interval_gap:{str(interval_variant)}:{int(mark_count)}",
    )
    start_index = balanced_choice_from_values(
        [int(value) for value in range(0, int(mark_count) - int(gap))],
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.start_index:{str(interval_variant)}:{int(mark_count)}:{int(gap)}",
    )
    end_index = int(start_index) + int(gap)

    min_abs_change = int(params.get("change_abs_min", group_default(gen_defaults, "change_abs_min", 5)))
    max_abs_change = int(params.get("change_abs_max", group_default(gen_defaults, "change_abs_max", 60)))
    if str(interval_variant) == "absolute_change_between_labels":
        signed_support = _signed_change_support(
            value_min=int(value_min),
            value_max=int(value_max),
            min_abs_change=int(min_abs_change),
            max_abs_change=int(max_abs_change),
        )
        abs_support = sorted({abs(int(value)) for value in signed_support})
        target_abs_change = balanced_choice_from_values(
            abs_support,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.absolute_change",
        )
        direction = balanced_choice_from_values(
            [-1, 1],
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.absolute_change_direction:{int(target_abs_change)}",
        )
        delta = int(direction) * int(target_abs_change)
        start_candidates = _start_values_for_delta(delta=int(delta), value_min=int(value_min), value_max=int(value_max))
        answer_value = int(target_abs_change)
    elif str(interval_variant) == "signed_change_between_labels":
        signed_support = _signed_change_support(
            value_min=int(value_min),
            value_max=int(value_max),
            min_abs_change=int(min_abs_change),
            max_abs_change=int(max_abs_change),
        )
        delta = balanced_choice_from_values(
            signed_support,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.signed_change",
        )
        start_candidates = _start_values_for_delta(delta=int(delta), value_min=int(value_min), value_max=int(value_max))
        answer_value = int(delta)
    elif str(interval_variant) == "percent_change_between_labels":
        percent_support = _percent_change_support(
            value_min=int(value_min),
            value_max=int(value_max),
            percent_min=int(params.get("percent_change_min", group_default(gen_defaults, "percent_change_min", -75))),
            percent_max=int(params.get("percent_change_max", group_default(gen_defaults, "percent_change_max", 150))),
            percent_step=int(params.get("percent_change_step", group_default(gen_defaults, "percent_change_step", 5))),
        )
        percent = balanced_choice_from_values(
            sorted(percent_support.keys()),
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.percent_change",
        )
        transition_rng = spawn_rng(
            int(instance_seed),
            f"{task_id}.percent_transition:{int(percent)}",
        )
        start_value, end_value = transition_rng.choice(percent_support[int(percent)])
        delta = int(end_value) - int(start_value)
        start_candidates = [int(start_value)]
        answer_value = int(percent)
    elif str(interval_variant) == "average_rate_over_interval":
        rate_candidates = _signed_change_support(
            value_min=int(value_min),
            value_max=int(value_max),
            min_abs_change=int(params.get("average_rate_abs_min", group_default(gen_defaults, "average_rate_abs_min", 2))),
            max_abs_change=int(params.get("average_rate_abs_max", group_default(gen_defaults, "average_rate_abs_max", 12))),
        )
        feasible_gaps_by_rate = {
            int(rate): [
                int(candidate_gap)
                for candidate_gap in range(int(gap_min), int(gap_max) + 1)
                if _start_values_for_delta(
                    delta=int(rate) * int(candidate_gap),
                    value_min=int(value_min),
                    value_max=int(value_max),
                )
            ]
            for rate in rate_candidates
        }
        rate_support = [
            int(rate)
            for rate, feasible_gaps in feasible_gaps_by_rate.items()
            if feasible_gaps
        ]
        rate = balanced_choice_from_values(
            rate_support,
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.average_rate",
        )
        gap = balanced_choice_from_values(
            feasible_gaps_by_rate[int(rate)],
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.average_rate_gap:{int(rate)}:{int(mark_count)}",
        )
        start_index = balanced_choice_from_values(
            [int(value) for value in range(0, int(mark_count) - int(gap))],
            params=params,
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.start_index:{str(interval_variant)}:{int(mark_count)}:{int(gap)}",
        )
        end_index = int(start_index) + int(gap)
        delta = int(rate) * int(gap)
        start_candidates = _start_values_for_delta(delta=int(delta), value_min=int(value_min), value_max=int(value_max))
        answer_value = int(rate)
    else:
        raise ValueError(f"unsupported interval-change variant: {interval_variant}")

    if not start_candidates:
        raise ValueError("no feasible start values for interval-change query")
    start_value = balanced_choice_from_values(
        start_candidates,
        params=params,
        instance_seed=int(instance_seed),
        namespace=f"{task_id}.start_value:{str(interval_variant)}:{int(delta)}",
    )
    end_value = int(start_value) + int(delta)

    rng = spawn_rng(
        int(instance_seed),
        f"{task_id}.values:{str(interval_variant)}:{int(mark_count)}:{int(start_index)}:{int(end_index)}:{int(start_value)}:{int(end_value)}",
    )
    values = [int(rng.randint(int(value_min), int(value_max))) for _ in range(int(mark_count))]
    values[int(start_index)] = int(start_value)
    values[int(end_index)] = int(end_value)
    labels = list(
        sample_chart_labels(
            count=int(mark_count),
            instance_seed=int(instance_seed),
            namespace=f"{task_id}.labels:{str(interval_variant)}:{int(mark_count)}",
        )
    )
    ordered_annotation_labels = [str(labels[index]) for index in range(int(start_index), int(end_index) + 1)]
    annotation_labels = list(ordered_annotation_labels)
    trace_extras: Dict[str, Any] = {
        "value_min": int(value_min),
        "value_max": int(value_max),
        "mark_count_range": [int(mark_count_min), int(mark_count_max)],
        "mark_count": int(mark_count),
        "interval_gap": int(gap),
        "interval_gap_range": [int(gap_min), int(gap_max)],
        "start_index": int(start_index),
        "end_index": int(end_index),
        "start_label": str(labels[int(start_index)]),
        "end_label": str(labels[int(end_index)]),
        "start_value": int(start_value),
        "end_value": int(end_value),
        "delta": int(delta),
        "answer_value": int(answer_value),
        "labels": [str(label) for label in labels],
        "values_by_label": {str(label): int(value) for label, value in zip(labels, values)},
        "annotation_labels": list(annotation_labels),
        "ordered_annotation_labels": [str(label) for label in ordered_annotation_labels],
        "annotation_point_indices": [int(index) for index in range(int(start_index), int(end_index) + 1)],
        "change_abs_range": [int(min_abs_change), int(max_abs_change)],
        "percent_change_range": [
            int(params.get("percent_change_min", group_default(gen_defaults, "percent_change_min", -75))),
            int(params.get("percent_change_max", group_default(gen_defaults, "percent_change_max", 150))),
        ],
        "percent_change_step": int(params.get("percent_change_step", group_default(gen_defaults, "percent_change_step", 5))),
        "average_rate_abs_range": [
            int(params.get("average_rate_abs_min", group_default(gen_defaults, "average_rate_abs_min", 2))),
            int(params.get("average_rate_abs_max", group_default(gen_defaults, "average_rate_abs_max", 12))),
        ],
    }
    return [int(value) for value in values], int(answer_value), annotation_labels, trace_extras


__all__ = [
    "build_trend_interval_change_dataset_for_variant",
    "build_trend_structure_dataset_for_variant",
    "build_trend_threshold_crossing_dataset_for_variant",
]
