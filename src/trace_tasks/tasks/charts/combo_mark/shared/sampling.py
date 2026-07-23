"""Neutral sampling primitives for combo-mark chart tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import normalize_positive_weights, uniform_choice, weighted_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.combo_mark.shared.defaults import (
    GENERATION_DEFAULTS,
    SCENE_NAMESPACE,
    SCENE_VARIANTS,
    int_bounds,
    scene_default,
)
from trace_tasks.tasks.charts.combo_mark.shared.state import ComboDataset
from trace_tasks.tasks.charts.shared.label_assets import (
    resolve_chart_category_labels,
    resolve_chart_compact_axis_labels,
)


def choose_scene_variant(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    allowed_variants: Sequence[str] | None = None,
    sampling_divisor: int = 1,
) -> tuple[str, dict[str, float], dict[str, Any]]:
    """Choose one combo visual variant without knowing public task identity."""

    support = tuple(str(value) for value in (allowed_variants or SCENE_VARIANTS))
    requested = params.get("scene_variant")
    if requested is not None:
        selected = str(requested)
        if selected not in set(support):
            raise ValueError(f"unsupported scene_variant: {selected}; supported: {support}")
        stripped = dict(params)
        stripped.pop("scene_variant", None)
        return selected, {item: (1.0 if item == selected else 0.0) for item in support}, stripped

    raw_weights = params.get(
        "scene_variant_weights",
        scene_default(GENERATION_DEFAULTS, "scene_variant_weights", None),
    )
    if isinstance(raw_weights, Mapping):
        probabilities = normalize_positive_weights(
            {str(key): float(value) for key, value in raw_weights.items() if str(key) in set(support)},
            default_keys=support,
        )
    else:
        probabilities = normalize_positive_weights({}, default_keys=support)
    positives = [key for key in support if float(probabilities.get(str(key), 0.0)) > 0.0]
    if not positives:
        raise ValueError("scene_variant_weights must leave at least one positive variant")
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.scene_variant")
    return str(weighted_choice(rng, probabilities, sort_keys=True)), dict(probabilities), dict(params)


def sample_labels(rng: Any, count: int) -> tuple[str, ...]:
    labels = resolve_chart_compact_axis_labels(
        rng,
        count=int(count),
        min_chars=2,
        max_chars=3,
    ).labels
    return tuple(str(label) for label in labels)


def sample_values(rng: Any, *, count: int, low: int, high: int) -> tuple[int, ...]:
    return tuple(int(rng.randint(int(low), int(high))) for _ in range(int(count)))


def choose_metric_pair(instance_seed: int) -> tuple[str, str]:
    labels = resolve_chart_category_labels(
        spawn_rng(int(instance_seed), "charts.combo.metric_pair"),
        count=2,
        min_chars=2,
        max_chars=8,
        allow_spaces=False,
    ).labels
    return str(labels[0]), str(labels[1])


def sample_base_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    allowed_scene_variants: Sequence[str] | None = None,
    label_count_bounds: tuple[int, int] | None = None,
    scene_sampling_divisor: int = 1,
) -> tuple[ComboDataset, dict[str, Any]]:
    """Sample scene data and labels without binding an answer."""

    scene_variant, scene_probabilities, scene_params = choose_scene_variant(
        params=params,
        instance_seed=int(instance_seed),
        allowed_variants=allowed_scene_variants,
        sampling_divisor=int(scene_sampling_divisor),
    )
    fallback_bounds = label_count_bounds or (7, 11)
    label_min, label_max = int_bounds(
        scene_params,
        GENERATION_DEFAULTS,
        low_key="label_count_min",
        high_key="label_count_max",
        fallback=fallback_bounds,
    )
    value_min, value_max = int_bounds(
        scene_params,
        GENERATION_DEFAULTS,
        low_key="value_min",
        high_key="value_max",
        fallback=(12, 88),
    )
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.values")
    label_count = int(rng.randint(int(label_min), int(label_max)))
    labels = sample_labels(rng, label_count)
    primary_values = sample_values(rng, count=label_count, low=value_min, high=value_max)
    line_values = sample_values(rng, count=label_count, low=value_min, high=value_max)
    primary_name, line_name = choose_metric_pair(int(instance_seed))
    dataset = ComboDataset(
        labels=tuple(labels),
        primary_values=tuple(primary_values),
        line_values=tuple(line_values),
        primary_name=str(primary_name),
        line_name=str(line_name),
        scene_variant=str(scene_variant),
        label_count_range=(int(label_min), int(label_max)),
        scene_variant_probabilities=dict(scene_probabilities),
    )
    trace = {
        "label_count": int(label_count),
        "label_count_range": [int(label_min), int(label_max)],
        "value_range": [int(value_min), int(value_max)],
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": dict(scene_probabilities),
    }
    return dataset, trace


def dataset_with_values(
    dataset: ComboDataset,
    *,
    primary_values: Sequence[int] | None = None,
    line_values: Sequence[int] | None = None,
) -> ComboDataset:
    return ComboDataset(
        labels=tuple(dataset.labels),
        primary_values=tuple(int(value) for value in (primary_values if primary_values is not None else dataset.primary_values)),
        line_values=tuple(int(value) for value in (line_values if line_values is not None else dataset.line_values)),
        primary_name=str(dataset.primary_name),
        line_name=str(dataset.line_name),
        scene_variant=str(dataset.scene_variant),
        label_count_range=tuple(dataset.label_count_range),
        scene_variant_probabilities=dict(dataset.scene_variant_probabilities),
    )


def unique_extremum(labels: Sequence[str], values: Sequence[int], *, mode: str) -> tuple[str, int]:
    if not labels:
        raise ValueError("empty candidate set")
    target = max(values) if str(mode) == "max" else min(values)
    winners = [str(label) for label, value in zip(labels, values) if int(value) == int(target)]
    if len(winners) != 1:
        raise ValueError("extremum tie")
    return str(winners[0]), int(target)


def select_threshold(values: Sequence[int], *, rng: Any, above: bool, min_candidates: int = 2) -> int:
    sorted_values = sorted(set(int(value) for value in values))
    if len(sorted_values) < 4:
        raise ValueError("not enough distinct threshold values")
    candidates = []
    for low, high in zip(sorted_values[:-1], sorted_values[1:]):
        threshold = int((int(low) + int(high)) // 2)
        count = sum(1 for value in values if (int(value) > threshold if above else int(value) < threshold))
        if int(min_candidates) <= count <= len(values) - 1:
            candidates.append(threshold)
    if not candidates:
        raise ValueError("no usable threshold")
    return int(candidates[int(rng.randrange(0, len(candidates)))])


def threshold_candidates_for(values: Sequence[int], *, above: bool) -> tuple[int, ...]:
    sorted_values = sorted(set(int(value) for value in values))
    if len(sorted_values) < 2:
        return ()
    thresholds = []
    for low, high in zip(sorted_values[:-1], sorted_values[1:]):
        thresholds.append(int(low) if bool(above) else int(high))
    return tuple(dict.fromkeys(thresholds))


def rank_gap_index(
    *,
    primary_values: Sequence[int],
    line_values: Sequence[int],
    gap_mode: str,
) -> tuple[int, int, list[int]]:
    """Rank category indices by a semantic primary-vs-line gap metric."""

    candidates: list[tuple[int, int]] = []
    for idx, (primary_value, line_value) in enumerate(zip(primary_values, line_values)):
        signed_gap = int(primary_value) - int(line_value)
        if str(gap_mode) == "largest_absolute":
            candidates.append((idx, abs(int(signed_gap))))
        elif str(gap_mode) == "smallest_nonzero_absolute":
            if int(signed_gap) != 0:
                candidates.append((idx, abs(int(signed_gap))))
        elif str(gap_mode) == "largest_primary_over_line":
            if int(signed_gap) > 0:
                candidates.append((idx, int(signed_gap)))
        elif str(gap_mode) == "largest_line_over_primary" and int(signed_gap) < 0:
            candidates.append((idx, -int(signed_gap)))
    if not candidates:
        raise ValueError("no gap-extremum candidates")
    target_value = (
        min(value for _, value in candidates)
        if str(gap_mode) == "smallest_nonzero_absolute"
        else max(value for _, value in candidates)
    )
    winners = [idx for idx, value in candidates if int(value) == int(target_value)]
    if len(winners) != 1:
        raise ValueError("gap extremum tie")
    return int(winners[0]), int(target_value), [int(idx) for idx, _ in candidates]


def conditioned_extremum_index(
    *,
    labels: Sequence[str],
    primary_values: Sequence[int],
    line_values: Sequence[int],
    condition_role: str,
    condition_relation: str,
    target_role: str,
    extremum: str,
    rng: Any,
) -> tuple[int, int, list[int]]:
    """Return the winning category index after a semantic series filter."""

    if str(condition_role) == "primary":
        condition_values = primary_values
    elif str(condition_role) == "line":
        condition_values = line_values
    else:
        raise ValueError(f"unsupported condition role: {condition_role}")
    if str(target_role) == "primary":
        all_target_values = primary_values
    elif str(target_role) == "line":
        all_target_values = line_values
    else:
        raise ValueError(f"unsupported target role: {target_role}")
    above = str(condition_relation) == "above"
    threshold = select_threshold(condition_values, rng=rng, above=above)
    candidate_indices = [
        idx
        for idx, value in enumerate(condition_values)
        if (int(value) > int(threshold) if bool(above) else int(value) < int(threshold))
    ]
    target_values = [int(all_target_values[idx]) for idx in candidate_indices]
    answer, _target = unique_extremum(
        [labels[idx] for idx in candidate_indices],
        target_values,
        mode=str(extremum),
    )
    return int(tuple(str(label) for label in labels).index(answer)), int(threshold), [int(idx) for idx in candidate_indices]


def balanced_count_from_bounds(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    low_key: str,
    high_key: str,
    fallback: tuple[int, int],
    high_cap: int,
    sampling_divisor: int,
    namespace: str,
) -> tuple[int, dict[int, float], tuple[int, int]]:
    """Sample one integer from a bounded support without task/query dispatch."""

    low = int(params.get(low_key, scene_default(GENERATION_DEFAULTS, low_key, int(fallback[0]))))
    high = int(params.get(high_key, scene_default(GENERATION_DEFAULTS, high_key, int(fallback[1]))))
    high = min(int(high), int(high_cap))
    if int(high) < int(low):
        raise ValueError("balanced count support is infeasible")
    support = tuple(range(int(low), int(high) + 1))
    probabilities = {int(value): 1.0 / float(len(support)) for value in support}
    selected = int(
        uniform_choice(
            spawn_rng(int(instance_seed), str(namespace)),
            support,
            sort_keys=True,
        )
    )
    return int(selected), probabilities, (int(low), int(high))


__all__ = [
    "balanced_count_from_bounds",
    "choose_metric_pair",
    "choose_scene_variant",
    "conditioned_extremum_index",
    "dataset_with_values",
    "sample_base_dataset",
    "sample_labels",
    "sample_values",
    "rank_gap_index",
    "select_threshold",
    "threshold_candidates_for",
    "unique_extremum",
]
