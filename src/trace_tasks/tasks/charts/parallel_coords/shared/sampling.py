"""Dataset sampling primitives for the parallel-coordinates chart scene."""

from __future__ import annotations

from itertools import combinations
from typing import Any, Mapping, Sequence

from .....core.sampling import support_probability_map, uniform_choice
from .....core.seed import spawn_rng
from ....shared.config_defaults import resolve_required_int_bounds
from ...shared.label_assets import resolve_chart_entity_labels
from ...shared.labeled_chart_variants import resolve_chart_axis_variant_for_namespace
from .defaults import (
    GENERATION_DEFAULTS,
    PROFILE_PALETTE,
    SCENE_NAMESPACE,
    SUPPORTED_SCENE_VARIANTS,
    generation_int,
)
from .state import ParallelDataset, ParallelProfile, ParallelQueryState


def _balanced_int(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    namespace: str,
    low: int,
    high: int,
) -> tuple[int, dict[str, float]]:
    values = tuple(int(value) for value in range(int(low), int(high) + 1))
    if not values:
        raise ValueError(f"empty integer support for {namespace}")
    selected = uniform_choice(
        spawn_rng(int(instance_seed), str(namespace)),
        values,
        sort_keys=True,
    )
    return int(selected), support_probability_map(values)


def _resolve_scene_variant(params: Mapping[str, Any], *, instance_seed: int) -> tuple[str, dict[str, float]]:
    return resolve_chart_axis_variant_for_namespace(
        params=params,
        gen_defaults=GENERATION_DEFAULTS,
        instance_seed=int(instance_seed),
        supported_variants=SUPPORTED_SCENE_VARIANTS,
        namespace=f"{SCENE_NAMESPACE}.scene_variant",
        explicit_key="scene_variant",
        weights_key="scene_variant_weights",
        balance_flag_key="balanced_scene_variant_sampling",
    )


def _choose_axis_pair(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    axis_count: int,
    adjacent_only: bool,
    namespace: str,
) -> tuple[int, int]:
    pairs = (
        [(index, index + 1) for index in range(int(axis_count) - 1)]
        if bool(adjacent_only)
        else [(i, j) for i in range(int(axis_count)) for j in range(i + 1, int(axis_count))]
    )
    if not pairs:
        raise ValueError("axis_count must allow at least one axis pair")
    return tuple(
        uniform_choice(
            spawn_rng(int(instance_seed), str(namespace)),
            tuple(pairs),
            sort_keys=True,
        )
    )  # type: ignore[return-value]


def _inversion_count(values: Sequence[int]) -> int:
    return sum(1 for i, j in combinations(range(len(values)), 2) if int(values[i]) > int(values[j]))


def _make_exact_inversion_permutation(n: int, inversions: int, rng) -> tuple[int, ...]:
    """Return one permutation of 0..n-1 with exactly the requested inversion count."""

    target = int(inversions)
    if target < 0 or target > (int(n) * (int(n) - 1)) // 2:
        raise ValueError("inversion count outside feasible range")
    for _ in range(2000):
        values = list(range(int(n)))
        rng.shuffle(values)
        if _inversion_count(values) == target:
            return tuple(values)

    remaining = int(target)
    code: list[int] = []
    for i in range(int(n)):
        max_here = int(n) - 1 - int(i)
        take = min(max_here, remaining)
        code.append(int(take))
        remaining -= int(take)
    pool = list(range(int(n)))
    out: list[int] = []
    for take in code:
        out.append(pool.pop(int(take)))
    return tuple(out)


def _rank_values(order: Sequence[int], *, value_min: int, value_max: int) -> dict[int, int]:
    n = len(order)
    if n <= 1:
        return {int(order[0]): int(value_min)}
    low = int(value_min) + 1
    high = int(value_max) - 1
    step = max(1, int((high - low) / max(1, n - 1)))
    return {int(profile_index): int(min(high, low + (rank * step))) for rank, profile_index in enumerate(order)}


def _sample_base(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    use_crossing_bounds: bool,
) -> tuple[str, tuple[str, ...], list[str], int, int, int, int, dict[str, Any]]:
    """Sample scene-global axes, profiles, labels, values, and range metadata."""

    scene_variant, scene_probs = _resolve_scene_variant(params, instance_seed=int(instance_seed))
    axis_min, axis_max = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="axis_count_min",
        max_key="axis_count_max",
        fallback_min=4,
        fallback_max=6,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    profile_min, profile_max = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="profile_count_min",
        max_key="profile_count_max",
        fallback_min=5,
        fallback_max=8,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    if bool(use_crossing_bounds):
        axis_min, axis_max = resolve_required_int_bounds(
            params,
            GENERATION_DEFAULTS,
            min_key="crossing_axis_count_min",
            max_key="crossing_axis_count_max",
            fallback_min=int(axis_min),
            fallback_max=int(axis_max),
            context=f"crossing generation defaults for {SCENE_NAMESPACE}",
        )
        profile_min, profile_max = resolve_required_int_bounds(
            params,
            GENERATION_DEFAULTS,
            min_key="crossing_profile_count_min",
            max_key="crossing_profile_count_max",
            fallback_min=int(profile_min),
            fallback_max=int(profile_max),
            context=f"crossing generation defaults for {SCENE_NAMESPACE}",
        )
    value_min, value_max = resolve_required_int_bounds(
        params,
        GENERATION_DEFAULTS,
        min_key="value_min",
        max_key="value_max",
        fallback_min=1,
        fallback_max=20,
        context=f"generation defaults for {SCENE_NAMESPACE}",
    )
    axis_count, axis_count_probs = _balanced_int(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.axis_count",
        low=int(axis_min),
        high=int(axis_max),
    )
    profile_count, profile_count_probs = _balanced_int(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.profile_count",
        low=int(profile_min),
        high=int(profile_max),
    )
    metric_rng = spawn_rng(int(instance_seed), f"{namespace}.metrics")
    metrics = tuple(
        str(label)
        for label in resolve_chart_entity_labels(
            metric_rng,
            count=int(axis_count),
            min_chars=2,
            max_chars=8,
            allow_spaces=False,
        ).labels
    )
    profile_rng = spawn_rng(int(instance_seed), f"{namespace}.profile_labels")
    profile_labels = [
        str(label)
        for label in resolve_chart_entity_labels(
            profile_rng,
            count=int(profile_count),
            min_chars=2,
            max_chars=6,
            allow_spaces=False,
        ).labels
    ]
    trace_params = {
        "scene_variant": str(scene_variant),
        "scene_variant_probabilities": dict(scene_probs),
        "axis_count": int(axis_count),
        "axis_count_probabilities": dict(axis_count_probs),
        "profile_count": int(profile_count),
        "profile_count_probabilities": dict(profile_count_probs),
        "value_min": int(value_min),
        "value_max": int(value_max),
    }
    return (
        str(scene_variant),
        metrics,
        profile_labels,
        int(profile_count),
        int(axis_count),
        int(value_min),
        int(value_max),
        trace_params,
    )


def _finish_dataset(
    *,
    scene_variant: str,
    metrics: Sequence[str],
    profile_labels: Sequence[str],
    values: Sequence[Sequence[int]],
    value_min: int,
    value_max: int,
    answer: int | str,
    answer_type: str,
    axis_i: int,
    axis_j: int,
    threshold: int | None,
    annotation_profile_ids: Sequence[str],
    crossing_pairs: Sequence[tuple[str, str]],
    trace_params: Mapping[str, Any],
) -> ParallelDataset:
    """Normalize sampled semantic state into immutable scene dataclasses."""

    profiles = tuple(
        ParallelProfile(
            profile_id=f"profile_{index}",
            label=str(label),
            values=tuple(int(value) for value in values[int(index)]),
            color_rgb=tuple(int(channel) for channel in PROFILE_PALETTE[int(index)]),
        )
        for index, label in enumerate(profile_labels)
    )
    params = {
        **dict(trace_params),
        "axis_i": int(axis_i),
        "axis_j": int(axis_j),
        "axis_i_label": str(metrics[int(axis_i)]),
        "axis_j_label": str(metrics[int(axis_j)]),
        "answer": int(answer) if str(answer_type) == "integer" else str(answer),
        "answer_type": str(answer_type),
        "annotation_profile_ids": [str(value) for value in annotation_profile_ids],
        "crossing_pairs": [[str(first), str(second)] for first, second in crossing_pairs],
    }
    return ParallelDataset(
        scene_variant=str(scene_variant),
        metrics=tuple(str(value) for value in metrics),
        profiles=profiles,
        value_min=int(value_min),
        value_max=int(value_max),
        query=ParallelQueryState(
            answer=int(answer) if str(answer_type) == "integer" else str(answer),
            answer_type=str(answer_type),
            axis_i=int(axis_i),
            axis_j=int(axis_j),
            threshold=threshold,
            annotation_profile_ids=tuple(str(value) for value in annotation_profile_ids),
            crossing_pairs=tuple((str(first), str(second)) for first, second in crossing_pairs),
            params=dict(params),
        ),
    )


def sample_axis_condition_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    comparator_pair: tuple[str, str],
    namespace: str,
) -> ParallelDataset:
    """Sample profiles with an exact count satisfying two axis threshold predicates."""

    (
        scene_variant,
        metrics,
        profile_labels,
        profile_count,
        axis_count,
        value_min,
        value_max,
        trace_params,
    ) = _sample_base(
        params=params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        use_crossing_bounds=False,
    )
    threshold_min = generation_int(params, "condition_threshold_min", 8)
    threshold_max = generation_int(params, "condition_threshold_max", 14)
    threshold, threshold_probs = _balanced_int(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.threshold",
        low=int(threshold_min),
        high=int(threshold_max),
    )
    if int(threshold) <= int(value_min) or int(threshold) >= int(value_max):
        raise ValueError("axis condition threshold must lie inside the sampled value range")
    count_min = generation_int(params, "condition_answer_count_min", 1)
    count_max = min(generation_int(params, "condition_answer_count_max", 5), int(profile_count) - 1)
    target_count, target_count_probs = _balanced_int(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.answer",
        low=int(count_min),
        high=max(int(count_min), int(count_max)),
    )
    rng = spawn_rng(int(instance_seed), f"{namespace}.values")
    values: list[list[int]] = [
        [int(rng.randint(int(value_min), int(value_max))) for _ in range(int(axis_count))]
        for _ in range(int(profile_count))
    ]
    axis_i, axis_j = _choose_axis_pair(
        params,
        instance_seed=int(instance_seed),
        axis_count=int(axis_count),
        adjacent_only=True,
        namespace=f"{namespace}.axis_pair",
    )
    annotation_indices = set(rng.sample(list(range(int(profile_count))), int(target_count)))
    first_comparator, second_comparator = (str(comparator_pair[0]), str(comparator_pair[1]))
    for profile_index in range(int(profile_count)):
        is_target = int(profile_index) in annotation_indices
        if is_target:
            values[profile_index][axis_i] = (
                int(rng.randint(int(threshold) + 1, int(value_max)))
                if first_comparator == "above"
                else int(rng.randint(int(value_min), int(threshold) - 1))
            )
            values[profile_index][axis_j] = (
                int(rng.randint(int(threshold) + 1, int(value_max)))
                if second_comparator == "above"
                else int(rng.randint(int(value_min), int(threshold) - 1))
            )
            continue
        fail_first_axis = rng.random() < 0.5
        values[profile_index][axis_i] = (
            int(rng.randint(int(value_min), int(threshold)))
            if first_comparator == "above"
            else int(rng.randint(int(threshold), int(value_max)))
        ) if fail_first_axis else (
            int(rng.randint(int(threshold) + 1, int(value_max)))
            if first_comparator == "above"
            else int(rng.randint(int(value_min), int(threshold) - 1))
        )
        values[profile_index][axis_j] = (
            int(rng.randint(int(value_min), int(threshold)))
            if second_comparator == "above"
            else int(rng.randint(int(threshold), int(value_max)))
        ) if not fail_first_axis else (
            int(rng.randint(int(threshold) + 1, int(value_max)))
            if second_comparator == "above"
            else int(rng.randint(int(value_min), int(threshold) - 1))
        )
    annotation_profile_ids = tuple(f"profile_{index}" for index in sorted(annotation_indices))
    trace_params.update(
        {
            "threshold": int(threshold),
            "threshold_probabilities": dict(threshold_probs),
            "target_count": int(target_count),
            "target_count_probabilities": dict(target_count_probs),
            "axis_predicates": [first_comparator, second_comparator],
        }
    )
    return _finish_dataset(
        scene_variant=scene_variant,
        metrics=metrics,
        profile_labels=profile_labels,
        values=values,
        value_min=value_min,
        value_max=value_max,
        answer=int(target_count),
        answer_type="integer",
        axis_i=int(axis_i),
        axis_j=int(axis_j),
        threshold=int(threshold),
        annotation_profile_ids=annotation_profile_ids,
        crossing_pairs=(),
        trace_params=trace_params,
    )


def sample_axis_delta_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    change_mode: str,
    namespace: str,
) -> ParallelDataset:
    """Sample one unique profile with the extreme axis-to-axis change."""

    (
        scene_variant,
        metrics,
        profile_labels,
        profile_count,
        axis_count,
        value_min,
        value_max,
        trace_params,
    ) = _sample_base(
        params=params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        use_crossing_bounds=False,
    )
    if int(value_max) - int(value_min) < int(profile_count):
        raise ValueError("delta task requires enough value range for distinct profile deltas")
    rng = spawn_rng(int(instance_seed), f"{namespace}.values")
    values: list[list[int]] = [
        [int(rng.randint(int(value_min), int(value_max))) for _ in range(int(axis_count))]
        for _ in range(int(profile_count))
    ]
    axis_i, axis_j = _choose_axis_pair(
        params,
        instance_seed=int(instance_seed),
        axis_count=int(axis_count),
        adjacent_only=True,
        namespace=f"{namespace}.axis_pair",
    )
    target_index = int(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{namespace}.target_profile"),
            tuple(range(int(profile_count))),
            sort_keys=True,
        )
    )
    deltas = list(range(1, min(12, int(value_max) - int(value_min)) + 1))
    rng.shuffle(deltas)
    if len(deltas) < int(profile_count):
        raise ValueError("not enough distinct deltas for profile extrema")
    target_delta = int(max(deltas))
    other_deltas = [int(value) for value in deltas if int(value) != int(target_delta)]
    mode = str(change_mode)
    if mode not in {"increase", "decrease", "absolute"}:
        raise ValueError(f"unsupported change mode: {mode}")
    for profile_index in range(int(profile_count)):
        delta = int(target_delta if int(profile_index) == int(target_index) else other_deltas.pop())
        base = int(rng.randint(int(value_min), int(value_max) - int(delta)))
        if mode == "increase":
            values[profile_index][axis_i] = int(base)
            values[profile_index][axis_j] = int(base + delta)
        elif mode == "decrease":
            values[profile_index][axis_i] = int(base + delta)
            values[profile_index][axis_j] = int(base)
        elif int(profile_index) == int(target_index) or rng.random() < 0.5:
            values[profile_index][axis_i] = int(base)
            values[profile_index][axis_j] = int(base + delta)
        else:
            values[profile_index][axis_i] = int(base + delta)
            values[profile_index][axis_j] = int(base)
    trace_params.update(
        {
            "target_profile_index": int(target_index),
            "target_delta": int(target_delta),
            "change_mode": str(mode),
        }
    )
    return _finish_dataset(
        scene_variant=scene_variant,
        metrics=metrics,
        profile_labels=profile_labels,
        values=values,
        value_min=value_min,
        value_max=value_max,
        answer=str(profile_labels[int(target_index)]),
        answer_type="string",
        axis_i=int(axis_i),
        axis_j=int(axis_j),
        threshold=None,
        annotation_profile_ids=(f"profile_{target_index}",),
        crossing_pairs=(),
        trace_params=trace_params,
    )


def sample_all_crossings_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> ParallelDataset:
    """Sample one adjacent-axis interval with an exact crossing count."""

    (
        scene_variant,
        metrics,
        profile_labels,
        profile_count,
        axis_count,
        value_min,
        value_max,
        trace_params,
    ) = _sample_base(
        params=params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        use_crossing_bounds=True,
    )
    rng = spawn_rng(int(instance_seed), f"{namespace}.values")
    axis_i, axis_j = _choose_axis_pair(
        params,
        instance_seed=int(instance_seed),
        axis_count=int(axis_count),
        adjacent_only=True,
        namespace=f"{namespace}.axis_pair",
    )
    max_crossings = min(
        generation_int(params, "crossing_answer_count_max", 10),
        (int(profile_count) * (int(profile_count) - 1)) // 2,
    )
    min_crossings = min(generation_int(params, "crossing_answer_count_min", 1), int(max_crossings))
    target_count, target_count_probs = _balanced_int(
        params,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.answer",
        low=int(min_crossings),
        high=int(max_crossings),
    )
    values: list[list[int]] = [
        [int(rng.randint(int(value_min), int(value_max))) for _ in range(int(axis_count))]
        for _ in range(int(profile_count))
    ]
    order_a = tuple(range(int(profile_count)))
    order_b = _make_exact_inversion_permutation(int(profile_count), int(target_count), rng)
    rank_values_a = _rank_values(order_a, value_min=int(value_min), value_max=int(value_max))
    rank_values_b = _rank_values(order_b, value_min=int(value_min), value_max=int(value_max))
    for profile_index in range(int(profile_count)):
        values[profile_index][axis_i] = int(rank_values_a[int(profile_index)])
        values[profile_index][axis_j] = int(rank_values_b[int(profile_index)])
    profile_ids = [f"profile_{index}" for index in range(int(profile_count))]
    crossing_pairs = tuple(
        (profile_ids[a], profile_ids[b])
        for a, b in combinations(range(int(profile_count)), 2)
        if (values[a][axis_i] - values[b][axis_i]) * (values[a][axis_j] - values[b][axis_j]) < 0
    )
    trace_params.update(
        {
            "target_count": int(target_count),
            "target_count_probabilities": dict(target_count_probs),
        }
    )
    return _finish_dataset(
        scene_variant=scene_variant,
        metrics=metrics,
        profile_labels=profile_labels,
        values=values,
        value_min=value_min,
        value_max=value_max,
        answer=int(len(crossing_pairs)),
        answer_type="integer",
        axis_i=int(axis_i),
        axis_j=int(axis_j),
        threshold=None,
        annotation_profile_ids=tuple(profile_ids),
        crossing_pairs=crossing_pairs,
        trace_params=trace_params,
    )


__all__ = [
    "sample_all_crossings_dataset",
    "sample_axis_condition_dataset",
    "sample_axis_delta_dataset",
]
