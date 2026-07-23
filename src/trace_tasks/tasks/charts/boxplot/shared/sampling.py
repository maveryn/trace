"""Neutral sampling primitives for the boxplot chart scene."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.sampling import uniform_choice_with_probabilities
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.boxplot.shared.defaults import (
    BOXPLOT_DEFAULTS,
    GENERATION_DEFAULTS,
    SCENE_NAMESPACE,
)
from trace_tasks.tasks.charts.shared.chart_scene_types import BoxPlotSpec
from trace_tasks.tasks.charts.shared.distribution.config import (
    _build_boxplot_spec_for_median,
    _resolve_boxplot_category_count_bounds,
    _resolve_boxplot_value_bounds,
)
from trace_tasks.tasks.charts.shared.label_assets import sample_chart_labels
from trace_tasks.tasks.charts.shared.labeled_chart_values import balanced_choice_from_values
from trace_tasks.tasks.charts.shared.labeled_chart_sampling import choose_mark_count


def select_semantic_branch(
    *,
    params: Mapping[str, Any],
    branch_key: str,
    support: Sequence[str],
    instance_seed: int,
    namespace: str,
) -> tuple[str, dict[str, float], dict[str, Any]]:
    """Select a non-public semantic branch from explicit support with seeded RNG."""

    values = tuple(str(value) for value in support if str(value))
    if not values:
        raise ValueError("branch support must be non-empty")
    requested = params.get(str(branch_key))
    if requested is not None:
        selected = str(requested)
        if selected not in values:
            raise ValueError(f"unsupported {branch_key}: {selected}; supported: {values}")
        stripped = dict(params)
        stripped.pop(str(branch_key), None)
        return selected, _probabilities(values, selected), stripped

    rng = spawn_rng(int(instance_seed), str(namespace))
    selected, probabilities = uniform_choice_with_probabilities(rng, values)
    return str(selected), dict(probabilities), dict(params)


def _probabilities(values: Sequence[str], selected: str | None = None) -> dict[str, float]:
    labels = tuple(str(value) for value in values)
    if selected is not None:
        return {label: (1.0 if label == str(selected) else 0.0) for label in labels}
    if not labels:
        return {}
    weight = 1.0 / float(len(labels))
    return {label: float(weight) for label in labels}


def resolve_category_count_bounds(params: Mapping[str, Any]) -> tuple[int, int]:
    """Resolve scene category-count bounds without public task identity."""

    return _resolve_boxplot_category_count_bounds(
        params,
        gen_defaults=GENERATION_DEFAULTS,
        defaults=BOXPLOT_DEFAULTS,
        task_id=SCENE_NAMESPACE,
    )


def resolve_value_bounds(params: Mapping[str, Any], *, instance_seed: int) -> tuple[int, int]:
    """Resolve visible value bounds for one boxplot scene."""

    return _resolve_boxplot_value_bounds(
        params,
        gen_defaults=GENERATION_DEFAULTS,
        defaults=BOXPLOT_DEFAULTS,
        task_id=SCENE_NAMESPACE,
        instance_seed=int(instance_seed),
    )


def choose_category_count(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
    minimum: int | None = None,
) -> tuple[int, tuple[int, int]]:
    """Choose a category count from resolved scene bounds."""

    lower, upper = resolve_category_count_bounds(params)
    if minimum is not None:
        lower = max(int(lower), int(minimum))
    if int(lower) > int(upper):
        raise ValueError("boxplot category-count bounds are infeasible")
    count = choose_mark_count(
        list(range(int(lower), int(upper) + 1)),
        params=params,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    return int(count), (int(lower), int(upper))


def sample_labels(*, count: int, instance_seed: int, namespace: str) -> tuple[str, ...]:
    """Sample compact visible labels for the x-axis categories."""

    return tuple(
        str(label)
        for label in sample_chart_labels(
            count=int(count),
            instance_seed=int(instance_seed),
            namespace=str(namespace),
            max_chars=3,
        )
    )


def build_boxplot_for_median(
    *,
    label: str,
    median: int,
    value_min: int,
    value_max: int,
    rng: Any,
    fill_rgb: Sequence[int],
    outline_rgb: Sequence[int],
) -> BoxPlotSpec:
    """Build one valid boxplot around a fixed median value."""

    return _build_boxplot_spec_for_median(
        label=str(label),
        median=int(median),
        value_min=int(value_min),
        value_max=int(value_max),
        rng=rng,
        fill_rgb=tuple(int(channel) for channel in fill_rgb[:3]),
        outline_rgb=tuple(int(channel) for channel in outline_rgb[:3]),
    )


def quartiles_by_label(boxplots: Sequence[BoxPlotSpec]) -> dict[str, dict[str, int]]:
    """Return the visible boxplot statistics keyed by display label."""

    return {
        str(spec.label): boxplot_stats(spec)
        for spec in boxplots
    }


def boxplot_stats(spec: BoxPlotSpec) -> dict[str, int]:
    """Return integer statistics for one boxplot spec."""

    return {
        "whisker_min": int(spec.whisker_min),
        "q1": int(spec.q1),
        "median": int(spec.median),
        "q3": int(spec.q3),
        "whisker_max": int(spec.whisker_max),
        "iqr": int(spec.q3) - int(spec.q1),
    }


def sample_clustered_unique(rng: Any, pool_min: int, pool_max: int, count: int) -> list[int]:
    """Sample unique integers clustered near the high end of a support."""

    if int(count) <= 0:
        return []
    window_size = max(int(count) + 2, 4)
    lower = max(int(pool_min), int(pool_max) - int(window_size) + 1)
    pool = list(range(int(lower), int(pool_max) + 1))
    if len(pool) < int(count):
        pool = list(range(int(pool_min), int(pool_max) + 1))
    if len(pool) < int(count):
        raise ValueError("insufficient clustered support for unique sampling")
    rng.shuffle(pool)
    return [int(value) for value in pool[: int(count)]]


def sample_clustered_unique_low(rng: Any, pool_min: int, pool_max: int, count: int) -> list[int]:
    """Sample unique integers clustered near the low end of a support."""

    if int(count) <= 0:
        return []
    window_size = max(int(count) + 2, 4)
    upper = min(int(pool_max), int(pool_min) + int(window_size) - 1)
    pool = list(range(int(pool_min), int(upper) + 1))
    if len(pool) < int(count):
        pool = list(range(int(pool_min), int(pool_max) + 1))
    if len(pool) < int(count):
        raise ValueError("insufficient clustered support for unique low-end sampling")
    rng.shuffle(pool)
    return [int(value) for value in pool[: int(count)]]


def sample_unique_top_gap_margins(
    *,
    rng: Any,
    support_max: int,
    count: int,
    gap_min: int,
    gap_max: int,
) -> list[int]:
    """Sample positive margins with a unique winner separated by a target gap."""

    if int(count) < 2:
        raise ValueError("reference-median boxplot tasks require at least two queried-side candidates")
    if int(support_max) < int(count):
        raise ValueError("reference-median margin support is too small for requested category count")
    feasible_pairs: list[tuple[int, int]] = []
    for winner in range(1, int(support_max) + 1):
        for gap in range(int(gap_min), int(gap_max) + 1):
            runner_up = int(winner) - int(gap)
            if int(runner_up) < 1:
                continue
            if int(runner_up) - 1 < int(count) - 2:
                continue
            feasible_pairs.append((int(winner), int(runner_up)))
    if not feasible_pairs:
        raise ValueError("unable to construct unique reference-median winner margins")
    winner_margin, runner_up_margin = feasible_pairs[int(rng.randint(0, len(feasible_pairs) - 1))]
    other_pool = list(range(1, int(runner_up_margin)))
    other_margins = rng.sample(other_pool, int(count) - 2) if int(count) > 2 else []
    return sorted([int(winner_margin), int(runner_up_margin), *[int(value) for value in other_margins]])


def balanced_int_choice(values: Sequence[int], *, params: Mapping[str, Any], instance_seed: int, namespace: str) -> int:
    """Choose one integer while preserving the chart-domain balanced sampler."""

    return int(
        balanced_choice_from_values(
            [int(value) for value in values],
            params=params,
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
    )


__all__ = [
    "balanced_int_choice",
    "boxplot_stats",
    "build_boxplot_for_median",
    "choose_category_count",
    "quartiles_by_label",
    "resolve_value_bounds",
    "sample_clustered_unique",
    "sample_clustered_unique_low",
    "sample_labels",
    "sample_unique_top_gap_margins",
    "select_semantic_branch",
]
