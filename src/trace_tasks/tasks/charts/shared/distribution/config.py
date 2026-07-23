"""Shared generators for histogram and boxplot chart tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from .....core.seed import spawn_rng
from ....shared.config_defaults import resolve_required_int_bounds
from ..chart_scene_types import BoxPlotSpec, HistogramBinSpec, ViolinPlotSpec
from ..label_assets import sample_chart_labels
from ..labeled_chart_defaults import LabeledChartDefaults
from ..labeled_chart_marks import resolve_chart_mark_colors
from ..labeled_chart_values import balanced_choice_from_values, resolve_value_bounds
from ..labeled_chart_variants import resolve_chart_axis_variant
from ..labeled_chart_sampling import choose_mark_count
from ..labeled_chart_render_params import resolve_chart_render_params_for_task


HistogramQueryVariant = str
BoxPlotQueryVariant = str
DensityQueryVariant = str

_CUMULATIVE_HISTOGRAM_VARIANTS = {
    "rank_item_bin_label",
}


@dataclass(frozen=True)
class DistributionChartDefaults:
    """Stable fallback defaults for distribution-style chart tasks."""

    bin_count_min: int = 8
    bin_count_max: int = 20
    bin_width_min: int = 1
    bin_width_max: int = 1
    bin_start_min: int = 1
    bin_start_max: int = 99
    bin_frequency_min: int = 1
    bin_frequency_max: int = 20
    interval_bin_span_min: int = 5
    interval_bin_span_max: int = 15
    outside_interval_bin_count_min: int = 2
    outside_interval_bin_count_max: int = 10
    category_count_min: int = 6
    category_count_max: int = 15
    value_min: int = 1
    value_max: int = 20
    violin_category_count_min: int = 4
    violin_category_count_max: int = 7


def _resolve_histogram_bin_count_bounds(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    defaults: DistributionChartDefaults,
    task_id: str,
) -> Tuple[int, int]:
    return resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="bin_count_min",
        max_key="bin_count_max",
        fallback_min=int(defaults.bin_count_min),
        fallback_max=int(defaults.bin_count_max),
        context=f"generation defaults for {task_id}",
    )


def _resolve_boxplot_category_count_bounds(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    defaults: DistributionChartDefaults,
    task_id: str,
) -> Tuple[int, int]:
    return resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="category_count_min",
        max_key="category_count_max",
        fallback_min=int(defaults.category_count_min),
        fallback_max=int(defaults.category_count_max),
        context=f"generation defaults for {task_id}",
    )


def _resolve_violin_category_count_bounds(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    defaults: DistributionChartDefaults,
    task_id: str,
) -> Tuple[int, int]:
    return resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="violin_category_count_min",
        max_key="violin_category_count_max",
        fallback_min=int(defaults.violin_category_count_min),
        fallback_max=int(defaults.violin_category_count_max),
        context=f"generation defaults for {task_id}",
    )


def _resolve_bin_width_bounds(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    defaults: DistributionChartDefaults,
    task_id: str,
) -> Tuple[int, int]:
    return resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="bin_width_min",
        max_key="bin_width_max",
        fallback_min=int(defaults.bin_width_min),
        fallback_max=int(defaults.bin_width_max),
        context=f"generation defaults for {task_id}",
    )


def _resolve_bin_start_bounds(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    defaults: DistributionChartDefaults,
    task_id: str,
) -> Tuple[int, int]:
    return resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="bin_start_min",
        max_key="bin_start_max",
        fallback_min=int(defaults.bin_start_min),
        fallback_max=int(defaults.bin_start_max),
        context=f"generation defaults for {task_id}",
    )


def _resolve_bin_frequency_bounds(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    defaults: DistributionChartDefaults,
    task_id: str,
) -> Tuple[int, int]:
    return resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="bin_frequency_min",
        max_key="bin_frequency_max",
        fallback_min=int(defaults.bin_frequency_min),
        fallback_max=int(defaults.bin_frequency_max),
        context=f"generation defaults for {task_id}",
    )


def _resolve_interval_bin_span_bounds(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    defaults: DistributionChartDefaults,
    task_id: str,
) -> Tuple[int, int]:
    return resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="interval_bin_span_min",
        max_key="interval_bin_span_max",
        fallback_min=int(defaults.interval_bin_span_min),
        fallback_max=int(defaults.interval_bin_span_max),
        context=f"generation defaults for {task_id}",
    )


def _resolve_outside_interval_bin_count_bounds(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    defaults: DistributionChartDefaults,
    task_id: str,
) -> Tuple[int, int]:
    return resolve_required_int_bounds(
        params,
        gen_defaults,
        min_key="outside_interval_bin_count_min",
        max_key="outside_interval_bin_count_max",
        fallback_min=int(defaults.outside_interval_bin_count_min),
        fallback_max=int(defaults.outside_interval_bin_count_max),
        context=f"generation defaults for {task_id}",
    )


def _resolve_boxplot_value_bounds(
    params: Mapping[str, Any],
    *,
    gen_defaults: Mapping[str, Any],
    defaults: DistributionChartDefaults,
    task_id: str,
    instance_seed: int | None = None,
) -> Tuple[int, int]:
    chart_defaults = LabeledChartDefaults(value_min=int(defaults.value_min), value_max=int(defaults.value_max))
    return resolve_value_bounds(
        params,
        gen_defaults=gen_defaults,
        defaults=chart_defaults,
        task_id=task_id,
        instance_seed=instance_seed,
    )


def _build_boxplot_spec_for_median(
    *,
    label: str,
    median: int,
    value_min: int,
    value_max: int,
    rng: Any,
    fill_rgb: Tuple[int, int, int],
    outline_rgb: Tuple[int, int, int],
) -> BoxPlotSpec:
    """Construct one valid boxplot summary around a chosen median."""

    left_cap = min(3, int(median) - int(value_min) - 1)
    right_cap = min(3, int(value_max) - int(median))
    if int(left_cap) < 1 or int(right_cap) < 1:
        raise ValueError("no feasible quartile support for requested median")
    left_delta = int(rng.randint(1, int(left_cap)))
    right_delta = int(rng.randint(1, int(right_cap)))
    q1 = int(median) - int(left_delta)
    q3 = int(median) + int(right_delta)
    whisker_min = max(int(value_min), int(q1) - int(rng.randint(0, min(2, int(q1) - int(value_min)))))
    whisker_max = min(int(value_max), int(q3) + int(rng.randint(0, min(2, int(value_max) - int(q3)))))
    return BoxPlotSpec(
        label=str(label),
        whisker_min=int(whisker_min),
        q1=int(q1),
        median=int(median),
        q3=int(q3),
        whisker_max=int(whisker_max),
        fill_rgb=fill_rgb,
        outline_rgb=outline_rgb,
    )


def _resolve_optional_positive_int_bounds(
    params: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    gen_defaults: Mapping[str, Any] | None = None,
) -> Tuple[int, int] | None:
    """Resolve one optional positive integer bounds pair."""

    defaults = gen_defaults if isinstance(gen_defaults, Mapping) else {}
    min_value = params.get(min_key, defaults.get(min_key))
    max_value = params.get(max_key, defaults.get(max_key))
    if min_value is None and max_value is None:
        return None
    if min_value is None or max_value is None:
        raise ValueError(f"{min_key} and {max_key} must be set together")
    min_int = int(min_value)
    max_int = int(max_value)
    if min_int <= 0 or max_int <= 0:
        raise ValueError(f"{min_key} and {max_key} must be positive")
    if min_int > max_int:
        raise ValueError(f"{min_key} must be <= {max_key}")
    return min_int, max_int


def _resolve_optional_nonnegative_int_bounds(
    params: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    gen_defaults: Mapping[str, Any] | None = None,
) -> Tuple[int, int] | None:
    """Resolve one optional nonnegative integer bounds pair."""

    defaults = gen_defaults if isinstance(gen_defaults, Mapping) else {}
    min_value = params.get(min_key, defaults.get(min_key))
    max_value = params.get(max_key, defaults.get(max_key))
    if min_value is None and max_value is None:
        return None
    if min_value is None or max_value is None:
        raise ValueError(f"{min_key} and {max_key} must be set together")
    min_int = int(min_value)
    max_int = int(max_value)
    if min_int < 0 or max_int < 0:
        raise ValueError(f"{min_key} and {max_key} must be nonnegative")
    if min_int > max_int:
        raise ValueError(f"{min_key} must be <= {max_key}")
    return min_int, max_int


__all__ = [
    "BoxPlotQueryVariant",
    "DensityQueryVariant",
    "DistributionChartDefaults",
    "HistogramQueryVariant",
]
