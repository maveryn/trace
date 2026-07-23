"""Neutral sampling primitives for histogram chart scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.charts.histogram.shared.defaults import (
    GENERATION_DEFAULTS,
    HISTOGRAM_DEFAULTS,
    SCENE_NAMESPACE,
)
from trace_tasks.tasks.charts.shared.chart_scene_types import HistogramBinSpec
from trace_tasks.tasks.charts.shared.distribution.histogram import build_histogram_dataset_for_variant


def build_histogram_dataset(
    *,
    dataset_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
    mark_style: Mapping[str, Any],
) -> tuple[list[HistogramBinSpec], int, list[str], dict[str, Any]]:
    """Build one histogram dataset for a semantic variant.

    ``dataset_variant`` names a chart-construction primitive, not a public
    task/query id. Public task files decide which variant their objective uses.
    """

    return build_histogram_dataset_for_variant(
        query_id=str(dataset_variant),
        params=dict(params),
        instance_seed=int(instance_seed),
        gen_defaults=GENERATION_DEFAULTS,
        defaults=HISTOGRAM_DEFAULTS,
        task_id=SCENE_NAMESPACE,
        mark_style=dict(mark_style),
    )


def counts_by_label_from_bins(bins: Sequence[HistogramBinSpec]) -> dict[str, int]:
    """Return visible counts keyed by x-axis label."""

    return {str(bin_spec.label): int(bin_spec.count) for bin_spec in bins}


def labels_from_bins(bins: Sequence[HistogramBinSpec]) -> list[str]:
    """Return visible histogram x-axis labels."""

    return [str(bin_spec.label) for bin_spec in bins]


__all__ = [
    "build_histogram_dataset",
    "counts_by_label_from_bins",
    "labels_from_bins",
]
