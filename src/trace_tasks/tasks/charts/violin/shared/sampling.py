"""Neutral sampling primitives for violin chart tasks."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.shared.distribution.density import build_density_dataset_for_variant
from trace_tasks.tasks.charts.violin.shared.defaults import (
    DATASET_NAMESPACE,
    DISTRIBUTION_DEFAULTS,
    GENERATION_DEFAULTS,
)


def build_violin_dataset(
    *,
    dataset_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
    mark_style: Mapping[str, Any],
) -> tuple[Any, Any, Any, dict[str, Any]]:
    """Build one violin dataset for a public task's selected semantic variant."""

    return build_density_dataset_for_variant(
        query_id=str(dataset_variant),
        params=dict(params),
        instance_seed=int(instance_seed),
        gen_defaults=GENERATION_DEFAULTS,
        defaults=DISTRIBUTION_DEFAULTS,
        task_id=DATASET_NAMESPACE,
        mark_style=dict(mark_style),
    )


def normalize_support_trace(trace_extras: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Normalize symbolic support metadata by visible label."""

    return {
        str(label): dict(values)
        for label, values in trace_extras["support_by_label"].items()
    }


__all__ = ["build_violin_dataset", "normalize_support_trace"]
