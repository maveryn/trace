"""Scene-neutral sampling primitives for population-pyramid charts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .defaults import resolve_row_count, resolve_series_colors, resolve_series_labels, sample_age_labels, sample_title
from .state import PopulationPyramidBase, PopulationPyramidDataset, PopulationPyramidQuery, PopulationPyramidRow


def sample_scene_base(params: Mapping[str, Any], *, instance_seed: int) -> PopulationPyramidBase:
    row_count, row_count_probabilities = resolve_row_count(params, instance_seed=int(instance_seed))
    age_labels, age_label_meta = sample_age_labels(params, row_count=int(row_count), instance_seed=int(instance_seed))
    left_label, right_label, series_label_meta = resolve_series_labels(params, instance_seed=int(instance_seed))
    left_color, right_color = resolve_series_colors(params, instance_seed=int(instance_seed))
    return PopulationPyramidBase(
        age_labels=tuple(age_labels),
        left_series_label=str(left_label),
        right_series_label=str(right_label),
        left_color_rgb=tuple(left_color),
        right_color_rgb=tuple(right_color),
        title=sample_title(params, instance_seed=int(instance_seed)),
        params={
            "row_count": int(row_count),
            "row_count_probabilities": dict(row_count_probabilities),
            "age_label_meta": dict(age_label_meta),
            "series_label_meta": dict(series_label_meta),
        },
    )


def sample_pair_for_gap(rng: Any, *, gap: int, value_min: int, value_max: int, direction: int) -> tuple[int, int]:
    lo = int(value_min)
    hi = int(value_max) - int(gap)
    if hi < lo:
        raise ValueError("gap exceeds value support")
    base = int(rng.randint(lo, hi))
    if int(direction) >= 0:
        return int(base + int(gap)), int(base)
    return int(base), int(base + int(gap))


def sample_pair_for_total(rng: Any, *, total_min: int, total_max: int, value_min: int, value_max: int) -> tuple[int, int]:
    feasible_totals = [
        int(total)
        for total in range(int(total_min), int(total_max) + 1)
        if (2 * int(value_min)) <= int(total) <= (2 * int(value_max))
    ]
    if not feasible_totals:
        raise ValueError("empty total support")
    total = int(rng.choice(feasible_totals))
    left_low = max(int(value_min), int(total) - int(value_max))
    left_high = min(int(value_max), int(total) - int(value_min))
    left = int(rng.randint(int(left_low), int(left_high)))
    right = int(total) - int(left)
    return int(left), int(right)


def threshold_metric(row: PopulationPyramidRow, side: str) -> int:
    if str(side) == "left":
        return int(row.left_value)
    if str(side) == "right":
        return int(row.right_value)
    if str(side) == "combined_total":
        return int(row.total)
    raise ValueError(f"unsupported threshold side: {side}")


def build_dataset_from_rows(
    *,
    base: PopulationPyramidBase,
    rows: tuple[PopulationPyramidRow, ...],
    branch_id: str,
    branch_probabilities: Mapping[str, float],
    answer: int | str,
    answer_type: str,
    annotation_type: str,
    annotation_row_ids: tuple[str, ...],
    params: Mapping[str, Any],
) -> PopulationPyramidDataset:
    payload_params = {
        **dict(params),
        **dict(base.params),
    }
    query = PopulationPyramidQuery(
        branch_id=str(branch_id),
        answer=answer,
        answer_type=str(answer_type),
        annotation_type=str(annotation_type),
        annotation_row_ids=tuple(str(row_id) for row_id in annotation_row_ids),
        params=dict(payload_params),
    )
    return PopulationPyramidDataset(
        left_series_label=str(base.left_series_label),
        right_series_label=str(base.right_series_label),
        left_color_rgb=tuple(base.left_color_rgb),
        right_color_rgb=tuple(base.right_color_rgb),
        rows=tuple(rows),
        branch_id=str(branch_id),
        branch_probabilities=dict(branch_probabilities),
        query=query,
        title=str(base.title),
    )
