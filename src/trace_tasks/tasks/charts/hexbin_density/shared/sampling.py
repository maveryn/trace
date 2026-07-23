"""Sampling primitives for hexbin-density charts."""

from __future__ import annotations

import math
from typing import Any, Mapping, Tuple

from trace_tasks.core.sampling import uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.hexbin_density.shared.defaults import (
    SCENE_NAMESPACE,
    balanced_int,
    resolve_bounds,
    resolve_density_palette,
)
from trace_tasks.tasks.charts.hexbin_density.shared.state import HexBin, HexbinDataset, ThresholdQuery


def occupied_cells(
    *,
    row_count: int,
    column_count: int,
    occupied_count: int,
    instance_seed: int,
) -> Tuple[Tuple[int, int], ...]:
    all_cells = [(row, col) for row in range(int(row_count)) for col in range(int(column_count))]
    if int(occupied_count) >= len(all_cells):
        return tuple(all_cells)
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.occupied_cells")
    center_count = int(
        uniform_choice(
            spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.cluster_count"),
            (2, 3),
            sort_keys=True,
        )
    )
    centers = [(rng.uniform(0.15, 0.85), rng.uniform(0.15, 0.85)) for _ in range(int(center_count))]

    def score(cell: Tuple[int, int]) -> float:
        row, col = cell
        x = (float(col) + 0.5) / float(max(1, int(column_count)))
        y = (float(row) + 0.5) / float(max(1, int(row_count)))
        dist = min(math.hypot(float(x) - float(cx), float(y) - float(cy)) for cx, cy in centers)
        return float(dist) + rng.uniform(0.0, 0.18)

    chosen = sorted(all_cells, key=score)[: int(occupied_count)]
    return tuple(sorted(chosen, key=lambda item: (int(item[0]), int(item[1]))))


def answer_support(params: Mapping[str, Any], *, occupied_count: int) -> Tuple[int, ...]:
    answer_min, answer_max = resolve_bounds(
        params,
        min_key="threshold_count_answer_min",
        max_key="threshold_count_answer_max",
        fallback_min=4,
        fallback_max=18,
    )
    answer_max = min(int(answer_max), max(1, int(occupied_count) - 1))
    answer_min = min(max(1, int(answer_min)), int(answer_max))
    return tuple(range(int(answer_min), int(answer_max) + 1))


def build_threshold_dataset(
    params: Mapping[str, Any],
    *,
    instance_seed: int,
    threshold_direction: str,
) -> HexbinDataset:
    """Construct a density field whose one-bound threshold count is controlled."""

    row_min, row_max = resolve_bounds(params, min_key="row_count_min", max_key="row_count_max", fallback_min=5, fallback_max=7)
    col_min, col_max = resolve_bounds(
        params,
        min_key="column_count_min",
        max_key="column_count_max",
        fallback_min=7,
        fallback_max=10,
    )
    row_count = balanced_int(
        params,
        key="row_count",
        support=range(int(row_min), int(row_max) + 1),
        instance_seed=int(instance_seed),
        namespace="row_count",
    )
    column_count = balanced_int(
        params,
        key="column_count",
        support=range(int(col_min), int(col_max) + 1),
        instance_seed=int(instance_seed),
        namespace="column_count",
    )
    occupied_min, occupied_max = resolve_bounds(
        params,
        min_key="occupied_bin_count_min",
        max_key="occupied_bin_count_max",
        fallback_min=24,
        fallback_max=42,
    )
    occupied_cap = int(row_count) * int(column_count)
    occupied_support = range(min(int(occupied_min), occupied_cap), min(int(occupied_max), occupied_cap) + 1)
    occupied_count = balanced_int(
        params,
        key="occupied_bin_count",
        support=occupied_support,
        instance_seed=int(instance_seed),
        namespace="occupied_bin_count",
    )
    threshold_min, threshold_max = resolve_bounds(
        params,
        min_key="density_threshold_level_min",
        max_key="density_threshold_level_max",
        fallback_min=2,
        fallback_max=5,
    )
    threshold_support = range(max(2, int(threshold_min)), min(5, int(threshold_max)) + 1)
    threshold_level = balanced_int(
        params,
        key="density_threshold_level",
        support=threshold_support,
        instance_seed=int(instance_seed),
        namespace="density_threshold_level",
    )
    target_count = balanced_int(
        params,
        key="threshold_count_answer",
        support=answer_support(params, occupied_count=int(occupied_count)),
        instance_seed=int(instance_seed),
        namespace="threshold_count_answer",
    )
    cells = occupied_cells(
        row_count=int(row_count),
        column_count=int(column_count),
        occupied_count=int(occupied_count),
        instance_seed=int(instance_seed),
    )
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.density_levels")
    cell_indices = list(range(len(cells)))
    rng.shuffle(cell_indices)
    matching_indices = set(cell_indices[: int(target_count)])
    palette_scheme, palette, palette_trace = resolve_density_palette(params, instance_seed=int(instance_seed))
    bins: list[HexBin] = []
    direction = str(threshold_direction)
    for index, (row, col) in enumerate(cells):
        if direction == "above":
            choices = list(range(int(threshold_level), 6)) if index in matching_indices else list(range(1, int(threshold_level)))
        elif direction == "below":
            choices = list(range(1, int(threshold_level))) if index in matching_indices else list(range(int(threshold_level), 6))
        else:
            raise ValueError(f"unsupported threshold_direction: {threshold_direction}")
        if not choices:
            raise RuntimeError("invalid density-threshold construction")
        level = int(choices[rng.randrange(len(choices))])
        bins.append(
            HexBin(
                bin_id=f"bin_r{int(row):02d}_c{int(col):02d}",
                row_index=int(row),
                column_index=int(col),
                density_level=int(level),
                fill_rgb=tuple(int(channel) for channel in palette[int(level) - 1]),
            )
        )
    bins_sorted = tuple(sorted(bins, key=lambda item: (int(item.row_index), int(item.column_index))))
    if direction == "above":
        annotation_bins = tuple(bin_item for bin_item in bins_sorted if int(bin_item.density_level) >= int(threshold_level))
        phrase = "at least"
        operator = ">="
    else:
        annotation_bins = tuple(bin_item for bin_item in bins_sorted if int(bin_item.density_level) < int(threshold_level))
        phrase = "below"
        operator = "<"
    if len(annotation_bins) != int(target_count):
        raise RuntimeError("hexbin threshold construction lost target count")
    return HexbinDataset(
        row_count=int(row_count),
        column_count=int(column_count),
        bins=bins_sorted,
        query=ThresholdQuery(
            threshold_direction=str(direction),
            threshold_phrase=str(phrase),
            threshold_operator=str(operator),
            threshold_level=int(threshold_level),
            answer=int(target_count),
            annotation_bin_ids=tuple(str(bin_item.bin_id) for bin_item in annotation_bins),
            trace={
                "density_threshold_direction": str(direction),
                "density_threshold_phrase": str(phrase),
                "density_threshold_operator": str(operator),
                "density_threshold_level": int(threshold_level),
                "matching_bin_ids": [str(bin_item.bin_id) for bin_item in annotation_bins],
                "density_level_by_bin_id": {str(bin_item.bin_id): int(bin_item.density_level) for bin_item in bins_sorted},
            },
        ),
        density_palette_scheme=str(palette_scheme),
        density_palette_rgb=tuple(palette),
        density_palette_trace=dict(palette_trace),
    )


__all__ = [
    "answer_support",
    "build_threshold_dataset",
    "occupied_cells",
]
