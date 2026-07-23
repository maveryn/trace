"""Neutral Star Battle sampling helpers."""

from __future__ import annotations

from string import ascii_uppercase
from typing import Any, Dict, List, Mapping, Tuple

from trace_tasks.core.sampling import integer_range_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.puzzles.shared.common import get_int_range

from .rules import (
    connected_region,
    grow_regions,
    legal_cells,
    region_cells,
    sample_solution,
    scope_cells,
    star_counts_by_region,
)
from .state import CandidateCellSpec, Cell, StarBattleDataset


def build_base_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> Dict[str, Any]:
    """Sample a partial Star Battle board with at least one legal next cell."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.base")
    size_min, size_max = get_int_range(
        params,
        generation_defaults,
        min_key="grid_size_min",
        max_key="grid_size_max",
        fallback_min=6,
        fallback_max=9,
    )
    explicit_size = params.get("grid_size")
    if explicit_size is None:
        size, _size_probs = integer_range_choice(rng, int(size_min), int(size_max))
    else:
        size = int(explicit_size)
    if not int(size_min) <= int(size) <= int(size_max):
        raise ValueError("grid_size outside configured range")

    stars = sample_solution(int(size), rng=rng)
    region_grid = grow_regions(int(size), stars, rng=rng)
    for region_index in range(int(size)):
        if not connected_region(region_grid, int(region_index)):
            raise RuntimeError("Star Battle region is not connected")
    region_star_counts = star_counts_by_region(stars, region_grid, int(size))
    if any(count != 1 for count in region_star_counts):
        raise RuntimeError("Star Battle generated region without exactly one solution star")

    reveal_min_frac = float(params.get("visible_star_fraction_min", generation_defaults.get("visible_star_fraction_min", 0.30)))
    reveal_max_frac = float(params.get("visible_star_fraction_max", generation_defaults.get("visible_star_fraction_max", 0.55)))
    reveal_min = max(1, int(round(float(size) * float(reveal_min_frac))))
    reveal_max = max(reveal_min, min(int(size) - 1, int(round(float(size) * float(reveal_max_frac)))))
    reveal_count, _reveal_probs = integer_range_choice(rng, int(reveal_min), int(reveal_max))
    shuffled_stars = list(stars)
    rng.shuffle(shuffled_stars)
    visible_stars = sorted([tuple(cell) for cell in shuffled_stars[: int(reveal_count)]])
    legal = legal_cells(size=int(size), region_grid=region_grid, visible_stars=visible_stars)
    if not legal:
        raise RuntimeError("Star Battle partial board has no legal cells")

    regions = {
        str(region_index): tuple(tuple(cell) for cell in cells)
        for region_index, cells in region_cells(region_grid).items()
    }
    return {
        "size": int(size),
        "grid_size_range": (int(size_min), int(size_max)),
        "solution_stars": tuple(tuple(cell) for cell in stars),
        "visible_stars": tuple(tuple(cell) for cell in visible_stars),
        "region_grid": tuple(tuple(int(value) for value in row) for row in region_grid),
        "regions": regions,
        "legal_cells": tuple(tuple(cell) for cell in legal),
    }


def choose_scope(
    *,
    scope_kind: str,
    base: Mapping[str, Any],
    target_min: int,
    target_max: int,
    rng,
) -> Tuple[Dict[str, Any], Tuple[Cell, ...], Tuple[Cell, ...]]:
    """Choose a visible row/column/region scope with a target legal-cell count."""

    size = int(base["size"])
    legal_set = {tuple(cell) for cell in base["legal_cells"]}
    candidates: List[Tuple[Dict[str, Any], Tuple[Cell, ...], Tuple[Cell, ...]]] = []
    regions = dict(base["regions"])

    if str(scope_kind) == "marked_region":
        for region_index, cells in regions.items():
            scope = tuple(tuple(cell) for cell in cells)
            scoped_legal = tuple(sorted([cell for cell in scope if cell in legal_set]))
            if int(target_min) <= len(scoped_legal) <= int(target_max):
                candidates.append(({"marked_region_index": int(region_index)}, scope, scoped_legal))
    elif str(scope_kind) == "marked_row":
        for row in range(size):
            scope = tuple((row, col) for col in range(size))
            scoped_legal = tuple(sorted([cell for cell in scope if cell in legal_set]))
            if int(target_min) <= len(scoped_legal) <= int(target_max):
                candidates.append(({"marked_row_index": int(row)}, scope, scoped_legal))
    elif str(scope_kind) == "marked_column":
        for col in range(size):
            scope = tuple((row, col) for row in range(size))
            scoped_legal = tuple(sorted([cell for cell in scope if cell in legal_set]))
            if int(target_min) <= len(scoped_legal) <= int(target_max):
                candidates.append(({"marked_col_index": int(col)}, scope, scoped_legal))
    elif str(scope_kind) == "whole_board":
        scope = tuple((row, col) for row in range(size) for col in range(size))
        scoped_legal = tuple(sorted([cell for cell in scope if cell in legal_set]))
        if int(target_min) <= len(scoped_legal) <= int(target_max):
            candidates.append(({}, scope, scoped_legal))
    else:
        raise ValueError(f"unsupported Star Battle scope kind: {scope_kind}")

    if not candidates:
        raise RuntimeError(f"could not find Star Battle scope for {scope_kind}")
    return candidates[int(rng.randrange(len(candidates)))]


def build_valid_cell_dataset(
    *,
    scope_kind: str,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> StarBattleDataset:
    """Sample a labeled-option board with exactly one legal candidate cell."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.valid.{scope_kind}")
    option_min, option_max = get_int_range(
        params,
        generation_defaults,
        min_key="option_count_min",
        max_key="option_count_max",
        fallback_min=5,
        fallback_max=8,
    )
    explicit_option_count = params.get("option_count")
    if explicit_option_count is None:
        option_count, _option_probs = integer_range_choice(rng, int(option_min), int(option_max))
    else:
        option_count = int(explicit_option_count)
    option_count = max(4, min(8, int(option_count)))
    labels = list(ascii_uppercase[:option_count])

    for attempt in range(300):
        base = build_base_dataset(
            params=params,
            instance_seed=int(instance_seed) + int(attempt),
            generation_defaults=generation_defaults,
            namespace=namespace,
        )
        scope_info, scope, scoped_legal = choose_scope(
            scope_kind=str(scope_kind),
            base=base,
            target_min=1,
            target_max=int(base["size"]) * int(base["size"]),
            rng=rng,
        )
        legal_set = {tuple(cell) for cell in scoped_legal}
        visible_star_set = {tuple(cell) for cell in base["visible_stars"]}
        invalid_scope = [
            tuple(cell)
            for cell in scope
            if tuple(cell) not in legal_set and tuple(cell) not in visible_star_set
        ]
        if len(scoped_legal) < 1 or len(invalid_scope) < int(option_count) - 1:
            continue
        correct_cell = tuple(scoped_legal[int(rng.randrange(len(scoped_legal)))])
        rng.shuffle(invalid_scope)
        distractors = list(invalid_scope[: int(option_count) - 1])
        correct_index = int(rng.randrange(int(option_count)))
        ordered: List[Cell] = []
        cursor = 0
        for index in range(int(option_count)):
            if index == correct_index:
                ordered.append(correct_cell)
            else:
                ordered.append(tuple(distractors[cursor]))
                cursor += 1
        candidate_specs = tuple(
            CandidateCellSpec(
                label=str(labels[index]),
                row=int(cell[0]),
                col=int(cell[1]),
                is_correct=bool(index == correct_index),
                is_legal=bool(tuple(cell) in legal_set),
            )
            for index, cell in enumerate(ordered)
        )
        if sum(1 for spec in candidate_specs if spec.is_legal) != 1:
            continue
        return StarBattleDataset(
            **base,
            **scope_info,
            scope_cells=tuple(tuple(cell) for cell in scope),
            scoped_legal_cells=tuple(tuple(cell) for cell in scoped_legal),
            candidate_specs=candidate_specs,
            answer_value=str(labels[correct_index]),
            answer_type="option_letter",
            option_count=int(option_count),
            target_answer_support=tuple(labels),
            correct_option_index=int(correct_index),
            correct_cell=tuple(correct_cell),
        )
    raise RuntimeError(f"failed to build Star Battle valid-cell dataset for {scope_kind}")


def build_remaining_count_dataset(
    *,
    scope_kind: str,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> StarBattleDataset:
    """Sample a marked scope whose legal Star Battle cells match a target count."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.count.{scope_kind}")
    count_min, count_max = get_int_range(
        params,
        generation_defaults,
        min_key="target_count_min",
        max_key="target_count_max",
        fallback_min=1,
        fallback_max=6,
    )
    support = tuple(int(value) for value in range(int(count_min), int(count_max) + 1))
    if not support or min(support) < 1:
        raise ValueError("Star Battle remaining-count support must be positive")
    target_count, _count_probs = integer_range_choice(rng, int(count_min), int(count_max))

    for attempt in range(300):
        base = build_base_dataset(
            params=params,
            instance_seed=int(instance_seed) + int(attempt),
            generation_defaults=generation_defaults,
            namespace=namespace,
        )
        scope_info, scope, scoped_legal = choose_scope(
            scope_kind=str(scope_kind),
            base=base,
            target_min=int(target_count),
            target_max=int(target_count),
            rng=rng,
        )
        return StarBattleDataset(
            **base,
            **scope_info,
            scope_cells=tuple(tuple(cell) for cell in scope),
            scoped_legal_cells=tuple(tuple(cell) for cell in scoped_legal),
            candidate_specs=tuple(),
            answer_value=int(len(scoped_legal)),
            answer_type="integer",
            option_count=0,
            target_answer_support=support,
            target_count_range=(int(count_min), int(count_max)),
        )
    raise RuntimeError(f"failed to build Star Battle remaining-count dataset for {scope_kind}")


__all__ = [
    "build_base_dataset",
    "build_remaining_count_dataset",
    "build_valid_cell_dataset",
    "choose_scope",
]
