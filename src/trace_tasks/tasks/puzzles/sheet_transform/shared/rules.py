"""Sampling primitives for transparent-sheet overlay puzzles."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import (
    integer_range_choice,
    sample_without_replacement,
    support_probability_map,
    uniform_choice,
)
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import resolve_required_int_bounds
from trace_tasks.tasks.shared.mcq import option_label_for_index

from .constraints import resolve_correct_option_index
from .state import Cells, OverlayDataset


def _canonicalize_cells(cells: Iterable[Tuple[int, int]]) -> Cells:
    """Return cells in stable row-major order."""

    return tuple(sorted((int(cell_x), int(cell_y)) for cell_x, cell_y in cells))


def _all_grid_cells(grid_size: int) -> tuple[tuple[int, int], ...]:
    """Return all cells in one square grid."""

    return tuple(
        (int(cell_x), int(cell_y))
        for cell_y in range(int(grid_size))
        for cell_x in range(int(grid_size))
    )


def _mark_specs(cells: Cells, *, prefix: str) -> tuple[Dict[str, Any], ...]:
    """Build deterministic mark specs from canonical cells."""

    return tuple(
        {
            "mark_id": f"{str(prefix)}_{int(index)}",
            "cell": [int(cell_x), int(cell_y)],
        }
        for index, (cell_x, cell_y) in enumerate(cells, start=1)
    )


def _resolve_integer_axis(
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    *,
    exact_key: str,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
    rng,
    context: str,
) -> tuple[int, tuple[int, int], dict[str, float]]:
    """Resolve an integer support and sample from it without seed modulo."""

    lower, upper = resolve_required_int_bounds(
        params,
        generation_defaults,
        min_key=str(min_key),
        max_key=str(max_key),
        fallback_min=int(fallback_min),
        fallback_max=int(fallback_max),
        context=str(context),
    )
    support = tuple(range(int(lower), int(upper) + 1))
    explicit = params.get(str(exact_key))
    if explicit is not None:
        selected = int(explicit)
        if selected not in set(support):
            raise ValueError(f"{exact_key} must be inside {support}")
        return (
            int(selected),
            (int(lower), int(upper)),
            support_probability_map(support, selected=int(selected), sort_keys=True),
        )
    selected, probabilities = integer_range_choice(rng, int(lower), int(upper))
    return int(selected), (int(lower), int(upper)), dict(probabilities)


def _sample_overlay_sources(
    *,
    grid_size: int,
    sheet_mark_count_min: int,
    sheet_mark_count_max: int,
    overlap_count_min: int,
    overlap_count_max: int,
    rng,
) -> tuple[Cells, Cells, Cells, Cells, tuple[int, int, int, int]]:
    """Sample source sheets with one non-trivial union overlap relation."""

    feasible: list[tuple[int, int, int, int]] = []
    cell_capacity = int(grid_size * grid_size)
    for left_count in range(int(sheet_mark_count_min), int(sheet_mark_count_max) + 1):
        for right_count in range(int(sheet_mark_count_min), int(sheet_mark_count_max) + 1):
            for overlap_count in range(int(overlap_count_min), int(overlap_count_max) + 1):
                if int(overlap_count) >= int(left_count):
                    continue
                if int(overlap_count) >= int(right_count):
                    continue
                union_count = int(left_count + right_count - overlap_count)
                if int(union_count) <= int(cell_capacity):
                    feasible.append(
                        (
                            int(left_count),
                            int(right_count),
                            int(overlap_count),
                            int(union_count),
                        )
                    )
    if not feasible:
        raise ValueError("no feasible overlay source configuration")

    left_count, right_count, overlap_count, union_count = uniform_choice(rng, feasible)
    union_cells = _canonicalize_cells(
        sample_without_replacement(
            rng,
            _all_grid_cells(int(grid_size)),
            int(union_count),
        )
    )
    overlap_cells = _canonicalize_cells(
        sample_without_replacement(rng, union_cells, int(overlap_count))
    )
    overlap_set = set(overlap_cells)
    remainder = [cell for cell in union_cells if cell not in overlap_set]
    rng.shuffle(remainder)
    left_only_count = int(left_count - overlap_count)
    left_only_cells = remainder[:left_only_count]
    right_only_cells = remainder[left_only_count:]
    left_cells = _canonicalize_cells(tuple(list(overlap_cells) + list(left_only_cells)))
    right_cells = _canonicalize_cells(tuple(list(overlap_cells) + list(right_only_cells)))
    return (
        left_cells,
        right_cells,
        overlap_cells,
        union_cells,
        (int(left_count), int(right_count), int(overlap_count), int(union_count)),
    )


def _option_signature(cells: Iterable[tuple[int, int]]) -> Cells:
    """Canonicalize one option signature."""

    return _canonicalize_cells(tuple(cells))


def _build_overlay_options(
    *,
    left_cells: Cells,
    right_cells: Cells,
    overlap_cells: Cells,
    union_cells: Cells,
    grid_size: int,
    option_count: int,
    correct_option_index: int,
    rng,
) -> tuple[tuple[Dict[str, Any], ...], str]:
    """Build labeled option specs with exactly one correct union result."""

    union_signature = _option_signature(union_cells)
    all_cells = set(_all_grid_cells(int(grid_size)))
    distractors: list[tuple[Cells, str]] = []
    seen = {union_signature}

    def _add(candidate_cells: Iterable[tuple[int, int]], *, kind: str) -> None:
        signature = _option_signature(candidate_cells)
        if not signature or signature in seen:
            return
        seen.add(signature)
        distractors.append((signature, str(kind)))

    _add(left_cells, kind="left_only")
    _add(right_cells, kind="right_only")
    _add(overlap_cells, kind="overlap_only")

    for removed in union_signature:
        _add((cell for cell in union_signature if cell != removed), kind="missing_one")
        if len(distractors) >= int(option_count) - 1:
            break
    if len(distractors) < int(option_count) - 1:
        for added in sorted(all_cells - set(union_signature)):
            _add(tuple(list(union_signature) + [added]), kind="extra_one")
            if len(distractors) >= int(option_count) - 1:
                break
    if len(distractors) < int(option_count) - 1:
        empty_cells = sorted(all_cells - set(union_signature))
        for removed in union_signature:
            for added in empty_cells:
                moved = tuple(cell for cell in union_signature if cell != removed) + (added,)
                _add(moved, kind="moved_one")
                if len(distractors) >= int(option_count) - 1:
                    break
            if len(distractors) >= int(option_count) - 1:
                break
    if len(distractors) < int(option_count) - 1:
        raise ValueError("failed to construct enough overlay distractors")

    rng.shuffle(distractors)
    option_signatures = [cells for cells, _kind in distractors[: int(option_count) - 1]]
    option_kinds = [kind for _cells, kind in distractors[: int(option_count) - 1]]
    option_signatures.insert(int(correct_option_index), union_signature)
    option_kinds.insert(int(correct_option_index), "union")
    answer_option_label = str(option_label_for_index(int(correct_option_index)))

    option_specs: list[Dict[str, Any]] = []
    for option_index, (cells, kind) in enumerate(
        zip(option_signatures, option_kinds, strict=True)
    ):
        option_choice_id = f"option_choice_{int(option_index + 1)}"
        option_specs.append(
            {
                "option_choice_id": str(option_choice_id),
                "option_label": str(option_label_for_index(int(option_index))),
                "mark_specs": list(_mark_specs(cells, prefix=f"option_{int(option_index + 1)}")),
                "cells": [[int(cell_x), int(cell_y)] for cell_x, cell_y in cells],
                "mark_count": int(len(cells)),
                "candidate_kind": str(kind),
                "is_correct": bool(option_index == int(correct_option_index)),
            }
        )
    return tuple(option_specs), str(answer_option_label)


def sample_overlay_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> OverlayDataset:
    """Construct one transparent-sheet overlay union puzzle dataset."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    option_count, option_count_range, option_count_probabilities = _resolve_integer_axis(
        params,
        generation_defaults,
        exact_key="option_count",
        min_key="option_count_min",
        max_key="option_count_max",
        fallback_min=4,
        fallback_max=4,
        rng=rng,
        context="overlay option-count bounds",
    )
    grid_size, grid_size_range, grid_size_probabilities = _resolve_integer_axis(
        params,
        generation_defaults,
        exact_key="grid_size",
        min_key="grid_size_min",
        max_key="grid_size_max",
        fallback_min=4,
        fallback_max=5,
        rng=rng,
        context="overlay grid-size bounds",
    )
    sheet_mark_count_min, sheet_mark_count_max = resolve_required_int_bounds(
        params,
        generation_defaults,
        min_key="sheet_mark_count_min",
        max_key="sheet_mark_count_max",
        fallback_min=2,
        fallback_max=5,
        context="overlay sheet-mark-count bounds",
    )
    overlap_count_min, overlap_count_max = resolve_required_int_bounds(
        params,
        generation_defaults,
        min_key="overlap_count_min",
        max_key="overlap_count_max",
        fallback_min=1,
        fallback_max=2,
        context="overlay overlap-count bounds",
    )
    left_cells, right_cells, overlap_cells, union_cells, source_counts = (
        _sample_overlay_sources(
            grid_size=int(grid_size),
            sheet_mark_count_min=int(sheet_mark_count_min),
            sheet_mark_count_max=int(sheet_mark_count_max),
            overlap_count_min=int(overlap_count_min),
            overlap_count_max=int(overlap_count_max),
            rng=rng,
        )
    )
    (
        correct_option_index,
        correct_index_probabilities,
        correct_option_index_sampling_mode,
    ) = resolve_correct_option_index(
        params,
        option_count=int(option_count),
        rng=rng,
    )
    option_specs, answer_option_label = _build_overlay_options(
        left_cells=left_cells,
        right_cells=right_cells,
        overlap_cells=overlap_cells,
        union_cells=union_cells,
        grid_size=int(grid_size),
        option_count=int(option_count),
        correct_option_index=int(correct_option_index),
        rng=rng,
    )
    left_count, right_count, overlap_count, union_count = source_counts
    return OverlayDataset(
        grid_size=int(grid_size),
        grid_size_range=tuple(grid_size_range),
        option_count=int(option_count),
        option_count_range=tuple(option_count_range),
        sheet_mark_count_range=(int(sheet_mark_count_min), int(sheet_mark_count_max)),
        overlap_count_range=(int(overlap_count_min), int(overlap_count_max)),
        left_cells=left_cells,
        right_cells=right_cells,
        overlap_cells=overlap_cells,
        union_cells=union_cells,
        left_mark_specs=_mark_specs(left_cells, prefix="left_sheet"),
        right_mark_specs=_mark_specs(right_cells, prefix="right_sheet"),
        left_mark_count=int(left_count),
        right_mark_count=int(right_count),
        overlap_count=int(overlap_count),
        union_mark_count=int(union_count),
        option_specs=tuple(option_specs),
        answer_option_label=str(answer_option_label),
        correct_option_index=int(correct_option_index),
        correct_option_choice_id=f"option_choice_{int(correct_option_index) + 1}",
        option_count_probabilities=dict(option_count_probabilities),
        grid_size_probabilities=dict(grid_size_probabilities),
        correct_option_index_probabilities=dict(correct_index_probabilities),
        correct_option_index_sampling_mode=str(correct_option_index_sampling_mode),
    )


def overlay_solver_trace(dataset: OverlayDataset) -> dict[str, Any]:
    """Return a compact symbolic solver trace for the sampled overlay union."""

    return {
        "grid_size": int(dataset.grid_size),
        "left_cells": [[int(x), int(y)] for x, y in dataset.left_cells],
        "right_cells": [[int(x), int(y)] for x, y in dataset.right_cells],
        "overlap_cells": [[int(x), int(y)] for x, y in dataset.overlap_cells],
        "union_cells": [[int(x), int(y)] for x, y in dataset.union_cells],
        "correct_option_index": int(dataset.correct_option_index),
        "correct_option_label": str(dataset.answer_option_label),
        "correct_option_choice_id": str(dataset.correct_option_choice_id),
        "option_signatures": {
            str(spec["option_choice_id"]): [
                [int(cell[0]), int(cell[1])]
                for cell in _option_signature(
                    tuple((int(cell[0]), int(cell[1])) for cell in spec["cells"])
                )
            ]
            for spec in dataset.option_specs
        },
    }


__all__ = ["overlay_solver_trace", "sample_overlay_dataset"]
