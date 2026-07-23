"""Sampling primitives for paper fold-cut puzzles."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import integer_range_choice, uniform_choice
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.mcq import option_label_for_index

from .constraints import resolve_correct_option_index
from .defaults import PaperFoldCutGenerationDefaults, resolve_fold_cut_generation_defaults
from .state import Cells, PaperFoldCutDataset


def _canonicalize_cells(cells: Iterable[Tuple[int, int]]) -> Cells:
    """Return cells in deterministic row-major order."""

    return tuple(sorted((int(cell_x), int(cell_y)) for cell_x, cell_y in cells))


def internal_grammar_id_for_fold_sequence(fold_sequence: Sequence[Mapping[str, Any]]) -> str:
    """Return a stable trace label for the sampled fold-count and axis grammar."""

    if len(fold_sequence) == 1:
        axis = str(fold_sequence[0]["fold_axis"])
        return f"single_{axis}_fold_cut_result"
    if len(fold_sequence) == 2:
        return "double_fold_cut_result"
    raise ValueError("paper fold-cut puzzles support one or two folds")


def fold_sequence_for_axes(
    *,
    rng,
    fold_count: int,
    fold_axis: str,
) -> tuple[Dict[str, Any], ...]:
    """Build a center-fold sequence from task-owned semantic axis choices."""

    if int(fold_count) == 1:
        if str(fold_axis) == "vertical":
            direction = str(uniform_choice(rng, ("left_to_right", "right_to_left")))
        elif str(fold_axis) == "horizontal":
            direction = str(uniform_choice(rng, ("top_to_bottom", "bottom_to_top")))
        else:
            raise ValueError(f"unsupported fold axis: {fold_axis}")
        return (
            {
                "fold_index": 1,
                "fold_axis": str(fold_axis),
                "fold_direction": str(direction),
            },
        )
    if int(fold_count) == 2:
        first_axis = str(fold_axis)
        if first_axis not in {"vertical", "horizontal"}:
            raise ValueError(f"unsupported fold axis: {fold_axis}")
        axes = (first_axis, "horizontal" if first_axis == "vertical" else "vertical")
        sequence: list[Dict[str, Any]] = []
        for index, axis in enumerate(axes, start=1):
            if str(axis) == "vertical":
                direction = str(uniform_choice(rng, ("left_to_right", "right_to_left")))
            else:
                direction = str(uniform_choice(rng, ("top_to_bottom", "bottom_to_top")))
            sequence.append(
                {
                    "fold_index": int(index),
                    "fold_axis": str(axis),
                    "fold_direction": str(direction),
                }
            )
        return tuple(sequence)
    raise ValueError("fold_count must be 1 or 2")


def folded_dimensions(
    *,
    grid_size: int,
    fold_sequence: Sequence[Mapping[str, Any]],
) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    """Return final folded dimensions and dimensions before each fold."""

    dimensions: list[tuple[int, int]] = [(int(grid_size), int(grid_size))]
    for step in fold_sequence:
        width, height = dimensions[-1]
        axis = str(step["fold_axis"])
        if axis == "vertical":
            if int(width) % 2 != 0:
                raise ValueError("vertical fold requires an even current grid width")
            dimensions.append((int(width // 2), int(height)))
        elif axis == "horizontal":
            if int(height) % 2 != 0:
                raise ValueError("horizontal fold requires an even current grid height")
            dimensions.append((int(width), int(height // 2)))
        else:
            raise ValueError(f"unsupported fold axis: {axis}")
    final_width, final_height = dimensions[-1]
    return int(final_width), int(final_height), tuple(dimensions)


def _unfold_cut_cell(
    *,
    cut_cell: tuple[int, int],
    fold_sequence: Sequence[Mapping[str, Any]],
    dimensions: Sequence[tuple[int, int]],
) -> Cells:
    """Expand one cut through the folded stack into full-sheet cells."""

    cells = {(int(cut_cell[0]), int(cut_cell[1]))}
    for step_index in reversed(range(len(fold_sequence))):
        step = fold_sequence[int(step_index)]
        previous_width, previous_height = dimensions[int(step_index)]
        axis = str(step["fold_axis"])
        direction = str(step["fold_direction"])
        expanded: set[tuple[int, int]] = set()
        for cell_x, cell_y in cells:
            if axis == "vertical":
                half = int(previous_width // 2)
                if direction == "left_to_right":
                    expanded.add((int(half + cell_x), int(cell_y)))
                    expanded.add((int((half - 1) - cell_x), int(cell_y)))
                elif direction == "right_to_left":
                    expanded.add((int(cell_x), int(cell_y)))
                    expanded.add((int((previous_width - 1) - cell_x), int(cell_y)))
                else:
                    raise ValueError(f"unsupported vertical fold direction: {direction}")
            elif axis == "horizontal":
                half = int(previous_height // 2)
                if direction == "top_to_bottom":
                    expanded.add((int(cell_x), int(half + cell_y)))
                    expanded.add((int(cell_x), int((half - 1) - cell_y)))
                elif direction == "bottom_to_top":
                    expanded.add((int(cell_x), int(cell_y)))
                    expanded.add((int(cell_x), int((previous_height - 1) - cell_y)))
                else:
                    raise ValueError(f"unsupported horizontal fold direction: {direction}")
            else:
                raise ValueError(f"unsupported fold axis: {axis}")
        cells = expanded
    return _canonicalize_cells(cells)


def unfold_cut_cells(
    *,
    cut_cells: Cells,
    fold_sequence: Sequence[Mapping[str, Any]],
    dimensions: Sequence[tuple[int, int]],
) -> Cells:
    """Expand all folded-stack cuts into the unfolded full-sheet pattern."""

    unfolded: set[tuple[int, int]] = set()
    for cut_cell in cut_cells:
        unfolded.update(
            _unfold_cut_cell(
                cut_cell=(int(cut_cell[0]), int(cut_cell[1])),
                fold_sequence=fold_sequence,
                dimensions=dimensions,
            )
        )
    return _canonicalize_cells(unfolded)


def _hole_specs(cells: Cells, *, prefix: str) -> tuple[Dict[str, Any], ...]:
    """Build deterministic hole specs from cells."""

    return tuple(
        {
            "hole_id": f"{str(prefix)}_{int(index)}",
            "cell": [int(cell_x), int(cell_y)],
        }
        for index, (cell_x, cell_y) in enumerate(cells, start=1)
    )


def _cut_specs(cells: Cells) -> tuple[Dict[str, Any], ...]:
    """Build deterministic folded-stack cut specs."""

    return tuple(
        {
            "cut_id": f"cut_{int(index)}",
            "cell": [int(cell_x), int(cell_y)],
        }
        for index, (cell_x, cell_y) in enumerate(cells, start=1)
    )


def _all_cells(cols: int, rows: int) -> Cells:
    """Return all cells for a rectangular folded packet."""

    return tuple(
        (int(cell_x), int(cell_y))
        for cell_y in range(int(rows))
        for cell_x in range(int(cols))
    )


def _int_param(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: int,
) -> int:
    """Resolve one integer param from explicit params, config, or fallback."""

    if params.get(str(key)) is not None:
        return int(params[str(key)])
    return int(group_default(defaults, str(key), int(fallback)))


def _build_fold_cut_options(
    *,
    correct_cut_cells: Cells,
    correct_hole_cells: Cells,
    folded_grid_cols: int,
    folded_grid_rows: int,
    fold_sequence: Sequence[Mapping[str, Any]],
    dimensions: Sequence[tuple[int, int]],
    option_count: int,
    correct_option_index: int,
    rng,
) -> tuple[tuple[Dict[str, Any], ...], str]:
    """Build labeled unfolded-result options with one correct pattern."""

    correct_signature = _canonicalize_cells(correct_hole_cells)
    cut_count = int(len(correct_cut_cells))
    folded_cells = list(_all_cells(int(folded_grid_cols), int(folded_grid_rows)))
    seen = {correct_signature}
    distractors: list[tuple[Cells, Cells, str]] = []

    for _attempt in range(400):
        candidate_cut_cells = _canonicalize_cells(rng.sample(folded_cells, int(cut_count)))
        candidate_holes = unfold_cut_cells(
            cut_cells=candidate_cut_cells,
            fold_sequence=fold_sequence,
            dimensions=dimensions,
        )
        candidate_signature = _canonicalize_cells(candidate_holes)
        if candidate_signature in seen:
            continue
        seen.add(candidate_signature)
        distractors.append(
            (
                candidate_cut_cells,
                candidate_signature,
                "alternate_cut_location",
            )
        )
        if len(distractors) >= int(option_count) - 1:
            break
    if len(distractors) < int(option_count) - 1:
        raise ValueError("failed to construct enough fold-cut distractors")

    option_payloads = list(distractors[: int(option_count) - 1])
    option_payloads.insert(
        int(correct_option_index),
        (correct_cut_cells, correct_signature, "correct_unfolded_result"),
    )

    option_specs: list[Dict[str, Any]] = []
    for option_index, (cut_cells, hole_cells, candidate_kind) in enumerate(option_payloads):
        option_label = str(option_label_for_index(int(option_index)))
        option_choice_id = f"option_choice_{str(option_label)}"
        option_specs.append(
            {
                "option_choice_id": str(option_choice_id),
                "option_label": str(option_label),
                "hole_specs": _hole_specs(
                    hole_cells,
                    prefix=f"{str(option_choice_id)}_hole",
                ),
                "cells": [[int(cell_x), int(cell_y)] for cell_x, cell_y in hole_cells],
                "source_cut_cells": [
                    [int(cell_x), int(cell_y)] for cell_x, cell_y in cut_cells
                ],
                "hole_count": int(len(hole_cells)),
                "candidate_kind": str(candidate_kind),
                "is_correct": bool(option_index == int(correct_option_index)),
            }
        )
    answer_option_label = str(option_label_for_index(int(correct_option_index)))
    return tuple(option_specs), f"option_choice_{str(answer_option_label)}"


def sample_paper_fold_cut_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    fold_count: int,
    fold_axis: str,
    fold_count_probabilities: Mapping[str, float],
    fold_axis_probabilities: Mapping[str, float],
    namespace: str,
) -> PaperFoldCutDataset:
    """Build one fold-cut puzzle with a unique unfolded-result option."""

    defaults: PaperFoldCutGenerationDefaults = resolve_fold_cut_generation_defaults(
        generation_defaults
    )
    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    option_count_min = _int_param(
        params,
        generation_defaults,
        "option_count_min",
        defaults.option_count_min,
    )
    option_count_max = _int_param(
        params,
        generation_defaults,
        "option_count_max",
        defaults.option_count_max,
    )
    option_count, option_count_probabilities = integer_range_choice(
        rng,
        option_count_min,
        option_count_max,
    )
    grid_size = _int_param(params, generation_defaults, "grid_size", defaults.grid_size)
    if int(grid_size) % 2 != 0:
        raise ValueError("paper fold-cut puzzles require an even grid_size")

    cut_count_min = _int_param(
        params,
        generation_defaults,
        "cut_count_min",
        defaults.cut_count_min,
    )
    cut_count_max = _int_param(
        params,
        generation_defaults,
        "cut_count_max",
        defaults.cut_count_max,
    )
    fold_sequence = fold_sequence_for_axes(
        rng=rng,
        fold_count=int(fold_count),
        fold_axis=str(fold_axis),
    )
    folded_grid_cols, folded_grid_rows, dimensions = folded_dimensions(
        grid_size=int(grid_size),
        fold_sequence=fold_sequence,
    )
    folded_cell_capacity = int(folded_grid_cols * folded_grid_rows)
    effective_cut_count_max = min(int(cut_count_max), int(folded_cell_capacity))
    effective_cut_count_min = min(int(cut_count_min), int(effective_cut_count_max))
    if int(effective_cut_count_min) <= 0:
        raise ValueError("cut_count_min must be positive")
    cut_count, cut_count_probabilities = integer_range_choice(
        rng,
        int(effective_cut_count_min),
        int(effective_cut_count_max),
    )
    folded_cells = list(_all_cells(int(folded_grid_cols), int(folded_grid_rows)))
    cut_cells = _canonicalize_cells(rng.sample(folded_cells, int(cut_count)))
    unfolded_hole_cells = unfold_cut_cells(
        cut_cells=cut_cells,
        fold_sequence=fold_sequence,
        dimensions=dimensions,
    )

    (
        correct_option_index,
        correct_option_index_probabilities,
        correct_option_index_sampling_mode,
    ) = resolve_correct_option_index(
        params,
        option_count=int(option_count),
        rng=rng,
    )

    option_specs, correct_option_choice_id = _build_fold_cut_options(
        correct_cut_cells=cut_cells,
        correct_hole_cells=unfolded_hole_cells,
        folded_grid_cols=int(folded_grid_cols),
        folded_grid_rows=int(folded_grid_rows),
        fold_sequence=fold_sequence,
        dimensions=dimensions,
        option_count=int(option_count),
        correct_option_index=int(correct_option_index),
        rng=rng,
    )
    answer_option_label = str(option_label_for_index(int(correct_option_index)))

    return PaperFoldCutDataset(
        internal_grammar_id=internal_grammar_id_for_fold_sequence(fold_sequence),
        grid_size=int(grid_size),
        folded_grid_cols=int(folded_grid_cols),
        folded_grid_rows=int(folded_grid_rows),
        fold_sequence=tuple(dict(step) for step in fold_sequence),
        fold_count=int(len(fold_sequence)),
        folded_dimensions_by_step=tuple(
            (int(width), int(height)) for width, height in dimensions
        ),
        cut_count=int(cut_count),
        cut_count_range=(int(cut_count_min), int(cut_count_max)),
        cut_cells=tuple((int(x), int(y)) for x, y in cut_cells),
        cut_specs=_cut_specs(cut_cells),
        unfolded_hole_cells=tuple((int(x), int(y)) for x, y in unfolded_hole_cells),
        unfolded_hole_specs=_hole_specs(unfolded_hole_cells, prefix="unfolded_hole"),
        unfolded_hole_count=int(len(unfolded_hole_cells)),
        option_count=int(option_count),
        option_count_range=(int(option_count_min), int(option_count_max)),
        option_specs=tuple(dict(item) for item in option_specs),
        answer_option_label=str(answer_option_label),
        correct_option_index=int(correct_option_index),
        correct_option_choice_id=str(correct_option_choice_id),
        valid_option_choice_ids=(str(correct_option_choice_id),),
        option_count_probabilities=dict(option_count_probabilities),
        fold_count_probabilities=dict(fold_count_probabilities),
        fold_axis_probabilities=dict(fold_axis_probabilities),
        cut_count_probabilities=dict(cut_count_probabilities),
        correct_option_index_probabilities=dict(correct_option_index_probabilities),
        correct_option_index_sampling_mode=str(correct_option_index_sampling_mode),
    )


def fold_cut_solver_trace(dataset: PaperFoldCutDataset) -> dict[str, Any]:
    """Return symbolic solver fields for a sampled fold-cut puzzle."""

    return {
        "internal_grammar_id": str(dataset.internal_grammar_id),
        "fold_sequence": [dict(step) for step in dataset.fold_sequence],
        "cut_cells": [[int(x), int(y)] for x, y in dataset.cut_cells],
        "unfolded_hole_cells": [
            [int(x), int(y)] for x, y in dataset.unfolded_hole_cells
        ],
        "correct_option_index": int(dataset.correct_option_index),
        "correct_option_label": str(dataset.answer_option_label),
        "correct_option_choice_id": str(dataset.correct_option_choice_id),
        "option_signatures": {
            str(spec["option_choice_id"]): [
                [int(cell[0]), int(cell[1])] for cell in spec["cells"]
            ]
            for spec in dataset.option_specs
        },
    }


__all__ = [
    "fold_sequence_for_axes",
    "folded_dimensions",
    "internal_grammar_id_for_fold_sequence",
    "sample_paper_fold_cut_dataset",
    "fold_cut_solver_trace",
    "unfold_cut_cells",
]
