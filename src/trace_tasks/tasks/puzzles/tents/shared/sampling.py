"""Semantic board construction for Tents puzzle tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import (
    integer_range_choice,
    uniform_choice,
)
from trace_tasks.tasks.puzzles.shared.common import get_int_range
from trace_tasks.tasks.shared.mcq import option_label_for_index

from .defaults import resolve_grid_size
from .rules import (
    count_by_axis,
    legal_candidate_cells,
    neighbors4,
    neighbors8,
    touches_any_tent,
)
from .state import CandidateCellSpec, Cell, LabeledTentSpec, TentsSample


def option_labels(count: int) -> List[str]:
    """Return stable A, B, C option labels."""

    return [option_label_for_index(index) for index in range(int(count))]


def sample_single_legal_cell_board(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    rng,
) -> TentsSample:
    """Build a board with exactly one legal labeled cell for the marked tree."""

    rows, cols, row_range, col_range = resolve_grid_size(
        params,
        generation_defaults=generation_defaults,
        rng=rng,
    )
    option_count = 4
    labels = option_labels(int(option_count))
    correct_index, _probabilities = integer_range_choice(rng, 0, int(option_count) - 1)

    for _attempt in range(200):
        marked_tree = _sample_marked_tree(int(rows), int(cols), rng=rng)
        adjacent = neighbors4(marked_tree, int(rows), int(cols))
        if len(adjacent) != 4:
            continue
        correct_cell = tuple(uniform_choice(rng, adjacent))
        distractor_cells = [
            tuple(cell) for cell in adjacent if tuple(cell) != tuple(correct_cell)
        ]
        rng.shuffle(distractor_cells)

        ordered_cells: List[Cell] = []
        distractor_cursor = 0
        for label_index in range(int(option_count)):
            if int(label_index) == int(correct_index):
                ordered_cells.append(tuple(correct_cell))
            else:
                ordered_cells.append(tuple(distractor_cells[distractor_cursor]))
                distractor_cursor += 1

        board = _build_partial_board(
            rows=int(rows),
            cols=int(cols),
            marked_tree=tuple(marked_tree),
            candidate_cells=ordered_cells,
            valid_cells=[tuple(correct_cell)],
            generation_defaults=generation_defaults,
            params=params,
            rng=rng,
        )
        candidate_specs = tuple(
            CandidateCellSpec(
                label=str(labels[index]),
                row=int(cell[0]),
                col=int(cell[1]),
                is_correct=bool(index == int(correct_index)),
                is_legal=bool(tuple(cell) == tuple(correct_cell)),
            )
            for index, cell in enumerate(ordered_cells)
        )
        return TentsSample(
            rows=int(rows),
            cols=int(cols),
            grid_rows_range=tuple(row_range),
            grid_cols_range=tuple(col_range),
            marked_tree=tuple(marked_tree),
            candidate_specs=candidate_specs,
            labeled_tent_specs=(),
            visible_tents=tuple(board["visible_tents"]),
            tree_cells=tuple(board["tree_cells"]),
            row_clues=tuple(board["row_clues"]),
            col_clues=tuple(board["col_clues"]),
            legal_candidate_cells=tuple(board["legal_candidate_cells"]),
            option_count=int(option_count),
            target_answer_support=tuple(labels),
            construction_mode="one_legal_cell",
        )
    raise RuntimeError("failed to build single-legal-cell Tents board")


def sample_violating_tent_board(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    rng,
) -> TentsSample:
    """Build a one-board task with exactly one labeled tent lacking a tree."""

    _ = int(instance_seed)
    rows, cols, row_range, col_range = resolve_grid_size(
        params,
        generation_defaults=generation_defaults,
        rng=rng,
    )
    labels = option_labels(4)
    correct_index, _probabilities = integer_range_choice(rng, 0, len(labels) - 1)

    for _attempt in range(400):
        tent_cells = _sample_non_touching_cells(
            rows=int(rows),
            cols=int(cols),
            count=len(labels),
            rng=rng,
        )
        invalid_tent = tuple(tent_cells[int(correct_index)])
        tree_cells = _place_violation_task_trees(
            tent_cells=tent_cells,
            invalid_tent=invalid_tent,
            rows=int(rows),
            cols=int(cols),
            rng=rng,
        )
        if tree_cells is None:
            continue
        if _tent_has_adjacent_tree(
            invalid_tent,
            tree_cells,
            rows=int(rows),
            cols=int(cols),
        ):
            continue
        labeled_specs = tuple(
            LabeledTentSpec(
                label=str(labels[index]),
                row=int(cell[0]),
                col=int(cell[1]),
                is_correct=bool(index == int(correct_index)),
                violation_type=(
                    "no_adjacent_tree" if index == int(correct_index) else ""
                ),
            )
            for index, cell in enumerate(tent_cells)
        )
        if sum(1 for spec in labeled_specs if bool(spec.is_correct)) != 1:
            continue
        if any(
            not bool(spec.is_correct)
            and not _tent_has_adjacent_tree(
                spec.cell,
                tree_cells,
                rows=int(rows),
                cols=int(cols),
            )
            for spec in labeled_specs
        ):
            continue
        return TentsSample(
            rows=int(rows),
            cols=int(cols),
            grid_rows_range=tuple(row_range),
            grid_cols_range=tuple(col_range),
            marked_tree=None,
            candidate_specs=(),
            labeled_tent_specs=tuple(labeled_specs),
            visible_tents=tuple(tent_cells),
            tree_cells=tuple(tree_cells),
            row_clues=tuple(count_by_axis(tent_cells, int(rows), 0)),
            col_clues=tuple(count_by_axis(tent_cells, int(cols), 1)),
            legal_candidate_cells=tuple(),
            option_count=4,
            target_answer_support=tuple(labels),
            construction_mode="violating_tent",
        )
    raise RuntimeError("failed to build Tents violating-tent board")


def _sample_marked_tree(rows: int, cols: int, *, rng) -> Cell:
    return (
        int(rng.randint(1, max(1, int(rows) - 2))),
        int(rng.randint(1, max(1, int(cols) - 2))),
    )


def _sample_non_touching_cells(
    *,
    rows: int,
    cols: int,
    count: int,
    rng,
) -> List[Cell]:
    """Sample cells that do not touch side-by-side or diagonally."""

    for _attempt in range(200):
        cells = [
            (int(row), int(col)) for row in range(int(rows)) for col in range(int(cols))
        ]
        rng.shuffle(cells)
        selected: List[Cell] = []
        for cell in cells:
            if touches_any_tent(tuple(cell), selected, int(rows), int(cols)):
                continue
            selected.append(tuple(cell))
            if len(selected) >= int(count):
                break
        if len(selected) == int(count):
            return selected
    raise RuntimeError("failed to sample non-touching Tents cells")


def _tent_has_adjacent_tree(
    tent_cell: Cell,
    tree_cells: Sequence[Cell],
    *,
    rows: int,
    cols: int,
) -> bool:
    tree_set = {tuple(cell) for cell in tree_cells}
    return any(
        tuple(cell) in tree_set
        for cell in neighbors4(tuple(tent_cell), int(rows), int(cols))
    )


def _place_violation_task_trees(
    *,
    tent_cells: Sequence[Cell],
    invalid_tent: Cell,
    rows: int,
    cols: int,
    rng,
) -> List[Cell] | None:
    """Place one private tree for each non-violating labeled tent."""

    tent_set = {tuple(cell) for cell in tent_cells}
    invalid_neighbors = set(neighbors4(tuple(invalid_tent), int(rows), int(cols)))
    tree_cells: List[Cell] = []
    for tent_cell in tent_cells:
        if tuple(tent_cell) == tuple(invalid_tent):
            continue
        other_tents = [
            tuple(cell) for cell in tent_cells if tuple(cell) != tuple(tent_cell)
        ]
        choices = [
            tuple(cell)
            for cell in neighbors4(tuple(tent_cell), int(rows), int(cols))
            if tuple(cell) not in tent_set
            and tuple(cell) not in tree_cells
            and tuple(cell) not in invalid_neighbors
            and all(
                tuple(cell) not in set(neighbors4(other, int(rows), int(cols)))
                for other in other_tents
            )
        ]
        rng.shuffle(choices)
        if not choices:
            return None
        tree_cells.append(tuple(choices[0]))
    return tree_cells


def _candidate_blocker_positions(
    *,
    candidate_cell: Cell,
    valid_cells: Sequence[Cell],
    protected_cells: Sequence[Cell],
    visible_tents: Sequence[Cell],
    rows: int,
    cols: int,
) -> List[Cell]:
    valid_set = {tuple(cell) for cell in valid_cells}
    protected = {tuple(cell) for cell in protected_cells}
    tent_set = {tuple(cell) for cell in visible_tents}
    options: List[Cell] = []
    for cell in neighbors8(tuple(candidate_cell), int(rows), int(cols)):
        if tuple(cell) in protected or tuple(cell) in tent_set:
            continue
        if touches_any_tent(tuple(cell), tent_set, int(rows), int(cols)):
            continue
        if any(
            tuple(valid_cell) in set(neighbors8(tuple(cell), int(rows), int(cols)))
            for valid_cell in valid_set
        ):
            continue
        options.append(tuple(cell))
    return options


def _place_tree_for_tent(
    *,
    tent_cell: Cell,
    rows: int,
    cols: int,
    occupied: Sequence[Cell],
    protected_cells: Sequence[Cell],
    rng,
) -> Cell | None:
    occupied_set = {tuple(cell) for cell in occupied}
    protected_set = {tuple(cell) for cell in protected_cells}
    choices = [
        tuple(cell)
        for cell in neighbors4(tuple(tent_cell), int(rows), int(cols))
        if tuple(cell) not in occupied_set and tuple(cell) not in protected_set
    ]
    rng.shuffle(choices)
    return choices[0] if choices else None


def _place_extra_tents(
    *,
    rows: int,
    cols: int,
    visible_tents: List[Cell],
    protected_cells: Sequence[Cell],
    valid_cells: Sequence[Cell],
    count: int,
    rng,
) -> None:
    protected = {tuple(cell) for cell in protected_cells}
    valid_set = {tuple(cell) for cell in valid_cells}
    for _ in range(int(count)):
        candidates: List[Cell] = []
        for row in range(int(rows)):
            for col in range(int(cols)):
                cell = (int(row), int(col))
                if cell in protected or cell in visible_tents:
                    continue
                if touches_any_tent(cell, visible_tents, int(rows), int(cols)):
                    continue
                if any(
                    valid_cell in set(neighbors8(cell, int(rows), int(cols)))
                    for valid_cell in valid_set
                ):
                    continue
                candidates.append(cell)
        if not candidates:
            return
        rng.shuffle(candidates)
        visible_tents.append(tuple(candidates[0]))


def _build_partial_board(
    *,
    rows: int,
    cols: int,
    marked_tree: Cell,
    candidate_cells: Sequence[Cell],
    valid_cells: Sequence[Cell],
    generation_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    rng,
) -> Dict[str, Any]:
    """Construct visible tents/clues so exactly the requested candidates stay legal."""

    valid_set = {tuple(cell) for cell in valid_cells}
    candidate_set = {tuple(cell) for cell in candidate_cells}
    row_extra = [0 for _ in range(int(rows))]
    col_extra = [0 for _ in range(int(cols))]
    for row, col in valid_set:
        row_extra[int(row)] = 1
        col_extra[int(col)] = 1

    visible_tents: List[Cell] = []
    protected_cells = list(candidate_set | {tuple(marked_tree)})
    for cell in candidate_cells:
        row, col = int(cell[0]), int(cell[1])
        if tuple(cell) in valid_set:
            continue
        if tuple(cell) not in set(neighbors4(tuple(marked_tree), int(rows), int(cols))):
            continue
        if int(row_extra[row]) <= 0 or int(col_extra[col]) <= 0:
            continue
        if touches_any_tent(tuple(cell), visible_tents, int(rows), int(cols)):
            continue
        blocker_options = _candidate_blocker_positions(
            candidate_cell=tuple(cell),
            valid_cells=list(valid_set),
            protected_cells=protected_cells,
            visible_tents=visible_tents,
            rows=int(rows),
            cols=int(cols),
        )
        if not blocker_options:
            raise RuntimeError("failed to block invalid Tents candidate")
        rng.shuffle(blocker_options)
        visible_tents.append(tuple(blocker_options[0]))

    extra_low, extra_high = get_int_range(
        params,
        generation_defaults,
        min_key="background_tent_count_min",
        max_key="background_tent_count_max",
        fallback_min=2,
        fallback_max=5,
    )
    _place_extra_tents(
        rows=int(rows),
        cols=int(cols),
        visible_tents=visible_tents,
        protected_cells=protected_cells,
        valid_cells=list(valid_set),
        count=int(rng.randint(int(extra_low), int(extra_high))),
        rng=rng,
    )

    tree_cells: List[Cell] = [tuple(marked_tree)]
    occupied_for_trees = list(protected_cells) + list(visible_tents)
    for tent_cell in visible_tents:
        paired_tree = _place_tree_for_tent(
            tent_cell=tuple(tent_cell),
            rows=int(rows),
            cols=int(cols),
            occupied=occupied_for_trees + tree_cells,
            protected_cells=protected_cells,
            rng=rng,
        )
        if paired_tree is not None:
            tree_cells.append(tuple(paired_tree))
            occupied_for_trees.append(tuple(paired_tree))

    visible_row_counts = count_by_axis(visible_tents, int(rows), 0)
    visible_col_counts = count_by_axis(visible_tents, int(cols), 1)
    row_clues = [
        int(visible_row_counts[index] + row_extra[index]) for index in range(int(rows))
    ]
    col_clues = [
        int(visible_col_counts[index] + col_extra[index]) for index in range(int(cols))
    ]
    legal = legal_candidate_cells(
        marked_tree=tuple(marked_tree),
        candidate_cells=list(candidate_set),
        visible_tents=visible_tents,
        tree_cells=tree_cells,
        row_clues=row_clues,
        col_clues=col_clues,
        rows=int(rows),
        cols=int(cols),
    )
    if set(legal) != valid_set:
        raise RuntimeError(
            "partial Tents board does not expose the requested legal set"
        )
    return {
        "visible_tents": [tuple(cell) for cell in visible_tents],
        "tree_cells": [tuple(cell) for cell in tree_cells],
        "row_clues": [int(value) for value in row_clues],
        "col_clues": [int(value) for value in col_clues],
        "legal_candidate_cells": [tuple(cell) for cell in legal],
    }


__all__ = [
    "option_labels",
    "sample_single_legal_cell_board",
    "sample_violating_tent_board",
]
