"""Identity-free nonogram sampling helpers."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, List, Mapping, Sequence, Tuple

from trace_tasks.core.sampling import uniform_choice, uniform_choice_with_probabilities
from trace_tasks.core.seed import spawn_rng

from .defaults import resolve_grid_size, resolve_option_count
from .rules import (
    all_binary_lines,
    clue_for_line,
    col_clues_for_grid,
    grid_signature,
    line_matches_partial,
    row_clues_for_grid,
)
from .state import NonogramDataset, option_labels, option_panel_id


def sample_solution_grid(rows: int, cols: int, *, rng) -> List[List[int]]:
    """Sample a compact binary solution grid with readable clue rails."""

    for _attempt in range(300):
        density = rng.uniform(0.34, 0.56)
        grid = [
            [
                1 if rng.random() < float(density) else 0
                for _col in range(int(cols))
            ]
            for _row in range(int(rows))
        ]
        filled = sum(sum(row) for row in grid)
        total = int(rows) * int(cols)
        if filled < int(0.22 * total) or filled > int(0.72 * total):
            continue
        row_clues = row_clues_for_grid(grid)
        col_clues = col_clues_for_grid(grid)
        if max(len(clue) for clue in row_clues + col_clues) > 4:
            continue
        if sum(1 for clue in row_clues + col_clues if clue != [0]) < int(rows + cols - 2):
            continue
        return grid
    raise ValueError("failed to sample readable nonogram grid")


def _select_marked_row(
    grid: Sequence[Sequence[int]],
    *,
    rng,
) -> int:
    """Choose a row that has enough clue/partial information for completion."""

    row_clues = row_clues_for_grid(grid)
    candidates = [
        int(index)
        for index, clue in enumerate(row_clues)
        if clue != [0] and len(clue) <= 3 and 0 in [int(value) for value in grid[index]]
    ]
    if not candidates:
        candidates = [
            int(index)
            for index, clue in enumerate(row_clues)
            if clue != [0]
        ]
    if not candidates:
        raise ValueError("nonogram grid has no usable marked row")
    return int(uniform_choice(rng, candidates, sort_keys=True))


def _partial_line_for_completion(
    line: Sequence[int],
    *,
    rng,
) -> List[int | None]:
    """Reveal a few filled and empty cells in one target line."""

    length = len(line)
    filled_indices = [index for index, value in enumerate(line) if int(value) == 1]
    empty_indices = [index for index, value in enumerate(line) if int(value) == 0]
    partial: List[int | None] = [None for _ in range(int(length))]
    required: List[int] = []
    if filled_indices:
        required.append(int(uniform_choice(rng, filled_indices, sort_keys=True)))
    if empty_indices:
        required.append(int(uniform_choice(rng, empty_indices, sort_keys=True)))
    reveal_target = min(
        int(length) - 1,
        max(2, int(rng.randint(2, min(5, int(length))))),
    )
    available = [index for index in range(int(length)) if index not in required]
    rng.shuffle(available)
    reveal_indices = list(
        dict.fromkeys(required + available[: max(0, int(reveal_target) - len(required))])
    )
    for index in reveal_indices:
        partial[int(index)] = int(line[int(index)])
    return partial


def _selected_option_index(
    *,
    params: Mapping[str, Any],
    option_count: int,
    rng,
) -> tuple[int, dict[str, float]]:
    """Sample the correct option slot uniformly unless explicitly pinned."""

    explicit = params.get("correct_option_index")
    if explicit is not None:
        selected = int(explicit)
        if not 0 <= selected < int(option_count):
            raise ValueError("correct_option_index falls outside option count")
        probabilities = {
            str(index): (1.0 if int(index) == int(selected) else 0.0)
            for index in range(int(option_count))
        }
        return int(selected), probabilities
    selected, probabilities = uniform_choice_with_probabilities(
        rng,
        tuple(range(int(option_count))),
        sort_keys=True,
    )
    return int(selected), dict(probabilities)


def build_line_option_specs(
    *,
    line: Sequence[int],
    clue: Sequence[int],
    partial_line: Sequence[int | None],
    option_count: int,
    correct_index: int,
    rng,
) -> List[dict[str, Any]]:
    """Build labeled row-strip options with exactly one valid choice."""

    all_lines = all_binary_lines(len(line))
    correct = tuple(int(value) for value in line)
    valid_conflict = [
        candidate
        for candidate in all_lines
        if candidate != correct
        and clue_for_line(candidate) == list(clue)
        and not line_matches_partial(candidate, partial_line)
    ]
    partial_invalid = [
        candidate
        for candidate in all_lines
        if candidate != correct
        and clue_for_line(candidate) != list(clue)
        and line_matches_partial(candidate, partial_line)
    ]
    close_invalid = [
        candidate
        for candidate in all_lines
        if candidate != correct
        and clue_for_line(candidate) != list(clue)
        and not line_matches_partial(candidate, partial_line)
        and sum(int(a) != int(b) for a, b in zip(candidate, correct)) <= 3
    ]
    random_invalid = [
        candidate
        for candidate in all_lines
        if candidate != correct and candidate not in close_invalid
    ]

    distractors: List[Tuple[int, ...]] = []
    seen = {correct}
    for pool in (valid_conflict, partial_invalid, close_invalid, random_invalid):
        shuffled = list(pool)
        rng.shuffle(shuffled)
        for candidate in shuffled:
            if candidate in seen:
                continue
            seen.add(candidate)
            distractors.append(candidate)
            if len(distractors) >= int(option_count) - 1:
                break
        if len(distractors) >= int(option_count) - 1:
            break
    if len(distractors) < int(option_count) - 1:
        raise ValueError("failed to build enough nonogram line distractors")

    options = [tuple(candidate) for candidate in distractors[: int(option_count) - 1]]
    options.insert(int(correct_index), correct)
    labels = option_labels(int(option_count))
    return [
        {
            "option_panel_id": option_panel_id(labels[option_index]),
            "option_index": int(option_index),
            "option_label": str(labels[option_index]),
            "line": [int(value) for value in option_line],
            "is_correct": bool(option_index == int(correct_index)),
        }
        for option_index, option_line in enumerate(options)
    ]


def _mutate_grid(grid: Sequence[Sequence[int]], *, rng) -> List[List[int]]:
    """Return a nearby candidate grid used as a clue-violating distractor."""

    mutated = [[int(value) for value in row] for row in grid]
    rows = len(mutated)
    cols = len(mutated[0]) if rows else 0
    op = uniform_choice(
        rng,
        ("flip", "flip_two", "swap_rows", "swap_cols", "shift_row"),
        sort_keys=True,
    )
    if op == "swap_rows" and rows >= 2:
        a, b = rng.sample(range(rows), 2)
        mutated[a], mutated[b] = mutated[b], mutated[a]
    elif op == "swap_cols" and cols >= 2:
        a, b = rng.sample(range(cols), 2)
        for row in mutated:
            row[a], row[b] = row[b], row[a]
    elif op == "shift_row" and cols >= 2:
        row_index = int(rng.randrange(rows))
        shift = int(uniform_choice(rng, (-1, 1), sort_keys=True))
        row = list(mutated[row_index])
        mutated[row_index] = row[-shift:] + row[:-shift]
    else:
        flip_count = 2 if op == "flip_two" else 1
        for _ in range(int(flip_count)):
            row_index = int(rng.randrange(rows))
            col_index = int(rng.randrange(cols))
            mutated[row_index][col_index] = 1 - int(mutated[row_index][col_index])
    return mutated


def build_candidate_option_specs(
    *,
    grid: Sequence[Sequence[int]],
    option_count: int,
    correct_index: int,
    rng,
) -> List[dict[str, Any]]:
    """Build labeled full-grid options with exactly one clue-consistent grid."""

    correct = [[int(value) for value in row] for row in grid]
    target_row_clues = row_clues_for_grid(correct)
    target_col_clues = col_clues_for_grid(correct)
    seen = {grid_signature(correct)}
    distractors: List[List[List[int]]] = []
    for _attempt in range(800):
        candidate = _mutate_grid(correct, rng=rng)
        signature = grid_signature(candidate)
        if signature in seen:
            continue
        seen.add(signature)
        if (
            row_clues_for_grid(candidate) == target_row_clues
            and col_clues_for_grid(candidate) == target_col_clues
        ):
            continue
        distractors.append(candidate)
        if len(distractors) >= int(option_count) - 1:
            break
    if len(distractors) < int(option_count) - 1:
        raise ValueError("failed to build enough nonogram candidate distractors")

    options = [deepcopy(candidate) for candidate in distractors[: int(option_count) - 1]]
    options.insert(int(correct_index), deepcopy(correct))
    labels = option_labels(int(option_count))
    return [
        {
            "option_panel_id": option_panel_id(labels[option_index]),
            "option_index": int(option_index),
            "option_label": str(labels[option_index]),
            "grid": [[int(value) for value in row] for row in option_grid],
            "is_correct": bool(option_index == int(correct_index)),
        }
        for option_index, option_grid in enumerate(options)
    ]


def sample_line_completion_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> tuple[NonogramDataset, dict[str, Any]]:
    """Sample a marked-row completion dataset with one valid strip option."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    rows, cols, row_range, col_range = resolve_grid_size(
        params,
        generation_defaults,
        rng=rng,
    )
    option_count, option_count_probabilities = resolve_option_count(
        params,
        generation_defaults,
        rng=rng,
    )
    correct_index, correct_option_probabilities = _selected_option_index(
        params=params,
        option_count=int(option_count),
        rng=rng,
    )
    grid = sample_solution_grid(int(rows), int(cols), rng=rng)
    row_clues = row_clues_for_grid(grid)
    col_clues = col_clues_for_grid(grid)
    marked_row = _select_marked_row(grid, rng=rng)
    line = [int(value) for value in grid[int(marked_row)]]
    clue = clue_for_line(line)
    partial_line = _partial_line_for_completion(line, rng=rng)
    display_grid: List[List[int | None]] = [
        [None for _col in range(int(cols))]
        for _row in range(int(rows))
    ]
    for col_index, value in enumerate(partial_line):
        display_grid[int(marked_row)][int(col_index)] = value
    option_specs = build_line_option_specs(
        line=line,
        clue=clue,
        partial_line=partial_line,
        option_count=int(option_count),
        correct_index=int(correct_index),
        rng=rng,
    )
    answer_value = str(option_labels(int(option_count))[int(correct_index)])
    dataset = NonogramDataset(
        mode="line_completion",
        grid=grid,
        display_grid=display_grid,
        row_clues=row_clues,
        col_clues=col_clues,
        option_specs=option_specs,
        answer_value=str(answer_value),
        correct_option_panel_id=option_panel_id(answer_value),
        correct_option_index=int(correct_index),
        option_count=int(option_count),
        grid_rows_range=tuple(row_range),
        grid_cols_range=tuple(col_range),
        marked_axis="row",
        marked_index=int(marked_row),
        marked_clue=list(clue),
        line=list(line),
        partial_line=list(partial_line),
    )
    metadata = {
        "option_count_probabilities": dict(option_count_probabilities),
        "correct_option_index_probabilities": dict(correct_option_probabilities),
    }
    return dataset, metadata


def sample_candidate_solution_dataset(
    *,
    params: Mapping[str, Any],
    instance_seed: int,
    generation_defaults: Mapping[str, Any],
    namespace: str,
) -> tuple[NonogramDataset, dict[str, Any]]:
    """Sample a full candidate-solution selection instance."""

    rng = spawn_rng(int(instance_seed), f"{namespace}.dataset")
    rows, cols, row_range, col_range = resolve_grid_size(
        params,
        generation_defaults,
        rng=rng,
    )
    option_count, option_count_probabilities = resolve_option_count(
        params,
        generation_defaults,
        rng=rng,
    )
    correct_index, correct_option_probabilities = _selected_option_index(
        params=params,
        option_count=int(option_count),
        rng=rng,
    )
    grid = sample_solution_grid(int(rows), int(cols), rng=rng)
    option_specs = build_candidate_option_specs(
        grid=grid,
        option_count=int(option_count),
        correct_index=int(correct_index),
        rng=rng,
    )
    answer_value = str(option_labels(int(option_count))[int(correct_index)])
    dataset = NonogramDataset(
        mode="candidate_solution",
        grid=grid,
        display_grid=[
            [None for _col in range(int(cols))]
            for _row in range(int(rows))
        ],
        row_clues=row_clues_for_grid(grid),
        col_clues=col_clues_for_grid(grid),
        option_specs=option_specs,
        answer_value=str(answer_value),
        correct_option_panel_id=option_panel_id(answer_value),
        correct_option_index=int(correct_index),
        option_count=int(option_count),
        grid_rows_range=tuple(row_range),
        grid_cols_range=tuple(col_range),
    )
    metadata = {
        "option_count_probabilities": dict(option_count_probabilities),
        "correct_option_index_probabilities": dict(correct_option_probabilities),
    }
    return dataset, metadata


__all__ = [
    "build_candidate_option_specs",
    "build_line_option_specs",
    "sample_candidate_solution_dataset",
    "sample_line_completion_dataset",
    "sample_solution_grid",
]
