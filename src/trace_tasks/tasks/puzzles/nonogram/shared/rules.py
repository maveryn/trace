"""Nonogram clue and validation rules."""

from __future__ import annotations

from itertools import product
from typing import List, Sequence, Tuple


def clue_for_line(line: Sequence[int]) -> List[int]:
    """Return run-length clues for one binary nonogram line."""

    runs: List[int] = []
    current = 0
    for value in line:
        if int(value) == 1:
            current += 1
        elif current:
            runs.append(int(current))
            current = 0
    if current:
        runs.append(int(current))
    return runs or [0]


def format_clue(clue: Sequence[int]) -> str:
    """Format one clue sequence for visible clue text."""

    return " ".join(str(int(value)) for value in clue)


def row_clues_for_grid(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    """Return all row clues for one binary grid."""

    return [clue_for_line(row) for row in grid]


def col_clues_for_grid(grid: Sequence[Sequence[int]]) -> List[List[int]]:
    """Return all column clues for one binary grid."""

    if not grid:
        return []
    rows = len(grid)
    cols = len(grid[0])
    return [
        clue_for_line([int(grid[row][col]) for row in range(rows)])
        for col in range(cols)
    ]


def all_binary_lines(length: int) -> List[Tuple[int, ...]]:
    """Return every binary line of one small nonogram length."""

    return [
        tuple(int(value) for value in candidate)
        for candidate in product((0, 1), repeat=int(length))
    ]


def line_matches_partial(
    line: Sequence[int],
    partial_line: Sequence[int | None],
) -> bool:
    """Return whether a full line agrees with visible partial cells."""

    return all(
        value is None or int(line[index]) == int(value)
        for index, value in enumerate(partial_line)
    )


def grid_signature(grid: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...]:
    """Return a hashable binary-grid signature."""

    return tuple(tuple(int(value) for value in row) for row in grid)


__all__ = [
    "all_binary_lines",
    "clue_for_line",
    "col_clues_for_grid",
    "format_clue",
    "grid_signature",
    "line_matches_partial",
    "row_clues_for_grid",
]
