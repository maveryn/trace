"""Passive state objects for the Sudoku puzzle scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

DOMAIN = "puzzles"
SCENE_ID = "sudoku"
SIZE = 9
BOX_SIZE = 3
DIGITS: Tuple[int, ...] = tuple(range(1, 10))
SUPPORTED_SUDOKU_SCENE_VARIANTS: Tuple[str, ...] = (
    "sparse_grid",
    "filled_grid",
)
SUPPORTED_SUDOKU_UNIT_TYPES: Tuple[str, ...] = (
    "row",
    "column",
    "box",
)

Coord = Tuple[int, int]
Board = Tuple[Tuple[int, ...], ...]


@dataclass(frozen=True)
class SudokuSample:
    """One generated Sudoku board and the task-owned witness coordinates."""

    board: Board
    solution: Board
    answer: int | str
    annotation_coords: Tuple[Coord, ...]
    marked_cell: Coord | None
    highlighted_unit_type: str | None
    highlighted_unit_index: int | None
    repeated_digit_values: Tuple[int, ...]
    missing_digit_values: Tuple[int, ...]
    option_specs: Tuple[Dict[str, Any], ...]
    correct_option_label: str | None
    target_digit: int | None
    visible_count: int
    construction_mode: str


__all__ = [
    "BOX_SIZE",
    "Board",
    "Coord",
    "DIGITS",
    "DOMAIN",
    "SCENE_ID",
    "SIZE",
    "SUPPORTED_SUDOKU_SCENE_VARIANTS",
    "SUPPORTED_SUDOKU_UNIT_TYPES",
    "SudokuSample",
]
