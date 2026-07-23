"""State contracts for the 2048 games scene."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Tuple


SCENE_ID = "2048"
SIZE = 4
EMPTY = 0
Coord = Tuple[int, int]
Board = Tuple[Tuple[int, ...], ...]

SUPPORTED_2048_SCENE_VARIANTS: Tuple[str, ...] = ("standard_board",)
SUPPORTED_2048_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "dark",
    "paper",
    "neon",
    "pastel",
)
SUPPORTED_2048_DIRECTIONS: Tuple[str, ...] = ("up", "down", "left", "right")
SUPPORTED_2048_RESULT_BOARD_LABELS: Tuple[str, ...] = tuple(chr(ord("A") + index) for index in range(6))


@dataclass(frozen=True)
class Move2048Result:
    """Fully traced result of applying one 2048 move."""

    direction: str
    before: Board
    after: Board
    merge_pairs: Tuple[Tuple[Coord, Coord], ...]
    score: int
    moved: bool
    result_sources: Mapping[Coord, Tuple[Coord, ...]]


@dataclass(frozen=True)
class Sample2048:
    """Generated 2048 board state and grounded query target."""

    scene_variant: str
    style_variant: str
    board: Board
    move_direction: str
    move_result: Move2048Result
    all_move_results: Mapping[str, Move2048Result]
    construction_mode: str
    result_option_boards: Mapping[str, Board] = field(default_factory=dict)


def coord_to_cell_id(coord: Coord) -> str:
    """Return a stable id for one board cell."""

    row, col = int(coord[0]), int(coord[1])
    return f"cell_r{row}_c{col}"


def validate_board(board: Board) -> None:
    """Validate one 4 x 4 2048 board."""

    if len(board) != SIZE or any(len(row) != SIZE for row in board):
        raise ValueError("2048 board must be 4 x 4")
    for row in board:
        for value in row:
            int_value = int(value)
            if int_value < 0:
                raise ValueError("2048 board values must be non-negative")
            if int_value != EMPTY and (int_value & (int_value - 1)) != 0:
                raise ValueError("2048 non-empty tile values must be powers of two")


def validate_2048_scene_sample(sample: Sample2048) -> None:
    """Validate query-neutral scene consistency for one generated 2048 sample."""

    validate_board(sample.board)
    if str(sample.move_direction) not in SUPPORTED_2048_DIRECTIONS:
        raise ValueError("2048 sample has unsupported move direction")
    if sample.move_result.before != sample.board:
        raise ValueError("2048 move result does not start from sample board")
    if sample.move_result.direction != sample.move_direction:
        raise ValueError("2048 move result direction does not match sample direction")
    expected_directions = set(SUPPORTED_2048_DIRECTIONS)
    if set(str(key) for key in sample.all_move_results.keys()) != expected_directions:
        raise ValueError("2048 sample must include all one-move direction results")


__all__ = [
    "Board",
    "Coord",
    "EMPTY",
    "SCENE_ID",
    "SIZE",
    "SUPPORTED_2048_DIRECTIONS",
    "SUPPORTED_2048_RESULT_BOARD_LABELS",
    "SUPPORTED_2048_SCENE_VARIANTS",
    "SUPPORTED_2048_STYLE_VARIANTS",
    "Move2048Result",
    "Sample2048",
    "coord_to_cell_id",
    "validate_2048_scene_sample",
    "validate_board",
]
