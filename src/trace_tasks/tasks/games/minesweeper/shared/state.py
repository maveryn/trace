"""Passive Minesweeper state objects and coordinate identifiers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


Coord = Tuple[int, int]


@dataclass(frozen=True)
class MinesweeperSample:
    """One generated Minesweeper board plus task-owned answer witnesses."""

    size: int
    answer: int | str
    mine_coords: Tuple[Coord, ...]
    revealed_coords: Tuple[Coord, ...]
    flagged_coords: Tuple[Coord, ...]
    hidden_coords: Tuple[Coord, ...]
    forced_mine_coords: Tuple[Coord, ...]
    forced_safe_coords: Tuple[Coord, ...]
    forcing_clue_coords: Tuple[Coord, ...]
    annotation_coords: Tuple[Coord, ...]
    target_answer: int | None
    distractor_hidden_count: int
    construction_mode: str
    candidate_option_coords: Tuple[Tuple[str, Coord], ...] = tuple()


def coord_to_cell_id(coord: Coord) -> str:
    """Return the canonical visible-cell id for one Minesweeper coordinate."""

    row, col = int(coord[0]), int(coord[1])
    return f"cell_r{row}_c{col}"


def in_bounds(coord: Coord, *, size: int) -> bool:
    """Return whether one coordinate is inside the square board."""

    row, col = int(coord[0]), int(coord[1])
    return 0 <= row < int(size) and 0 <= col < int(size)


def all_coords(*, size: int) -> Tuple[Coord, ...]:
    """Return all coordinates in row-major order."""

    return tuple((row, col) for row in range(int(size)) for col in range(int(size)))


def sorted_coords(coords) -> Tuple[Coord, ...]:
    """Return canonical sorted coordinate tuples."""

    return tuple(sorted((int(row), int(col)) for row, col in coords))


__all__ = [
    "Coord",
    "MinesweeperSample",
    "all_coords",
    "coord_to_cell_id",
    "in_bounds",
    "sorted_coords",
]
