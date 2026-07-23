"""Bingo scene state contracts and intrinsic constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


BINGO_COLUMN_LABELS: Tuple[str, ...] = ("B", "I", "N", "G", "O")
BINGO_COLUMN_RANGES: Tuple[Tuple[int, int], ...] = (
    (1, 15),
    (16, 30),
    (31, 45),
    (46, 60),
    (61, 75),
)
BINGO_BOARD_SIZE = 5
SUPPORTED_BINGO_SCENE_VARIANTS: Tuple[str, ...] = ("single_card",)
SUPPORTED_BINGO_LINE_AXES: Tuple[str, ...] = ("row", "column")


@dataclass(frozen=True)
class BingoCellInstance:
    """One visible bingo cell before rendering."""

    cell_id: str
    row_index: int
    column_index: int
    column_label: str
    number: int
    is_marked: bool


@dataclass(frozen=True)
class BingoCardState:
    """One deterministic bingo-card state with numbers, marks, and derived line facts."""

    cells: Tuple[BingoCellInstance, ...]
    numbers_grid: Tuple[Tuple[int, ...], ...]
    mark_grid: Tuple[Tuple[bool, ...], ...]
    completed_row_indices: Tuple[int, ...]
    completed_column_indices: Tuple[int, ...]
    near_complete_row_indices: Tuple[int, ...] = ()
    near_complete_column_indices: Tuple[int, ...] = ()
    near_complete_gap_cell_ids: Tuple[str, ...] = ()
    called_numbers: Tuple[int, ...] = ()
    called_number_cell_ids: Tuple[str, ...] = ()
    line_sum_target_axis: str | None = None
    line_sum_target_line_index: int | None = None
    line_sum_target_cell_ids: Tuple[str, ...] = ()
    line_sum_target_value: int | None = None
    completed_line_sums: Tuple[Tuple[str, int, int], ...] = ()


def cell_id(*, row_index: int, column_index: int) -> str:
    """Return the canonical bingo cell id for one grid coordinate."""

    return f"cell_r{int(row_index)}_c{int(column_index)}"


__all__ = [
    "BINGO_BOARD_SIZE",
    "BINGO_COLUMN_LABELS",
    "BINGO_COLUMN_RANGES",
    "SUPPORTED_BINGO_LINE_AXES",
    "SUPPORTED_BINGO_SCENE_VARIANTS",
    "BingoCardState",
    "BingoCellInstance",
    "cell_id",
]
