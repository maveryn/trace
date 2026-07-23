"""Passive state and constants for nonogram puzzle tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

DOMAIN = "puzzles"
SCENE_ID = "nonogram"
SCENE_VARIANTS: tuple[str, ...] = (
    "nonogram_classic",
    "nonogram_card",
    "nonogram_blueprint",
)
OPTION_COUNTS: tuple[int, ...] = (4,)


@dataclass(frozen=True)
class NonogramDataset:
    """Scene state sampled before rendering and objective binding."""

    mode: str
    grid: List[List[int]]
    display_grid: List[List[int | None]]
    row_clues: List[List[int]]
    col_clues: List[List[int]]
    option_specs: List[Dict[str, Any]]
    answer_value: str
    correct_option_panel_id: str
    correct_option_index: int
    option_count: int
    grid_rows_range: tuple[int, int]
    grid_cols_range: tuple[int, int]
    marked_axis: str | None = None
    marked_index: int | None = None
    marked_clue: List[int] | None = None
    line: List[int] | None = None
    partial_line: List[int | None] | None = None


def option_panel_id(label: str) -> str:
    """Return the stable render id for one labeled option panel."""

    return f"option_{str(label)}"


def option_labels(option_count: int) -> tuple[str, ...]:
    """Return sequential option labels A.. for one option count."""

    if int(option_count) < 2:
        raise ValueError("nonogram option count must be at least 2")
    return tuple(chr(ord("A") + int(index)) for index in range(int(option_count)))


__all__ = [
    "DOMAIN",
    "NonogramDataset",
    "OPTION_COUNTS",
    "SCENE_ID",
    "SCENE_VARIANTS",
    "option_labels",
    "option_panel_id",
]
