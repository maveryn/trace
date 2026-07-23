"""Sudoku-grid style helpers for the puzzle scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

SUPPORTED_SUDOKU_STYLE_VARIANTS: Tuple[str, ...] = (
    "classic",
    "soft",
    "outlined",
    "notebook",
    "slate",
    "warm_paper",
)


@dataclass(frozen=True)
class SudokuTheme:
    """Resolved Sudoku-grid palette for one style variant."""

    board_fill_rgb: Tuple[int, int, int]
    board_border_rgb: Tuple[int, int, int]
    grid_line_rgb: Tuple[int, int, int]
    box_line_rgb: Tuple[int, int, int]
    cell_fill_rgb: Tuple[int, int, int]
    highlighted_cell_fill_rgba: Tuple[int, int, int, int]
    marked_cell_fill_rgba: Tuple[int, int, int, int]
    marked_cell_outline_rgb: Tuple[int, int, int]
    digit_rgb: Tuple[int, int, int]
    conflict_digit_rgb: Tuple[int, int, int]


def build_puzzle_sudoku_theme(*, style_variant: str) -> SudokuTheme:
    """Return one resolved Sudoku-grid theme for the active style variant."""

    variant = str(style_variant)
    if variant == "notebook":
        return SudokuTheme(
            board_fill_rgb=(248, 251, 255),
            board_border_rgb=(56, 82, 122),
            grid_line_rgb=(180, 204, 226),
            box_line_rgb=(62, 98, 150),
            cell_fill_rgb=(253, 255, 255),
            highlighted_cell_fill_rgba=(80, 148, 220, 54),
            marked_cell_fill_rgba=(218, 70, 82, 46),
            marked_cell_outline_rgb=(190, 48, 62),
            digit_rgb=(34, 56, 86),
            conflict_digit_rgb=(174, 42, 58),
        )
    if variant == "slate":
        return SudokuTheme(
            board_fill_rgb=(48, 58, 70),
            board_border_rgb=(20, 26, 34),
            grid_line_rgb=(106, 120, 134),
            box_line_rgb=(226, 232, 238),
            cell_fill_rgb=(58, 70, 84),
            highlighted_cell_fill_rgba=(86, 160, 236, 64),
            marked_cell_fill_rgba=(228, 78, 86, 56),
            marked_cell_outline_rgb=(245, 100, 106),
            digit_rgb=(238, 242, 246),
            conflict_digit_rgb=(255, 138, 126),
        )
    if variant == "warm_paper":
        return SudokuTheme(
            board_fill_rgb=(250, 240, 218),
            board_border_rgb=(104, 76, 48),
            grid_line_rgb=(188, 164, 132),
            box_line_rgb=(92, 66, 42),
            cell_fill_rgb=(255, 248, 232),
            highlighted_cell_fill_rgba=(74, 134, 208, 54),
            marked_cell_fill_rgba=(210, 76, 62, 46),
            marked_cell_outline_rgb=(178, 54, 44),
            digit_rgb=(54, 42, 32),
            conflict_digit_rgb=(162, 48, 42),
        )
    if variant == "soft":
        return SudokuTheme(
            board_fill_rgb=(250, 248, 241),
            board_border_rgb=(95, 100, 110),
            grid_line_rgb=(169, 174, 181),
            box_line_rgb=(72, 78, 88),
            cell_fill_rgb=(255, 253, 247),
            highlighted_cell_fill_rgba=(88, 150, 232, 56),
            marked_cell_fill_rgba=(229, 75, 75, 46),
            marked_cell_outline_rgb=(196, 46, 54),
            digit_rgb=(38, 44, 54),
            conflict_digit_rgb=(174, 44, 57),
        )
    if variant == "outlined":
        return SudokuTheme(
            board_fill_rgb=(255, 255, 255),
            board_border_rgb=(50, 58, 70),
            grid_line_rgb=(138, 145, 154),
            box_line_rgb=(42, 50, 62),
            cell_fill_rgb=(252, 253, 255),
            highlighted_cell_fill_rgba=(58, 123, 220, 62),
            marked_cell_fill_rgba=(215, 54, 64, 50),
            marked_cell_outline_rgb=(174, 34, 48),
            digit_rgb=(28, 34, 44),
            conflict_digit_rgb=(160, 34, 50),
        )
    return SudokuTheme(
        board_fill_rgb=(246, 244, 236),
        board_border_rgb=(66, 72, 82),
        grid_line_rgb=(156, 160, 166),
        box_line_rgb=(48, 54, 64),
        cell_fill_rgb=(255, 254, 249),
        highlighted_cell_fill_rgba=(70, 136, 226, 58),
        marked_cell_fill_rgba=(216, 56, 66, 48),
        marked_cell_outline_rgb=(184, 38, 50),
        digit_rgb=(34, 40, 50),
        conflict_digit_rgb=(168, 36, 50),
    )


__all__ = [
    "SUPPORTED_SUDOKU_STYLE_VARIANTS",
    "SudokuTheme",
    "build_puzzle_sudoku_theme",
]
