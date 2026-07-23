"""Passive state types for toggle-grid puzzle scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

DOMAIN = "puzzles"
SCENE_ID = "toggle_grid"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "toggle_clean",
    "toggle_notebook",
    "toggle_console",
)

GridState = Tuple[Tuple[int, ...], ...]
Cell = Tuple[int, int]
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class ResultOption:
    """One candidate output grid option for the result task."""

    option_label: str
    state: GridState
    is_correct: bool


@dataclass(frozen=True)
class SwitchOption:
    """One candidate switch cell for the repair task."""

    option_label: str
    row: int
    col: int
    is_correct: bool


@dataclass(frozen=True)
class ToggleDataset:
    """Generated toggle-grid state shared by renderer and public task files."""

    rows: int
    cols: int
    start_state: GridState
    target_state: GridState
    pressed_cells: Tuple[Cell, ...]
    result_options: Tuple[ResultOption, ...]
    switch_options: Tuple[SwitchOption, ...]
    correct_option_label: str
    scene_variant: str
    target_answer_support: Tuple[str, ...]


@dataclass(frozen=True)
class ToggleRenderParams:
    """Render dimensions and typography for one toggle-grid scene."""

    canvas_width: int
    canvas_height: int
    main_cell_size_px: int
    mini_cell_size_px: int
    panel_title_font_size_px: int
    option_font_size_px: int


@dataclass(frozen=True)
class RenderedToggleScene:
    """Rendered image plus pixel projections for answer witnesses."""

    image: object
    scene_bbox_px: BBox
    start_grid_bbox_px: BBox
    target_grid_bbox_px: BBox | None
    start_cell_bbox_map: dict[str, BBox]
    target_cell_bbox_map: dict[str, BBox]
    option_panel_bbox_map: dict[str, BBox]
