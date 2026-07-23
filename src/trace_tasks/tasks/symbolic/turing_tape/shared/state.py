"""State containers for the symbolic Turing tape scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


SCENE_ID = "turing_tape"
SCENE_VARIANTS: Tuple[str, ...] = ("clean_grid", "lab_panel", "notebook_grid")
TURING_SYMBOLS: Tuple[str, ...] = ("0", "1", "2")
TURING_MOVES: Tuple[str, str] = ("L", "R")


@dataclass(frozen=True)
class TuringTransition:
    state: str
    read_symbol: str
    write_symbol: str
    move: str
    next_state: str


@dataclass(frozen=True)
class TuringTrace:
    step: int
    state: str
    head_position: int
    read_symbol: str
    write_symbol: str
    move: str
    next_state: str


@dataclass(frozen=True)
class TuringDataset:
    tape_length: int
    symbol_count: int
    symbols: Tuple[str, ...]
    query_symbol: str
    steps: int
    states: Tuple[str, ...]
    start_state: str
    start_head: int
    initial_tape: Tuple[str, ...]
    final_tape: Tuple[str, ...]
    transitions: Tuple[TuringTransition, ...]
    traces: Tuple[TuringTrace, ...]
    answer_count: int


@dataclass(frozen=True)
class TuringRenderParams:
    canvas_width: int
    canvas_height: int
    cell_size_px: int
    grid_gap_px: int
    panel_padding_px: int
    panel_corner_radius_px: int
    panel_border_width_px: int
    grid_line_width_px: int
    option_card_width_px: int
    option_card_height_px: int
    option_gap_px: int
    option_grid_cell_px: int
    label_font_size_px: int
    small_font_size_px: int
    arrow_width_px: int
    unit_size_jitter: Dict[str, Any]
    layout_seed: int
    font_family: str


@dataclass(frozen=True)
class RenderedTuringScene:
    image: Image.Image
    scene_bbox_px: Tuple[int, int, int, int]
    item_bboxes: Dict[str, Tuple[int, int, int, int]]
    entities: Tuple[Dict[str, Any], ...]
    layout_jitter: Dict[str, Any]
    style_metadata: Dict[str, Any]
