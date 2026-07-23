"""Passive state for polyomino assembly puzzles."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


DOMAIN = "puzzles"
SCENE_ID = "polyomino_assembly"
PROMPT_BUNDLE_ID = "puzzles_polyomino_assembly_v1"
PROMPT_SCENE_KEY = "polyomino_assembly"

SINGLE_QUERY_KEY = "single"
OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D")
SCENE_VARIANTS: Tuple[str, ...] = (
    "clean_table",
    "workbench_card",
    "outline_panel",
)

Cell = Tuple[int, int]
Cells = Tuple[Cell, ...]


@dataclass(frozen=True)
class PolyominoAssemblyDefaults:
    """Stable generation fallbacks for polyomino assembly scenes."""

    total_cell_count_min: int = 6
    total_cell_count_max: int = 10
    piece_cell_count_min: int = 2
    piece_cell_count_max: int = 6
    shape_bbox_max_dim: int = 5
    split_attempts: int = 500
    distractor_attempts: int = 2000


@dataclass(frozen=True)
class AssemblyRenderDefaults:
    """Stable rendering fallbacks for polyomino assembly scenes."""

    canvas_width: int = 820
    canvas_height: int = 760
    scene_margin_left_px: int = 48
    scene_margin_right_px: int = 48
    scene_margin_top_px: int = 42
    scene_margin_bottom_px: int = 42
    cell_size_px: int = 34
    cell_gap_px: int = 3
    panel_padding_px: int = 22
    top_panel_height_px: int = 226
    top_to_options_gap_px: int = 42
    option_panel_width_px: int = 324
    option_panel_height_px: int = 190
    option_gap_px: int = 28
    option_row_gap_px: int = 24
    panel_corner_radius_px: int = 18
    cell_corner_radius_px: int = 7
    border_width_px: int = 3
    option_label_font_size_px: int = 30
    source_gap_px: int = 34


@dataclass(frozen=True)
class AssemblyRenderParams:
    """Resolved rendering values for one polyomino assembly sample."""

    canvas_width: int
    canvas_height: int
    scene_margin_left_px: int
    scene_margin_right_px: int
    scene_margin_top_px: int
    scene_margin_bottom_px: int
    cell_size_px: int
    cell_gap_px: int
    panel_padding_px: int
    top_panel_height_px: int
    top_to_options_gap_px: int
    option_panel_width_px: int
    option_panel_height_px: int
    option_gap_px: int
    option_row_gap_px: int
    panel_corner_radius_px: int
    cell_corner_radius_px: int
    border_width_px: int
    option_label_font_size_px: int
    source_gap_px: int
    panel_fill_rgb: Tuple[int, int, int]
    option_panel_fill_rgb: Tuple[int, int, int]
    shape_fill_rgb: Tuple[int, int, int]
    source_shape_fill_rgb: Tuple[int, int, int]
    shape_color_name: str
    border_color_rgb: Tuple[int, int, int]
    text_color_rgb: Tuple[int, int, int]
    text_stroke_rgb: Tuple[int, int, int]
    unit_size_jitter: Dict[str, Any]


@dataclass(frozen=True)
class RenderedPolyominoAssemblyScene:
    """Rendered assembly scene with traced option geometry."""

    image: Image.Image
    entities: List[Dict[str, Any]]
    scene_bbox_px: List[float]
    item_bbox_map: Dict[str, List[float]]
    option_choice_bbox_map: Dict[str, List[float]]


DEFAULTS = PolyominoAssemblyDefaults()
RENDER_DEFAULTS = AssemblyRenderDefaults()


__all__ = [
    "AssemblyRenderDefaults",
    "AssemblyRenderParams",
    "Cell",
    "Cells",
    "DEFAULTS",
    "DOMAIN",
    "OPTION_LABELS",
    "PROMPT_BUNDLE_ID",
    "PROMPT_SCENE_KEY",
    "RENDER_DEFAULTS",
    "RenderedPolyominoAssemblyScene",
    "SCENE_ID",
    "SCENE_VARIANTS",
    "SINGLE_QUERY_KEY",
]
