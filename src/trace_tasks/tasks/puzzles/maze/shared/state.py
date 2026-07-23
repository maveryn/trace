"""Passive state and constants for maze puzzle tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image

DOMAIN = "puzzles"
SCENE_ID = "maze"
SCENE_NAMESPACE = f"{DOMAIN}.{SCENE_ID}"
TARGET_REACHABILITY_VALUES: Tuple[str, ...] = ("reachable", "unreachable")
SCENE_VARIANTS: Tuple[str, ...] = (
    "classic_wall_maze",
    "paper_labyrinth_maze",
    "block_wall_maze",
)
EXIT_LABEL_POOL: Tuple[str, ...] = tuple("ABCDEFGH")
TARGET_REACHABILITY_DESCRIPTIONS = {
    "reachable": "reachable",
    "unreachable": "unreachable",
}

Color = Tuple[int, int, int]
BBox = Tuple[float, float, float, float]
Cell = Tuple[int, int]
CellEdge = Tuple[Cell, Cell]


@dataclass(frozen=True)
class MazeExitRenderParams:
    canvas_width: int
    canvas_height: int
    scene_margin_left_px: int
    scene_margin_right_px: int
    scene_margin_top_px: int
    scene_margin_bottom_px: int
    wall_stroke_width_px: int
    wall_stroke_width_min_px: int
    wall_stroke_width_max_px: int
    outer_wall_stroke_width_px: int
    outer_wall_stroke_width_min_px: int
    outer_wall_stroke_width_max_px: int
    exit_marker_radius_px: int
    exit_marker_shape: str
    exit_label_font_size_px: int
    start_font_size_px: int
    panel_fill_rgb: Color
    floor_fill_rgb: Color
    wall_color_rgb: Color
    border_color_rgb: Color
    text_color_rgb: Color
    text_stroke_rgb: Color
    start_fill_rgb: Color
    start_outline_rgb: Color
    exit_outline_rgb: Color
    exit_palette: Tuple[Color, ...]
    subtle_grid_rgb: Color
    unit_size_scale: float
    unit_size_jitter: Dict[str, Any]


@dataclass(frozen=True)
class RenderedMazeExitScene:
    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    scene_bbox_px: BBox
    item_bbox_map: Dict[str, BBox]
    item_point_map: Dict[str, Tuple[float, float]]
    cell_bbox_map: Dict[str, BBox]


__all__ = [
    "BBox",
    "Cell",
    "CellEdge",
    "Color",
    "DOMAIN",
    "EXIT_LABEL_POOL",
    "MazeExitRenderParams",
    "RenderedMazeExitScene",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_VARIANTS",
    "TARGET_REACHABILITY_DESCRIPTIONS",
    "TARGET_REACHABILITY_VALUES",
]
