"""Passive Sokoban scene state and constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


DOMAIN = "games"
SCENE_ID = "sokoban"

PATH_MODE_SHORTEST = "shortest_route"
PATH_MODE_VALID = "valid_route"
PATH_MODE_BLOCKED = "blocked_route"
RELATION_MODE_NEAREST_TARGET = "nearest_target"
RELATION_MODE_NEAREST_BOX = "nearest_box"
RELATION_MODE_RANKED_PAIR = "ranked_pair_distance"
BOX_GOAL_STATUS_MODE_ON = "box_on_goal"
BOX_GOAL_STATUS_MODE_OFF = "box_off_goal"

PATH_CONTRACT_KIND = "path_options"
RELATION_CONTRACT_KIND = "relation_options"
BOX_GOAL_STATUS_CONTRACT_KIND = "matching_goal_status"
BOX_GOAL_DISTANCE_CONTRACT_KIND = "matching_goal_distance"
PUSH_STAND_CONTRACT_KIND = "push_stand_cell"

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "warehouse_classic",
    "paper_grid",
    "cool_room",
)

PATH_OPTION_COUNT_SUPPORT: Tuple[int, ...] = (4, 6)
RELATION_OPTION_COUNT_SUPPORT: Tuple[int, ...] = (4, 5, 6)
BOX_GOAL_STATUS_COUNT_SUPPORT: Tuple[int, ...] = (1, 2, 3, 4, 5)
BOX_GOAL_DISTANCE_OPTION_COUNT_SUPPORT: Tuple[int, ...] = (4, 6)
PUSH_STAND_OPTION_COUNT_SUPPORT: Tuple[int, ...] = (4,)

Cell = Tuple[int, int]
Color = Tuple[int, int, int]
BBox = Tuple[float, float, float, float]

DIRECTIONS: Dict[str, Cell] = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
DIRECTION_NAMES: Dict[str, str] = {
    "U": "up",
    "D": "down",
    "L": "left",
    "R": "right",
}


@dataclass(frozen=True)
class SokobanAxes:
    """Scene-level axes shared by Sokoban objectives."""

    scene_variant: str
    scene_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class SokobanRenderParams:
    """Resolved visual parameters for one Sokoban panel."""

    canvas_width: int
    canvas_height: int
    scene_margin_left_px: int
    scene_margin_top_px: int
    board_panel_width_px: int
    board_panel_height_px: int
    option_panel_width_px: int
    option_panel_height_px: int
    option_gap_px: int
    option_row_gap_px: int
    panel_corner_radius_px: int
    board_border_width_px: int
    grid_width_px: int
    coord_gutter_px: int
    main_cell_size_px: int
    mini_cell_size_px: int
    option_label_font_size_px: int
    cell_label_font_size_px: int
    sequence_font_size_px: int
    text_color_rgb: Color
    text_stroke_rgb: Color
    style_overrides: Dict[str, Color]
    unit_size_jitter: Dict[str, Any]


@dataclass(frozen=True)
class RenderedSokobanScene:
    """Rendered Sokoban scene with traceable geometry."""

    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    scene_bbox_px: BBox
    board_bbox_px: BBox
    option_panel_bbox_map: Dict[str, BBox]
    cell_bbox_map: Dict[str, BBox]


__all__ = [
    "BBox",
    "BOX_GOAL_STATUS_CONTRACT_KIND",
    "BOX_GOAL_STATUS_COUNT_SUPPORT",
    "BOX_GOAL_DISTANCE_CONTRACT_KIND",
    "BOX_GOAL_DISTANCE_OPTION_COUNT_SUPPORT",
    "BOX_GOAL_STATUS_MODE_OFF",
    "BOX_GOAL_STATUS_MODE_ON",
    "Cell",
    "Color",
    "DIRECTION_NAMES",
    "DIRECTIONS",
    "DOMAIN",
    "PATH_CONTRACT_KIND",
    "PATH_MODE_BLOCKED",
    "PATH_MODE_SHORTEST",
    "PATH_MODE_VALID",
    "PATH_OPTION_COUNT_SUPPORT",
    "PUSH_STAND_CONTRACT_KIND",
    "PUSH_STAND_OPTION_COUNT_SUPPORT",
    "RELATION_CONTRACT_KIND",
    "RELATION_MODE_NEAREST_BOX",
    "RELATION_MODE_NEAREST_TARGET",
    "RELATION_MODE_RANKED_PAIR",
    "RELATION_OPTION_COUNT_SUPPORT",
    "SCENE_ID",
    "SUPPORTED_SCENE_VARIANTS",
    "RenderedSokobanScene",
    "SokobanAxes",
    "SokobanRenderParams",
]
