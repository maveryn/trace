"""Passive state and constants for tower draughts board scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image


DOMAIN = "games"
SCENE_ID = "tower_draughts_board"
SCENE_NAMESPACE = "games.tower_draughts_board"

RED = 1
BLACK = -1
PLAYER_NAMES: Dict[int, str] = {RED: "red", BLACK: "black"}
PLAYER_SUPPORT: Tuple[str, ...] = ("red", "black")
STYLE_VARIANTS: Tuple[str, ...] = ("wood_table", "ink_board", "felt_mat", "night_tokens", "parchment")
TOP_KIND_SUPPORT: Tuple[str, ...] = ("regular", "crowned")

Coord = Tuple[int, int]


@dataclass(frozen=True)
class StackSpec:
    """One stack on a playable cell, from bottom disk to top disk."""

    coord: Coord
    disks: Tuple[int, ...]
    top_crowned: bool = False

    @property
    def owner(self) -> int:
        return int(self.disks[-1])

    @property
    def height(self) -> int:
        return int(len(self.disks))


@dataclass(frozen=True)
class TowerDraughtsDefaults:
    """Stable fallback defaults for tower draughts board scenes."""

    board_size_support: Tuple[int, ...] = (4, 5, 6)
    controlled_stack_count_support: Tuple[int, ...] = tuple(range(0, 11))
    marked_capture_count_support: Tuple[int, ...] = (0, 1, 2, 3, 4)
    stack_height_support: Tuple[int, ...] = (1, 2, 3, 4)
    min_occupied_fraction: float = 0.40
    max_occupied_fraction: float = 0.65
    crowned_top_probability: float = 0.25
    canvas_width: int = 760
    canvas_height: int = 740
    panel_margin_px: int = 50
    max_board_size_px: int = 520
    cell_size_min_px: int = 56
    cell_size_max_px: int = 80
    board_frame_width_px: int = 8
    marker_width_px: int = 5
    dynamic_canvas_size_enabled: bool = True
    canvas_min_width_px: int = 520
    canvas_min_height_px: int = 500
    canvas_side_padding_px: int = 136
    canvas_vertical_padding_px: int = 136


@dataclass(frozen=True)
class TowerDraughtsAxes:
    """Resolved semantic and visual axes for one generated board."""

    target_player: int
    marked_player: int
    top_kind: str
    style_variant: str
    board_size: int
    target_answer: int
    target_answer_support: Tuple[int, ...]
    board_size_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]
    target_player_probabilities: Dict[str, float]
    marked_player_probabilities: Dict[str, float]
    top_kind_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]


@dataclass(frozen=True)
class TowerDraughtsSample:
    """One symbolic tower draughts board plus objective witnesses."""

    board_size: int
    style_variant: str
    stacks: Tuple[StackSpec, ...]
    marked_coord: Coord | None
    target_player: int
    marked_player: int
    top_kind: str
    annotation_coords: Tuple[Coord, ...]
    answer: int
    construction_mode: str
    metadata: Dict[str, Any]


@dataclass(frozen=True)
class TowerDraughtsTheme:
    """Scene-local palette for a tower draughts board."""

    board_fill_rgb: Tuple[int, int, int]
    board_border_rgb: Tuple[int, int, int]
    light_cell_rgb: Tuple[int, int, int]
    dark_cell_rgb: Tuple[int, int, int]
    playable_outline_rgb: Tuple[int, int, int]
    red_piece_rgb: Tuple[int, int, int]
    red_piece_outline_rgb: Tuple[int, int, int]
    black_piece_rgb: Tuple[int, int, int]
    black_piece_outline_rgb: Tuple[int, int, int]
    crown_rgb: Tuple[int, int, int]
    crown_outline_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedTowerDraughtsScene:
    """Rendered board plus trace-friendly geometry maps."""

    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]


__all__ = [
    "BLACK",
    "DOMAIN",
    "PLAYER_NAMES",
    "PLAYER_SUPPORT",
    "RED",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "STYLE_VARIANTS",
    "TOP_KIND_SUPPORT",
    "Coord",
    "RenderedTowerDraughtsScene",
    "StackSpec",
    "TowerDraughtsAxes",
    "TowerDraughtsDefaults",
    "TowerDraughtsSample",
    "TowerDraughtsTheme",
]
