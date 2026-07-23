"""Passive scene state for 3D Tic-Tac-Toe boards."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from PIL import Image


DOMAIN = "games"
SCENE_ID = "tic_tac_toe_3d"
SCENE_NAMESPACE = "games.tic_tac_toe_3d"
BOARD_SIZE = 3
LAYERS: Tuple[Tuple[str, str], ...] = (("top", "top"), ("middle", "middle"), ("bottom", "bottom"))
LAYOUT_VARIANTS: Tuple[str, ...] = ("vertical_perspective_stack",)
STYLE_VARIANTS: Tuple[str, ...] = ("classic_grid", "paper_board", "arcade_blue", "mint_table", "charcoal_lines")
OPTION_LABELS: Tuple[str, ...] = tuple("ABCDEF")

Coord = Tuple[int, int, int]
Line = Tuple[Coord, Coord, Coord]
Board = Tuple[Tuple[Tuple[str, ...], ...], ...]
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class TicTacToe3DDefaults:
    """Stable fallback defaults for visible 3D Tic-Tac-Toe scenes."""

    layer_piece_count_support: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6)
    option_count_support: Tuple[int, ...] = (4,)
    canvas_width: int = 760
    canvas_height: int = 900
    canvas_min_width_px: int = 580
    canvas_min_height_px: int = 720
    canvas_side_padding_px: int = 72
    panel_margin_px: int = 38
    panel_inner_padding_px: int = 36
    cell_size_px: int = 104
    cell_gap_px: int = 0
    layer_gap_px: int = 52
    skew_x_px: int = 40
    skew_y_px: int = 20
    mark_size_px: int = 66
    option_font_size_px: int = 22


@dataclass(frozen=True)
class TicTacToe3DAxes:
    """Resolved semantic and visual axes for one scene instance."""

    layout_variant: str
    style_variant: str
    option_count: int
    answer_option_index: int
    target_layer: str
    target_answer: int
    layout_variant_probabilities: Dict[str, float]
    style_variant_probabilities: Dict[str, float]
    option_count_probabilities: Dict[str, float]
    answer_option_probabilities: Dict[str, float]
    target_layer_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class TicTacToe3DSample:
    """Constructed board state plus task-neutral witness coordinates."""

    board: Board
    answer: int | str
    answer_type: str
    target_player: str
    target_layer: str
    option_cells: Tuple[Coord, ...] = ()
    answer_cell: Coord | None = None
    support_cells: Tuple[Coord, ...] = ()
    annotation_coords: Tuple[Coord, ...] = ()
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class BoardVisualStyle:
    """Resolved nonsemantic board styling."""

    layer_fill_rgb: Tuple[int, int, int]
    cell_fill_rgb: Tuple[int, int, int]
    grid_rgb: Tuple[int, int, int]
    border_rgb: Tuple[int, int, int]
    x_rgb: Tuple[int, int, int]
    o_rgb: Tuple[int, int, int]
    option_fill_rgb: Tuple[int, int, int]
    option_outline_rgb: Tuple[int, int, int]
    option_text_rgb: Tuple[int, int, int]
    label_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class RenderedTicTacToe3DScene:
    """Rendered scene plus pixel projection maps."""

    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]


__all__ = [
    "BBox",
    "BOARD_SIZE",
    "Board",
    "BoardVisualStyle",
    "Coord",
    "DOMAIN",
    "LAYERS",
    "LAYOUT_VARIANTS",
    "Line",
    "OPTION_LABELS",
    "RenderedTicTacToe3DScene",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "STYLE_VARIANTS",
    "TicTacToe3DAxes",
    "TicTacToe3DDefaults",
    "TicTacToe3DSample",
]
