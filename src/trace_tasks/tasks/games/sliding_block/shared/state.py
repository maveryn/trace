"""Passive state and constants for the sliding-block games scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "sliding_block"
DOMAIN = "games"

SCENE_VARIANTS: Tuple[str, ...] = ("wooden_tray", "cool_grid", "paper_board")
EXIT_SIDES: Tuple[str, ...] = ("right", "left", "top", "bottom")
DIRECTIONS: Tuple[str, ...] = ("up", "down", "left", "right")
OPTION_LABELS: Tuple[str, ...] = tuple("ABCDEF")

BLOCK_FILLS: Tuple[Tuple[int, int, int], ...] = (
    (72, 118, 178),
    (72, 151, 120),
    (208, 136, 76),
    (132, 103, 184),
    (214, 179, 76),
    (69, 152, 166),
    (190, 89, 108),
    (141, 119, 82),
    (96, 133, 155),
    (175, 112, 72),
    (90, 160, 130),
    (118, 114, 190),
)
TARGET_FILL = (218, 75, 66)

STYLE_COLORS: Dict[str, Dict[str, Tuple[int, int, int]]] = {
    "wooden_tray": {
        "panel_fill": (246, 238, 222),
        "board_fill": (251, 244, 230),
        "grid": (169, 135, 94),
        "border": (105, 76, 50),
        "path": (255, 236, 165),
        "exit": (74, 86, 99),
    },
    "cool_grid": {
        "panel_fill": (243, 248, 252),
        "board_fill": (250, 253, 255),
        "grid": (177, 193, 208),
        "border": (72, 91, 112),
        "path": (218, 235, 255),
        "exit": (47, 93, 155),
    },
    "paper_board": {
        "panel_fill": (250, 248, 241),
        "board_fill": (255, 253, 246),
        "grid": (198, 187, 164),
        "border": (99, 91, 75),
        "path": (244, 238, 180),
        "exit": (76, 91, 88),
    },
}


@dataclass(frozen=True)
class BlockSpec:
    """One rectangular sliding block in grid coordinates."""

    block_id: str
    label: str
    row: int
    col: int
    height: int
    width: int
    role: str
    fill_rgb: Tuple[int, int, int]

    @property
    def cells(self) -> Tuple[Tuple[int, int], ...]:
        return tuple(
            (int(self.row + row_offset), int(self.col + col_offset))
            for row_offset in range(int(self.height))
            for col_offset in range(int(self.width))
        )


@dataclass(frozen=True)
class BlockMoveSpec:
    """One legal slide in grid-cell units."""

    block_id: str
    direction: str
    distance: int


@dataclass(frozen=True)
class SlidingBlockRenderParams:
    """Resolved visual parameters for one sliding-block render."""

    canvas_width: int
    canvas_height: int
    board_size_px: int
    panel_padding_px: int
    panel_corner_radius_px: int
    block_corner_radius_px: int
    board_border_width_px: int
    grid_width_px: int
    block_gap_px: int
    target_outline_width_px: int
    label_font_size_px: int
    target_label_font_size_px: int
    arrow_width_px: int
    arrow_head_length_px: int
    arrow_head_width_px: int
    panel_fill_rgb: Tuple[int, int, int] | None
    board_fill_rgb: Tuple[int, int, int] | None
    grid_rgb: Tuple[int, int, int] | None
    border_rgb: Tuple[int, int, int] | None
    path_rgb: Tuple[int, int, int] | None
    exit_rgb: Tuple[int, int, int] | None
    text_color_rgb: Tuple[int, int, int]
    text_stroke_rgb: Tuple[int, int, int]
    unit_size_jitter: Dict[str, Any]


@dataclass(frozen=True)
class RenderedSlidingBlockScene:
    """Rendered sliding-block scene with projected geometry maps."""

    image: Image.Image
    entities: List[Dict[str, Any]]
    scene_bbox_px: List[float]
    board_bbox_px: List[float]
    path_bbox_px: List[float]
    exit_arrow_bbox_px: List[float]
    block_bbox_map: Dict[str, List[float]]
    option_panel_bbox_map: Dict[str, List[float]]


__all__ = [
    "BLOCK_FILLS",
    "DIRECTIONS",
    "DOMAIN",
    "EXIT_SIDES",
    "OPTION_LABELS",
    "SCENE_ID",
    "SCENE_VARIANTS",
    "STYLE_COLORS",
    "TARGET_FILL",
    "BlockMoveSpec",
    "BlockSpec",
    "RenderedSlidingBlockScene",
    "SlidingBlockRenderParams",
]

