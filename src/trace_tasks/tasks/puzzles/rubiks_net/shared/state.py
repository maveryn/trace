"""Passive constants and records for Rubik cube-net scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image

Face = str
StickerKey = tuple[Face, int, int]
Vector3 = tuple[int, int, int]
RGB = tuple[int, int, int]

DOMAIN = "puzzles"
SCENE_ID = "rubiks_net"
PROMPT_BUNDLE_ID = "puzzles_rubiks_net_v1"
PROMPT_SCENE_KEY = "rubiks_net"

FACE_ORDER: tuple[str, ...] = ("U", "D", "L", "R", "F", "B")
FACE_DISPLAY_NAMES: Mapping[str, str] = {
    "U": "Upper",
    "D": "Down",
    "L": "Left",
    "R": "Right",
    "F": "Front",
    "B": "Back",
}
MOVE_FACES: tuple[str, ...] = ("U", "D", "L", "R", "F", "B")
OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")
SUPPORTED_SCENE_VARIANTS: tuple[str, ...] = (
    "classic_net",
    "paper_net",
    "cool_net",
)

FACE_LAYOUT: Mapping[str, tuple[int, int]] = {
    "U": (1, 0),
    "L": (0, 1),
    "F": (1, 1),
    "R": (2, 1),
    "B": (3, 1),
    "D": (1, 2),
}
FACE_ORIENTATIONS: Mapping[str, tuple[Vector3, Vector3, Vector3]] = {
    "F": ((0, 0, 1), (1, 0, 0), (0, 1, 0)),
    "B": ((0, 0, -1), (-1, 0, 0), (0, 1, 0)),
    "R": ((1, 0, 0), (0, 0, -1), (0, 1, 0)),
    "L": ((-1, 0, 0), (0, 0, 1), (0, 1, 0)),
    "U": ((0, 1, 0), (1, 0, 0), (0, 0, -1)),
    "D": ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
}
NORMAL_TO_FACE: Mapping[Vector3, str] = {
    tuple(value[0]): str(face) for face, value in FACE_ORIENTATIONS.items()
}


@dataclass(frozen=True)
class RubiksRenderParams:
    """Pixel-space layout and style parameters for one Rubik net panel."""

    canvas_width: int = 1480
    canvas_height: int = 940
    scene_margin_left_px: int = 58
    scene_margin_top_px: int = 52
    main_cell_size_px: int = 44
    face_gap_px: int = 0
    net_panel_padding_px: int = 22
    panel_corner_radius_px: int = 26
    option_panel_width_px: int = 136
    option_panel_height_px: int = 154
    option_gap_px: int = 18
    option_row_gap_px: int = 18
    result_option_panel_width_px: int = 198
    result_option_panel_height_px: int = 188
    result_option_gap_px: int = 18
    result_option_row_gap_px: int = 18
    swatch_size_px: int = 78
    border_width_px: int = 3
    sticker_gap_px: int = 2
    option_label_font_size_px: int = 29
    face_label_font_size_px: int = 22
    small_label_font_size_px: int = 18
    number_font_size_px: int = 40
    panel_fill_rgb: RGB = (248, 249, 252)
    net_panel_fill_rgb: RGB = (252, 252, 255)
    option_panel_fill_rgb: RGB = (251, 251, 255)
    target_swatch_panel_fill_rgb: RGB = (248, 249, 252)
    sticker_outline_rgb: RGB = (52, 58, 68)
    border_color_rgb: RGB = (86, 94, 108)
    text_color_rgb: RGB = (30, 34, 40)
    text_stroke_rgb: RGB = (255, 255, 255)
    coordinate_fill_rgb: RGB = (255, 246, 248)
    coordinate_grid_rgb: RGB = (176, 124, 138)
    unit_size_jitter: dict[str, Any] | None = None


@dataclass(frozen=True)
class RenderedRubiksScene:
    """Rendered image plus traced geometry for a Rubik puzzle instance."""

    image: Image.Image
    entities: list[dict[str, Any]]
    scene_bbox_px: list[float]
    net_panel_bbox_px: list[float]
    net_bbox_px: list[float]
    target_swatch_bbox_px: list[float] | None
    sticker_bbox_map: dict[str, list[float]]
    option_panel_bbox_map: dict[str, list[float]]
    candidate_net_bbox_map: dict[str, list[float]]


@dataclass(frozen=True)
class RubiksAxes:
    """Resolved non-semantic generation axes for one Rubik task instance."""

    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    option_count: int
    answer_option_index: int
    answer_option_probabilities: dict[str, float]


__all__ = [
    "DOMAIN",
    "FACE_DISPLAY_NAMES",
    "FACE_LAYOUT",
    "FACE_ORDER",
    "FACE_ORIENTATIONS",
    "MOVE_FACES",
    "NORMAL_TO_FACE",
    "OPTION_LABELS",
    "PROMPT_BUNDLE_ID",
    "PROMPT_SCENE_KEY",
    "RGB",
    "RenderedRubiksScene",
    "RubiksAxes",
    "RubiksRenderParams",
    "SCENE_ID",
    "SUPPORTED_SCENE_VARIANTS",
    "StickerKey",
    "Vector3",
]
