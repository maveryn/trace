"""Passive state for symbolic Braille-cell scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


BRAILLE_POSITIONS: tuple[int, ...] = (1, 2, 3, 4, 5, 6)
OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
WORD_OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D")
SUPPORTED_BRAILLE_SCENE_VARIANTS: tuple[str, ...] = ("clean_card", "notebook_card", "exam_scan")


@dataclass(frozen=True)
class BrailleCellSpec:
    item_id: str
    raised_positions: tuple[int, ...]
    label: str = ""
    role: str = "cell"
    marked: bool = False


@dataclass(frozen=True)
class BraillePlateSpec:
    item_id: str
    cells: tuple[BrailleCellSpec, ...]
    label: str = ""
    role: str = "plate"
    marked: bool = False


@dataclass(frozen=True)
class BrailleRenderParams:
    canvas_width: int = 980
    canvas_height: int = 680
    cell_width_px: int = 132
    cell_height_px: int = 184
    dot_radius_px: int = 13
    empty_dot_radius_px: int = 10
    cell_corner_radius_px: int = 18
    cell_border_width_px: int = 2
    marked_border_width_px: int = 5
    option_label_font_size_px: int = 28
    title_font_size_px: int = 24
    word_source_cell_width_px: int = 68
    word_source_cell_height_px: int = 98
    word_source_dot_radius_px: int = 7
    word_source_empty_dot_radius_px: int = 5
    word_option_cell_width_px: int = 34
    word_option_cell_height_px: int = 52
    word_option_dot_radius_px: int = 4
    word_option_empty_dot_radius_px: int = 3
    word_option_card_width_px: int = 280
    word_option_card_height_px: int = 154
    word_option_word_font_size_px: int = 30
    word_source_word_font_size_px: int = 42


@dataclass(frozen=True)
class RenderedBrailleScene:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    item_bboxes: dict[str, list[float]]
    dot_centers: dict[str, list[float]]
    raised_dot_centers: dict[str, list[float]]
    cell_dot_centers: dict[str, dict[str, list[float]]]
    scene_bbox_px: list[float]
    style_metadata: dict[str, Any]


__all__ = [
    "BRAILLE_POSITIONS",
    "OPTION_LABELS",
    "WORD_OPTION_LABELS",
    "SUPPORTED_BRAILLE_SCENE_VARIANTS",
    "BrailleCellSpec",
    "BraillePlateSpec",
    "BrailleRenderParams",
    "RenderedBrailleScene",
]
