"""Passive state for symbolic Morse-code scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D")
SUPPORTED_MORSE_SCENE_VARIANTS: tuple[str, ...] = ("clean_card", "notebook_card", "exam_scan")


@dataclass(frozen=True)
class MorseSymbolSpec:
    item_id: str
    symbol: str
    role: str = "symbol"


@dataclass(frozen=True)
class MorseLetterSpec:
    item_id: str
    letter: str
    symbols: tuple[MorseSymbolSpec, ...]
    role: str = "letter"


@dataclass(frozen=True)
class MorseWordSpec:
    item_id: str
    word: str
    letters: tuple[MorseLetterSpec, ...]
    label: str = ""
    role: str = "word_code"
    marked: bool = False


@dataclass(frozen=True)
class MorseRenderParams:
    canvas_width: int = 980
    canvas_height: int = 680
    code_symbol_dot_radius_px: int = 7
    code_symbol_dash_width_px: int = 32
    code_symbol_dash_height_px: int = 11
    code_symbol_gap_px: int = 8
    code_letter_gap_px: int = 36
    option_symbol_dot_radius_px: int = 3
    option_symbol_dash_width_px: int = 16
    option_symbol_dash_height_px: int = 6
    option_symbol_gap_px: int = 4
    option_letter_gap_px: int = 16
    source_card_height_px: int = 142
    option_card_width_px: int = 400
    option_card_height_px: int = 132
    word_option_card_width_px: int = 240
    word_option_card_height_px: int = 70
    option_label_font_size_px: int = 28
    source_word_font_size_px: int = 42
    option_word_font_size_px: int = 30
    card_corner_radius_px: int = 18
    card_border_width_px: int = 2
    marked_border_width_px: int = 5


@dataclass(frozen=True)
class RenderedMorseScene:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    item_bboxes: dict[str, list[float]]
    symbol_bboxes: dict[str, list[float]]
    scene_bbox_px: list[float]
    style_metadata: dict[str, Any]


__all__ = [
    "OPTION_LABELS",
    "SUPPORTED_MORSE_SCENE_VARIANTS",
    "MorseLetterSpec",
    "MorseRenderParams",
    "MorseSymbolSpec",
    "MorseWordSpec",
    "RenderedMorseScene",
]
