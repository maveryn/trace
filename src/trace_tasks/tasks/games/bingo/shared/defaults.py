"""Scene-level fallback defaults for Bingo games tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


SCENE_ID = "bingo"
FALLBACK_PROMPT_WIRING_KEYS: Tuple[str, ...] = ("bundle_id", "scene_key", "task_key")


@dataclass(frozen=True)
class BingoRenderFallbacks:
    """Stable code fallbacks for shared Bingo rendering grammar."""

    canvas_width: int = 1180
    canvas_height: int = 760
    card_width_px: int = 760
    card_height_px: int = 620
    card_corner_radius_px: int = 24
    panel_margin_px: int = 56
    title_font_size_px: int = 34
    title_band_height_px: int = 62
    header_font_size_px: int = 28
    header_height_px: int = 42
    grid_gap_px: int = 18
    number_font_size_px: int = 28
    cell_corner_radius_px: int = 14
    cell_gap_px: int = 10
    mark_inset_px: int = 12
    called_panel_width_px: int = 220
    called_panel_gap_px: int = 32
    called_panel_title_font_size_px: int = 26
    called_panel_number_font_size_px: int = 25
    mark_shape: str = "ellipse"
    cell_fill_pattern: str = "solid"
    dynamic_canvas_size_enabled: bool = True
    canvas_min_width_px: int = 560
    canvas_min_height_px: int = 440
    canvas_side_padding_px: int = 140
    canvas_vertical_padding_px: int = 70


RENDER_FALLBACKS = BingoRenderFallbacks()


__all__ = [
    "FALLBACK_PROMPT_WIRING_KEYS",
    "RENDER_FALLBACKS",
    "SCENE_ID",
    "BingoRenderFallbacks",
]
