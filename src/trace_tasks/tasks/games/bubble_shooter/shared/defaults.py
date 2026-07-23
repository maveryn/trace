"""Bubble-shooter scene defaults and prompt wiring constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


SCENE_ID = "bubble_shooter"
FALLBACK_PROMPT_WIRING_KEYS: Tuple[str, ...] = ("bundle_id", "scene_key", "task_key")


@dataclass(frozen=True)
class BubbleShooterRenderFallbacks:
    """Stable fallback rendering defaults for Bubble-shooter playfields."""

    canvas_width: int = 980
    canvas_height: int = 820
    panel_margin_px: int = 42
    playfield_width_px: int = 860
    playfield_height_px: int = 720
    playfield_border_width_px: int = 5
    board_top_px: int = 38
    board_height_px: int = 500
    bubble_gap_px: int = 2
    path_width_px: int = 5
    shooter_radius_px: int = 22
    option_radius_px: int = 17
    option_label_font_size_px: int = 22
    dynamic_canvas_size_enabled: bool = True
    canvas_min_width_px: int = 560
    canvas_min_height_px: int = 500
    canvas_side_padding_px: int = 120
    canvas_vertical_padding_px: int = 86


RENDER_FALLBACKS = BubbleShooterRenderFallbacks()


__all__ = [
    "BubbleShooterRenderFallbacks",
    "FALLBACK_PROMPT_WIRING_KEYS",
    "RENDER_FALLBACKS",
    "SCENE_ID",
]
