"""Brick-breaker scene defaults and prompt wiring constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


SCENE_ID = "brick_breaker"
FALLBACK_PROMPT_WIRING_KEYS: Tuple[str, ...] = ("bundle_id", "scene_key", "task_key")


@dataclass(frozen=True)
class BrickBreakerRenderFallbacks:
    """Stable fallback rendering defaults for Brick-breaker playfields."""

    canvas_width: int = 980
    canvas_height: int = 740
    panel_margin_px: int = 38
    playfield_width_px: int = 860
    playfield_height_px: int = 640
    playfield_border_width_px: int = 5
    brick_wall_top_px: int = 46
    brick_wall_height_px: int = 270
    brick_gap_px: int = 8
    lane_pad_height_px: int = 42
    lane_pad_gap_px: int = 8
    ball_radius_px: int = 16
    path_width_px: int = 5
    label_font_size_px: int = 24
    dynamic_canvas_size_enabled: bool = True
    canvas_min_width_px: int = 560
    canvas_min_height_px: int = 440
    canvas_side_padding_px: int = 120
    canvas_vertical_padding_px: int = 78


RENDER_FALLBACKS = BrickBreakerRenderFallbacks()


__all__ = [
    "BrickBreakerRenderFallbacks",
    "FALLBACK_PROMPT_WIRING_KEYS",
    "RENDER_FALLBACKS",
    "SCENE_ID",
]
