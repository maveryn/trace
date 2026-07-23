"""Bowling scene defaults and prompt wiring constants."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


SCENE_ID = "bowling"
FALLBACK_PROMPT_WIRING_KEYS: Tuple[str, ...] = ("bundle_id", "scene_key", "task_key")


@dataclass(frozen=True)
class BowlingRenderFallbacks:
    """Stable fallback rendering defaults for Bowling lane scenes."""

    canvas_width: int = 1000
    canvas_height: int = 740
    panel_margin_px: int = 36
    lane_width_px: int = 760
    lane_height_px: int = 660
    lane_border_width_px: int = 6
    pin_radius_px: int = 22
    ball_radius_px: int = 24
    path_width_px: int = 5
    label_font_size_px: int = 24


RENDER_FALLBACKS = BowlingRenderFallbacks()


__all__ = ["FALLBACK_PROMPT_WIRING_KEYS", "RENDER_FALLBACKS", "SCENE_ID", "BowlingRenderFallbacks"]
