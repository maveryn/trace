"""Fallback defaults for the Backgammon games scene.

YAML remains the source of truth. These values are last-resort fallbacks used
by config resolvers when a key is missing.
"""

from __future__ import annotations

from typing import Any, Tuple


FALLBACK_RENDERING_DEFAULTS: dict[str, Any] = {
    "canvas_width": 1000,
    "canvas_height": 720,
    "board_width_px": 900,
    "board_height_px": 560,
    "board_margin_px": 50,
    "board_border_width_px": 5,
    "point_label_font_size_px": 24,
    "header_font_size_px": 26,
    "checker_radius_px": 21,
    "die_size_px": 44,
    "dynamic_canvas_size_enabled": True,
    "canvas_min_width_px": 560,
    "canvas_min_height_px": 420,
    "canvas_side_padding_px": 90,
    "canvas_vertical_padding_px": 80,
}

FALLBACK_PROMPT_WIRING_KEYS: Tuple[str, ...] = (
    "bundle_id",
    "scene_key",
    "task_key",
)

__all__ = [
    "FALLBACK_PROMPT_WIRING_KEYS",
    "FALLBACK_RENDERING_DEFAULTS",
]
