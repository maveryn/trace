"""Fallback defaults for the 2048 games scene.

YAML remains the source of truth. These values are last-resort fallbacks used
by config resolvers when a key is missing.
"""

from __future__ import annotations

from typing import Any


SCENE_CONFIG_NAMESPACE = "games_2048_board"

FALLBACK_GENERATION_DEFAULTS: dict[str, Any] = {
    "scene_variant_weights": {"standard_board": 1.0},
    "style_variant_weights": {
        "classic": 1.0,
        "dark": 1.0,
        "paper": 1.0,
        "neon": 1.0,
        "pastel": 1.0,
    },
    "move_direction_weights": {
        "up": 1.0,
        "down": 1.0,
        "left": 1.0,
        "right": 1.0,
    },
    "balanced_scene_variant_sampling": True,
    "balanced_style_variant_sampling": True,
    "balanced_move_direction_sampling": True,
}

FALLBACK_RENDERING_DEFAULTS: dict[str, Any] = {
    "canvas_width": 900,
    "canvas_height": 900,
    "panel_margin_px": 64,
    "board_size_px": 560,
    "board_radius_px": 18,
    "cell_gap_px": 14,
    "cell_radius_px": 10,
    "tile_font_size_px": 46,
    "arrow_width_px": 9,
    "label_font_size_px": 24,
    "dynamic_canvas_size_enabled": True,
    "canvas_min_size_px": 520,
    "canvas_side_padding_px": 128,
    "canvas_side_padding_fraction": 0.20,
    "text_font_exclude_tags": (),
}

FALLBACK_PROMPT_WIRING_KEYS: tuple[str, ...] = (
    "bundle_id",
    "scene_key",
    "task_key",
)

__all__ = [
    "FALLBACK_GENERATION_DEFAULTS",
    "FALLBACK_PROMPT_WIRING_KEYS",
    "FALLBACK_RENDERING_DEFAULTS",
    "SCENE_CONFIG_NAMESPACE",
]
