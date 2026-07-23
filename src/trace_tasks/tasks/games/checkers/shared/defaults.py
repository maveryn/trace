"""Fallback defaults for the Checkers games scene."""

from __future__ import annotations

from typing import Any, Dict, Tuple


FALLBACK_GENERATION_DEFAULTS: Dict[str, Any] = {
    "scene_variant_weights": {
        "midgame_board": 1.0,
        "crowded_board": 1.0,
    },
    "style_variant_weights": {
        "classic": 1.0,
        "soft": 1.0,
        "outlined": 1.0,
        "wood_token": 1.0,
        "blue_table": 1.0,
        "charcoal": 1.0,
    },
    "balanced_scene_variant_sampling": True,
    "balanced_style_variant_sampling": True,
    "balanced_target_answer_sampling": True,
    "midgame_min_occupied_count": 8,
    "midgame_max_occupied_count": 12,
    "crowded_min_occupied_count": 13,
    "crowded_max_occupied_count": 17,
}

FALLBACK_RENDERING_DEFAULTS: Dict[str, Any] = {
    "canvas_width": 980,
    "canvas_height": 920,
    "panel_margin_px": 48,
    "player_badge_height_px": 52,
    "player_badge_width_px": 230,
    "header_gap_px": 18,
    "max_board_size_px": 780,
    "unit_size_scale_min": 0.5,
    "unit_size_scale_max": 1.0,
    "board_corner_radius_px": 26,
    "board_frame_width_px": 10,
    "piece_inset_fraction": 0.17,
    "player_badge_font_size_px": 22,
    "dynamic_canvas_size_enabled": True,
    "canvas_min_width_px": 560,
    "canvas_min_height_px": 560,
    "canvas_side_padding_px": 132,
    "canvas_vertical_padding_px": 92,
}

PROMPT_WIRING_KEYS: Tuple[str, str, str] = ("bundle_id", "scene_key", "task_key")

__all__ = [
    "FALLBACK_GENERATION_DEFAULTS",
    "FALLBACK_RENDERING_DEFAULTS",
    "PROMPT_WIRING_KEYS",
]
