"""Scene-wide fallback defaults for chess-variant games tasks."""

from __future__ import annotations

from typing import Any


SCENE_ID = "chess_variant"

SUPPORTED_RULE_FAMILIES: tuple[str, ...] = (
    "straight_range",
    "diagonal_range",
    "straight_or_diagonal_range",
    "leaper_2_1",
    "leaper_3_1",
)
SUPPORTED_SCENE_VARIANTS: tuple[str, ...] = ("sparse_board", "crowded_board")
SUPPORTED_STYLE_VARIANTS: tuple[str, ...] = (
    "classic",
    "soft",
    "outlined",
    "wood_token",
    "blue_glyph",
    "monochrome_glyph",
)

FALLBACK_GENERATION_DEFAULTS: dict[str, Any] = {
    "scene_variant_weights": {"sparse_board": 1.0, "crowded_board": 1.0},
    "rule_family_weights": {
        "straight_range": 1.0,
        "diagonal_range": 1.0,
        "straight_or_diagonal_range": 1.0,
        "leaper_2_1": 1.0,
        "leaper_3_1": 1.0,
    },
    "style_variant_weights": {style: 1.0 for style in SUPPORTED_STYLE_VARIANTS},
    "balanced_scene_variant_sampling": True,
    "balanced_rule_family_sampling": True,
    "balanced_style_variant_sampling": True,
    "balanced_target_answer_sampling": True,
    "balanced_range_k_sampling": True,
    "range_k_support": (2, 3, 4),
    "queen_range_k_support": (2, 3),
    "sparse_min_occupied_count": 7,
    "sparse_max_occupied_count": 12,
    "crowded_min_occupied_count": 13,
    "crowded_max_occupied_count": 18,
}

FALLBACK_RENDERING_DEFAULTS: dict[str, Any] = {
    "canvas_width": 980,
    "canvas_height": 920,
    "panel_margin_px": 48,
    "rule_badge_height_px": 58,
    "rule_badge_width_px": 360,
    "header_gap_px": 18,
    "max_board_size_px": 760,
    "unit_size_scale_min": 0.5,
    "unit_size_scale_max": 1.0,
    "board_corner_radius_px": 24,
    "board_frame_width_px": 10,
    "piece_inset_fraction": 0.18,
    "marked_square_outline_width_px": 7,
    "rule_badge_font_size_px": 22,
    "piece_font_size_px": 78,
    "dynamic_canvas_size_enabled": True,
    "canvas_min_width_px": 560,
    "canvas_min_height_px": 560,
    "canvas_side_padding_px": 132,
    "canvas_vertical_padding_px": 92,
}

PROMPT_WIRING_KEYS: tuple[str, ...] = ("bundle_id", "scene_key", "task_key")

__all__ = [
    "FALLBACK_GENERATION_DEFAULTS",
    "FALLBACK_RENDERING_DEFAULTS",
    "PROMPT_WIRING_KEYS",
    "SCENE_ID",
    "SUPPORTED_RULE_FAMILIES",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_STYLE_VARIANTS",
]
