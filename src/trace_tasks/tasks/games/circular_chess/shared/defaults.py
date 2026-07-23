"""Scene-wide fallback defaults for circular-chess games tasks."""

from __future__ import annotations

from typing import Any


SCENE_ID = "circular_chess"
RING_COUNT = 4
SECTOR_COUNT = 16

SUPPORTED_PIECE_KINDS: tuple[str, ...] = ("king", "queen", "rook", "bishop", "knight")
SUPPORTED_NON_KING_PIECE_KINDS: tuple[str, ...] = ("queen", "rook", "bishop", "knight")
SUPPORTED_SCENE_VARIANTS: tuple[str, ...] = ("sparse_board", "crowded_board")
SUPPORTED_STYLE_VARIANTS: tuple[str, ...] = (
    "classic_ring",
    "slate_ring",
    "parchment_ring",
    "emerald_ring",
    "monochrome_ring",
)

FALLBACK_GENERATION_DEFAULTS: dict[str, Any] = {
    "scene_variant_weights": {"sparse_board": 1.0, "crowded_board": 1.0},
    "marked_piece_kind_weights": {kind: 1.0 for kind in SUPPORTED_PIECE_KINDS},
    "marked_piece_color_weights": {"white": 1.0, "black": 1.0},
    "target_color_weights": {"white": 1.0, "black": 1.0},
    "style_variant_weights": {style: 1.0 for style in SUPPORTED_STYLE_VARIANTS},
    "balanced_scene_variant_sampling": True,
    "balanced_marked_piece_kind_sampling": True,
    "balanced_marked_piece_color_sampling": True,
    "balanced_target_color_sampling": True,
    "balanced_style_variant_sampling": True,
    "balanced_target_answer_sampling": True,
    "sparse_min_occupied_count": 8,
    "sparse_max_occupied_count": 13,
    "crowded_min_occupied_count": 14,
    "crowded_max_occupied_count": 22,
}

FALLBACK_RENDERING_DEFAULTS: dict[str, Any] = {
    "canvas_width": 900,
    "canvas_height": 900,
    "panel_margin_px": 48,
    "max_board_size_px": 700,
    "unit_size_scale_min": 0.5,
    "unit_size_scale_max": 1.0,
    "board_frame_width_px": 8,
    "cell_outline_width_px": 2,
    "piece_font_size_px": 56,
    "piece_bbox_fraction": 0.72,
    "marker_width_px": 5,
    "dynamic_canvas_size_enabled": True,
    "canvas_min_width_px": 620,
    "canvas_min_height_px": 620,
    "canvas_side_padding_px": 150,
    "canvas_vertical_padding_px": 150,
}

PROMPT_WIRING_KEYS: tuple[str, ...] = ("bundle_id", "scene_key", "task_key")

__all__ = [
    "FALLBACK_GENERATION_DEFAULTS",
    "FALLBACK_RENDERING_DEFAULTS",
    "PROMPT_WIRING_KEYS",
    "RING_COUNT",
    "SCENE_ID",
    "SECTOR_COUNT",
    "SUPPORTED_NON_KING_PIECE_KINDS",
    "SUPPORTED_PIECE_KINDS",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_STYLE_VARIANTS",
]
