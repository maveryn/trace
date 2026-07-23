"""Scene-wide fallback defaults for Connect Four games tasks."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.games.shared.style import SUPPORTED_CONNECT_FOUR_STYLE_VARIANTS

from .rules import COLUMNS, ROWS


SCENE_ID = "connect_four"
SUPPORTED_SCENE_VARIANTS: tuple[str, ...] = ("midgame_board", "crowded_board")
SUPPORTED_BOARD_SIZE_VARIANTS: tuple[str, ...] = ("standard_7x6", "small_6x5")
DEFAULT_SAFE_BOARD_SIZE_VARIANTS: tuple[str, ...] = ("square_5x5", "square_6x6")
SUPPORTED_SAFE_BOARD_SIZE_VARIANTS: tuple[str, ...] = (
    *DEFAULT_SAFE_BOARD_SIZE_VARIANTS,
    *SUPPORTED_BOARD_SIZE_VARIANTS,
)
SUPPORTED_WINNING_MOVE_LABEL_THREAT_KINDS: tuple[str, ...] = ("vertical_threat", "horizontal_threat")
COLUMN_LABELS: tuple[str, ...] = tuple("ABCDEFG")

FALLBACK_GENERATION_DEFAULTS: dict[str, Any] = {
    "scene_variant_weights": {"midgame_board": 1.0, "crowded_board": 1.0},
    "board_size_variant_weights": {"standard_7x6": 1.0, "small_6x5": 1.0},
    "style_variant_weights": {style: 1.0 for style in SUPPORTED_CONNECT_FOUR_STYLE_VARIANTS},
    "balanced_scene_variant_sampling": True,
    "balanced_board_size_variant_sampling": True,
    "balanced_style_variant_sampling": True,
    "balanced_target_answer_sampling": True,
    "midgame_min_occupied_count": 8,
    "midgame_max_occupied_count": 16,
    "crowded_min_occupied_count": 16,
    "crowded_max_occupied_count": 24,
    "standard_board_rows": ROWS,
    "standard_board_columns": COLUMNS,
    "small_board_rows": 5,
    "small_board_columns": 6,
    "square_5_board_rows": 5,
    "square_5_board_columns": 5,
    "square_6_board_rows": 6,
    "square_6_board_columns": 6,
}

FALLBACK_RENDERING_DEFAULTS: dict[str, Any] = {
    "canvas_width": 980,
    "canvas_height": 900,
    "panel_margin_px": 48,
    "player_badge_height_px": 52,
    "player_badge_width_px": 220,
    "header_gap_px": 18,
    "max_board_width_px": 780,
    "unit_size_scale_min": 0.5,
    "unit_size_scale_max": 1.0,
    "board_corner_radius_px": 30,
    "board_frame_width_px": 16,
    "disc_inset_fraction": 0.14,
    "player_badge_font_size_px": 22,
    "marked_square_outline_width_px": 6,
    "dynamic_canvas_size_enabled": True,
    "canvas_min_width_px": 560,
    "canvas_min_height_px": 520,
    "canvas_side_padding_px": 132,
    "canvas_vertical_padding_px": 92,
}

PROMPT_WIRING_KEYS: tuple[str, ...] = ("bundle_id", "scene_key", "task_key")

__all__ = [
    "COLUMN_LABELS",
    "DEFAULT_SAFE_BOARD_SIZE_VARIANTS",
    "FALLBACK_GENERATION_DEFAULTS",
    "FALLBACK_RENDERING_DEFAULTS",
    "PROMPT_WIRING_KEYS",
    "SCENE_ID",
    "SUPPORTED_BOARD_SIZE_VARIANTS",
    "SUPPORTED_SAFE_BOARD_SIZE_VARIANTS",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_WINNING_MOVE_LABEL_THREAT_KINDS",
]
