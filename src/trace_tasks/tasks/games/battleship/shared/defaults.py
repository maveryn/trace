"""Fallback defaults for the Battleship games scene.

YAML remains the source of truth. These values are last-resort fallbacks used
by config resolvers when a key is missing.
"""

from __future__ import annotations

from typing import Any


FALLBACK_GENERATION_DEFAULTS: dict[str, Any] = {
    "board_size_support": (8, 9, 10),
    "min_miss_count": 7,
    "max_miss_count": 16,
}

FALLBACK_RENDERING_DEFAULTS: dict[str, Any] = {
    "canvas_width": 1100,
    "canvas_height": 820,
    "panel_margin_px": 48,
    "max_board_size_px": 650,
    "board_border_width_px": 5,
    "grid_line_width_px": 2,
    "cell_padding_px": 7,
    "fleet_panel_width_px": 310,
    "board_panel_gap_px": 34,
    "fleet_icon_cell_px": 18,
    "label_font_size_px": 22,
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
]
