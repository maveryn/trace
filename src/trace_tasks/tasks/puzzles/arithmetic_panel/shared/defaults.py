"""Fallback defaults for arithmetic-constraint puzzle rendering."""

from __future__ import annotations

from typing import Any

FALLBACK_GENERATION_DEFAULTS: dict[str, Any] = {
    "answer_min": 1,
    "answer_max": 24,
    "digit_answer_min": 0,
    "digit_answer_max": 9,
    "visible_value_min": 1,
    "visible_value_max": 18,
    "balanced_scene_variant_sampling": True,
    "scene_variant_weights": {
        "constraint_sheet": 1.0,
        "constraint_card": 1.0,
        "constraint_outline": 1.0,
    },
}

FALLBACK_RENDERING_DEFAULTS: dict[str, Any] = {
    "canvas_width": 800,
    "canvas_height": 560,
    "panel_padding_px": 24,
    "panel_corner_radius_px": 20,
    "panel_border_width_px": 3,
    "cell_width_px": 78,
    "cell_height_px": 64,
    "node_radius_px": 34,
    "line_width_px": 4,
    "value_font_size_px": 32,
    "note_font_size_px": 22,
    "symbol_font_size_px": 28,
    "unit_size_scale_min": 0.5,
    "unit_size_scale_max": 1.0,
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
