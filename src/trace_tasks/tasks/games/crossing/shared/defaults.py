"""Scene-wide fallback defaults for lane-crossing games tasks."""

from __future__ import annotations

from typing import Any


SCENE_ID = "crossing"
SUPPORTED_CROSSING_SCENE_VARIANTS: tuple[str, ...] = ("traffic_crossing",)
SUPPORTED_CROSSING_STYLE_VARIANTS: tuple[str, ...] = (
    "day",
    "night",
    "retro",
    "paper",
    "construction",
)
START_LABELS: tuple[str, ...] = tuple(str(index + 1) for index in range(8))
VEHICLE_OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D")

FALLBACK_GENERATION_DEFAULTS: dict[str, Any] = {
    "scene_variant_weights": {"traffic_crossing": 1.0},
    "style_variant_weights": {style: 1.0 for style in SUPPORTED_CROSSING_STYLE_VARIANTS},
    "balanced_scene_variant_sampling": True,
    "balanced_style_variant_sampling": True,
    "balanced_lane_count_sampling": True,
    "balanced_row_count_sampling": True,
    "balanced_target_answer_sampling": True,
    "lane_count_support": (5, 6, 7, 8),
    "row_count_support": (5, 6, 7),
}

FALLBACK_RENDERING_DEFAULTS: dict[str, Any] = {
    "canvas_width": 1000,
    "canvas_height": 780,
    "playfield_width_px": 860,
    "playfield_height_px": 680,
    "panel_margin_px": 40,
    "border_width_px": 5,
    "safe_band_height_px": 82,
    "vehicle_width_px": 70,
    "vehicle_height_px": 42,
    "path_width_px": 5,
    "label_font_size_px": 24,
}

PROMPT_WIRING_KEYS: tuple[str, ...] = ("bundle_id", "scene_key", "task_key")

__all__ = [
    "FALLBACK_GENERATION_DEFAULTS",
    "FALLBACK_RENDERING_DEFAULTS",
    "PROMPT_WIRING_KEYS",
    "SCENE_ID",
    "START_LABELS",
    "SUPPORTED_CROSSING_SCENE_VARIANTS",
    "SUPPORTED_CROSSING_STYLE_VARIANTS",
    "VEHICLE_OPTION_LABELS",
]
