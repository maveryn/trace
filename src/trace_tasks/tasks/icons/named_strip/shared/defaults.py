"""Fallback defaults for named-strip icon scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ...shared.defaults import ICON_SHARED_DEFAULTS
from ...shared.procedural_named_icons import PROCEDURAL_NAMED_ICON_FILL_STYLES


DOMAIN = "icons"
SCENE_ID = "named_strip"


@dataclass(frozen=True)
class NamedStripRenderDefaults:
    """Stable fallback rendering defaults for horizontal named-icon strips."""

    canvas_width: int = 1280
    canvas_height: int = 320
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    reference_panel_width_px: int = ICON_SHARED_DEFAULTS.reference_panel_width_px
    reference_icon_size_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    reference_icon_size_min_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    reference_icon_size_max_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
    scene_icon_size_min_px: int = 42
    scene_icon_size_max_px: int = 58
    cell_box_width_min_px: int = 58
    cell_box_width_max_px: int = 72
    cell_box_height_min_px: int = 88
    cell_box_height_max_px: int = 108
    scene_max_overlap_fraction: float = 0.0
    scene_placement_max_attempts: int = ICON_SHARED_DEFAULTS.scene_placement_max_attempts
    scene_size_shrink_rounds: int = ICON_SHARED_DEFAULTS.scene_size_shrink_rounds
    scene_size_shrink_factor: float = ICON_SHARED_DEFAULTS.scene_size_shrink_factor
    palette_size_min: int = 8
    palette_size_max: int = 12
    color_channel_min: int = 24
    color_channel_max: int = 220
    min_color_distance: float = 40.0
    color_distance_space: str = "lab"
    background_color_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.background_color_rgb
    panel_fill_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_fill_rgb
    panel_border_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_border_rgb
    header_text_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.header_text_rgb
    cell_padding_px: int = 4
    cell_icon_padding_px: int = 8
    cell_corner_radius_px: int = 10
    cell_border_rgb: Tuple[int, int, int] = (218, 223, 233)
    cell_label_font_size_px: int = 0
    cell_label_color_rgb: Tuple[int, int, int] = (52, 60, 77)
    missing_mark_font_size_px: int = 0
    missing_mark_color_rgb: Tuple[int, int, int] = (84, 96, 118)
    icon_noise_edit_types: Tuple[str, ...] = ICON_SHARED_DEFAULTS.icon_noise_edit_types
    icon_noise_edit_count_range: Tuple[int, int] = ICON_SHARED_DEFAULTS.icon_noise_edit_count_range
    icon_noise_value_ranges: Dict[str, Dict[str, Tuple[float, float]]] = field(
        default_factory=lambda: dict(ICON_SHARED_DEFAULTS.icon_noise_value_ranges)
    )
    named_icon_fill_style_support: Tuple[str, ...] = PROCEDURAL_NAMED_ICON_FILL_STYLES


DEFAULT_RENDER = NamedStripRenderDefaults()


__all__ = ["DEFAULT_RENDER", "DOMAIN", "SCENE_ID", "NamedStripRenderDefaults"]
