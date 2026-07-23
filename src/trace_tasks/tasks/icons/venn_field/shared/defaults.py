"""Default parameters for Venn-field icon scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...shared.defaults import ICON_SHARED_DEFAULTS
from ...shared.procedural_named_icons import PROCEDURAL_NAMED_ICON_FILL_STYLES

DOMAIN = "icons"
SCENE_ID = "venn_field"
VENN_CATEGORIES: Tuple[str, ...] = ("left_only", "right_only", "both", "neither")


@dataclass(frozen=True)
class VennFieldDefaults:
    """Shared defaults for one-panel overlapping-circle icon fields."""

    object_count_min: int = 8
    object_count_max: int = 13
    target_count_min: int = 1
    target_count_max: int = 5
    target_opposite_count_min: int = 1
    target_opposite_count_max: int = 3
    canvas_width: int = 800
    canvas_height: int = 480
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    scene_icon_size_min_px: int = 40
    scene_icon_size_max_px: int = 72
    scene_max_overlap_fraction: float = 0.0
    scene_placement_max_attempts: int = 240
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    reference_panel_width_px: int = ICON_SHARED_DEFAULTS.reference_panel_width_px
    reference_icon_size_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
    palette_size_min: int = 8
    palette_size_max: int = 12
    color_channel_min: int = 24
    color_channel_max: int = 220
    min_color_distance: float = 40.0
    color_distance_space: str = "lab"
    background_color_rgb: Tuple[int, int, int] = (
        ICON_SHARED_DEFAULTS.background_color_rgb
    )
    panel_fill_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_fill_rgb
    panel_border_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_border_rgb
    header_text_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.header_text_rgb
    icon_noise_edit_types: Tuple[str, ...] = ICON_SHARED_DEFAULTS.icon_noise_edit_types
    icon_noise_edit_count_range: Tuple[int, int] = (
        ICON_SHARED_DEFAULTS.icon_noise_edit_count_range
    )
    named_icon_fill_style_support: Tuple[str, ...] = PROCEDURAL_NAMED_ICON_FILL_STYLES
    venn_boundary_margin_px: int = 12
    venn_left_fill_rgb: Tuple[int, int, int] = (90, 150, 235)
    venn_right_fill_rgb: Tuple[int, int, int] = (72, 190, 138)
    venn_left_outline_rgb: Tuple[int, int, int] = (38, 95, 190)
    venn_right_outline_rgb: Tuple[int, int, int] = (34, 135, 86)
    venn_fill_alpha: int = 54
    venn_outline_width_px: int = 3
    reference_marker_outline_rgb: Tuple[int, int, int] = (28, 32, 42)
    reference_marker_dot_rgb: Tuple[int, int, int] = (255, 255, 255)
    reference_marker_width_px: int = 4
    reference_marker_padding_px: int = 7


__all__ = ["DOMAIN", "SCENE_ID", "VENN_CATEGORIES", "VennFieldDefaults"]
