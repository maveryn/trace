"""Scene-local defaults for icon-cutout tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ...shared.defaults import ICON_SHARED_DEFAULTS
from ...shared.icon_noise import default_icon_noise_value_ranges


FRAGMENT_WINDOW_STYLES: Tuple[str, ...] = ("rectangle", "rounded", "ellipse")
WINDOW_STYLE_DIFFICULTY: Dict[str, float] = {
    "rectangle": 0.25,
    "rounded": 0.40,
    "ellipse": 0.55,
}


@dataclass(frozen=True)
class IconCutoutDefaults:
    """Stable fallback defaults for partial-icon matching scenes."""

    object_count_min: int = 6
    object_count_max: int = 6
    canvas_width: int = 1104
    canvas_height: int = 640
    reference_panel_width_px: int = 296
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    scene_icon_size_min_px: int = 96
    scene_icon_size_max_px: int = 112
    reference_icon_size_px: int = 224
    scene_max_overlap_fraction: float = 0.05
    scene_placement_max_attempts: int = ICON_SHARED_DEFAULTS.scene_placement_max_attempts
    scene_size_shrink_rounds: int = ICON_SHARED_DEFAULTS.scene_size_shrink_rounds
    scene_size_shrink_factor: float = ICON_SHARED_DEFAULTS.scene_size_shrink_factor
    cell_padding_px: int = 10
    cell_border_rgb: Tuple[int, int, int] = (218, 223, 233)
    cell_label_color_rgb: Tuple[int, int, int] = (52, 60, 77)
    cell_label_font_size_px: int = 22
    fragment_frame_rgb: Tuple[int, int, int] = (78, 91, 116)
    fragment_frame_width_px: int = 3
    pool_manifest: str = "all_icons.txt"
    rotation_candidates_degrees: Tuple[int, ...] = (0,)
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
    fragment_window_styles: Tuple[str, ...] = FRAGMENT_WINDOW_STYLES
    fragment_window_width_fraction_range: Tuple[float, float] = (0.44, 0.70)
    fragment_window_height_fraction_range: Tuple[float, float] = (0.44, 0.70)
    fragment_visible_alpha_ratio_range: Tuple[float, float] = (0.34, 0.64)
    fragment_alpha_density_min: float = 0.08
    fragment_sampling_attempts: int = 240
    reference_content_padding_px: int = 18
    scene_content_side_padding_px: int = 14
    scene_content_bottom_padding_px: int = 14
    scene_content_top_offset_px: int = 42
    icon_noise_edit_types: Tuple[str, ...] = ICON_SHARED_DEFAULTS.icon_noise_edit_types
    icon_noise_edit_count_range: Tuple[int, int] = ICON_SHARED_DEFAULTS.icon_noise_edit_count_range
    icon_noise_value_ranges: Dict[str, Dict[str, Tuple[float, float]]] = field(
        default_factory=default_icon_noise_value_ranges
    )


__all__ = [
    "FRAGMENT_WINDOW_STYLES",
    "WINDOW_STYLE_DIFFICULTY",
    "IconCutoutDefaults",
]
