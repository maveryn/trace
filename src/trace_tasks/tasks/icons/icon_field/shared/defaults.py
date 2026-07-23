"""Scene-local defaults for icon-field tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ...shared.defaults import ICON_SHARED_DEFAULTS
from ...shared.icon_noise import default_icon_noise_value_ranges


@dataclass(frozen=True)
class IconFieldDefaults:
    """Stable fallback defaults for single-panel icon frequency fields."""

    object_count_min: int = 1
    object_count_max: int = 20
    target_count_min: int = 0
    target_count_max: int = 10
    repeated_type_count_min: int = 1
    repeated_type_count_max: int = 4
    repeated_type_multiplicity_min: int = 2
    repeated_type_multiplicity_max: int = 4
    other_repeated_type_count_max: int = 3
    canvas_width: int = ICON_SHARED_DEFAULTS.canvas_width
    canvas_height: int = ICON_SHARED_DEFAULTS.canvas_height
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    scene_icon_size_min_px: int = 64
    scene_icon_size_max_px: int = 96
    scene_max_overlap_fraction: float = 0.1
    scene_placement_max_attempts: int = ICON_SHARED_DEFAULTS.scene_placement_max_attempts
    scene_size_shrink_rounds: int = ICON_SHARED_DEFAULTS.scene_size_shrink_rounds
    scene_size_shrink_factor: float = ICON_SHARED_DEFAULTS.scene_size_shrink_factor
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    pool_manifest: str = "all_icons.txt"
    rotation_candidates_degrees: Tuple[int, ...] = (0, 90, 180, 270)
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
    icon_noise_edit_types: Tuple[str, ...] = ICON_SHARED_DEFAULTS.icon_noise_edit_types
    icon_noise_edit_count_range: Tuple[int, int] = ICON_SHARED_DEFAULTS.icon_noise_edit_count_range
    icon_noise_value_ranges: Dict[str, Dict[str, Tuple[float, float]]] = field(
        default_factory=default_icon_noise_value_ranges
    )


__all__ = ["IconFieldDefaults"]
