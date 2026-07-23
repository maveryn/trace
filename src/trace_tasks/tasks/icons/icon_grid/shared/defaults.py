"""Scene-local defaults for icon-grid tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ...shared.defaults import ICON_SHARED_DEFAULTS
from ...shared.icon_noise import default_icon_noise_value_ranges


@dataclass(frozen=True)
class IconGridDefaults:
    """Stable fallback defaults for visible icon grids."""

    object_count_min: int = 6
    object_count_max: int = 12
    target_count_min: int = 1
    target_count_max: int = 5
    canvas_width: int = 860
    canvas_height: int = 660
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = 22
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    pool_manifest: str = "all_icons.txt"
    scene_icon_size_min_px: int = 56
    scene_icon_size_max_px: int = 86
    grid_cell_max_size_px: int = 112
    grid_cell_padding_px: int = 10
    grid_line_width_px: int = 2
    grid_border_width_px: int = 3
    grid_line_rgb: Tuple[int, int, int] = (170, 182, 202)
    cell_fill_rgb: Tuple[int, int, int] = (255, 255, 255)
    alternate_cell_fill_rgb: Tuple[int, int, int] = (248, 250, 253)
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


__all__ = ["IconGridDefaults"]
