"""Defaults and constants for the named-grid icons scene."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ...shared.defaults import ICON_SHARED_DEFAULTS
from ...shared.procedural_named_icons import PROCEDURAL_NAMED_ICON_FILL_STYLES


SCENE_ID = "named_grid"
DEFAULT_GRID_SIZE_SUPPORT: Tuple[Tuple[int, int], ...] = (
    (4, 4),
    (4, 5),
    (4, 6),
    (5, 4),
    (5, 5),
    (5, 6),
    (6, 4),
    (6, 5),
    (6, 6),
)


@dataclass(frozen=True)
class NamedGridDefaults:
    """Stable fallback defaults for named-grid rendering and sampling."""

    target_count_min: int = 1
    target_count_max: int = 5
    off_line_target_count_min: int = 1
    off_line_target_count_max: int = 4
    canvas_width: int = 880
    canvas_height: int = 680
    reference_panel_width_px: int = ICON_SHARED_DEFAULTS.reference_panel_width_px
    reference_icon_size_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    reference_icon_size_min_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    reference_icon_size_max_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    scene_icon_size_min_px: int = 48
    scene_icon_size_max_px: int = 72
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
    icon_noise_edit_types: Tuple[str, ...] = ICON_SHARED_DEFAULTS.icon_noise_edit_types
    icon_noise_edit_count_range: Tuple[int, int] = ICON_SHARED_DEFAULTS.icon_noise_edit_count_range
    icon_noise_value_ranges: Dict[str, Dict[str, Tuple[float, float]]] = field(
        default_factory=lambda: dict(ICON_SHARED_DEFAULTS.icon_noise_value_ranges)
    )
    named_icon_fill_style_support: Tuple[str, ...] = PROCEDURAL_NAMED_ICON_FILL_STYLES
    row_label_band_width_px: int = 48
    column_label_band_height_px: int = 42
    grid_label_gap_px: int = 10
    grid_cell_max_size_px: int = 104
    grid_cell_padding_px: int = 12
    grid_line_width_px: int = 2
    grid_border_width_px: int = 3
    axis_label_font_size_px: int = 24
    grid_line_rgb: Tuple[int, int, int] = (176, 187, 205)
    cell_fill_rgb: Tuple[int, int, int] = (255, 255, 255)
    alternate_cell_fill_rgb: Tuple[int, int, int] = (248, 250, 253)
    axis_label_rgb: Tuple[int, int, int] = (50, 60, 78)
