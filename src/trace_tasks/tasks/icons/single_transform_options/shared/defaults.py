"""Default parameter bundles for single-transform option icon scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ...shared.defaults import ICON_SHARED_DEFAULTS
from ...shared.icon_noise import default_icon_noise_value_ranges


@dataclass(frozen=True)
class SingleTransformOptionsDefaults:
    """Shared scene defaults for transform-result option grids."""

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
    reference_icon_size_px: int = 112
    scene_max_overlap_fraction: float = 0.05
    scene_placement_max_attempts: int = ICON_SHARED_DEFAULTS.scene_placement_max_attempts
    scene_size_shrink_rounds: int = ICON_SHARED_DEFAULTS.scene_size_shrink_rounds
    scene_size_shrink_factor: float = ICON_SHARED_DEFAULTS.scene_size_shrink_factor
    cell_padding_px: int = 10
    cell_border_rgb: Tuple[int, int, int] = (218, 223, 233)
    cell_label_color_rgb: Tuple[int, int, int] = (52, 60, 77)
    cell_label_font_size_px: int = 22
    operation_cue_font_size_px: int = 21
    operation_cue_color_rgb: Tuple[int, int, int] = (63, 73, 94)
    pool_manifest: str = "non_symmetry.txt"
    transform_check_size_px: int = 96
    palette_size_min: int = 1
    palette_size_max: int = 1
    color_channel_min: int = 24
    color_channel_max: int = 220
    min_color_distance: float = 42.0
    color_distance_space: str = "lab"
    background_color_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.background_color_rgb
    panel_fill_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_fill_rgb
    panel_border_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_border_rgb
    header_text_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.header_text_rgb
    reference_content_padding_px: int = 18
    scene_content_side_padding_px: int = 14
    scene_content_bottom_padding_px: int = 14
    scene_content_top_offset_px: int = 42
    icon_noise_edit_types: Tuple[str, ...] = ICON_SHARED_DEFAULTS.icon_noise_edit_types
    icon_noise_edit_count_range: Tuple[int, int] = ICON_SHARED_DEFAULTS.icon_noise_edit_count_range
    icon_noise_value_ranges: Dict[str, Dict[str, Tuple[float, float]]] = field(
        default_factory=default_icon_noise_value_ranges
    )


__all__ = ["SingleTransformOptionsDefaults"]
