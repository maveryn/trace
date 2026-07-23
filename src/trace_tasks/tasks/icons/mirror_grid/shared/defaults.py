"""Scene-local defaults for mirror-grid icon scenes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Tuple

from ...shared.defaults import ICON_SHARED_DEFAULTS


@dataclass(frozen=True)
class MirrorGridDefaults:
    """Stable fallback defaults for two-panel mirror-grid scenes."""

    option_count_choices: Tuple[int, ...] = (4, 6)
    canvas_width: int = 1104
    canvas_height: int = 640
    reference_panel_width_px: int = 296
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    scene_icon_size_min_px: int = ICON_SHARED_DEFAULTS.scene_icon_size_min_px
    scene_icon_size_max_px: int = ICON_SHARED_DEFAULTS.scene_icon_size_max_px
    reference_icon_size_px: int = 110
    cell_padding_px: int = 10
    cell_border_rgb: Tuple[int, int, int] = (218, 223, 233)
    cell_label_color_rgb: Tuple[int, int, int] = (52, 60, 77)
    cell_label_font_size_px: int = 22
    pool_manifest: str = "non_symmetry.txt"
    rotation_candidates_degrees: Tuple[int, ...] = (0, 90, 180, 270)
    palette_size_min: int = 4
    palette_size_max: int = 7
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
        default_factory=lambda: deepcopy(ICON_SHARED_DEFAULTS.icon_noise_value_ranges)
    )
    symmetric_icon_count_choices: Tuple[int, ...] = (2, 4, 6)
    both_axes_icon_count_choices: Tuple[int, ...] = (4,)
    nonsymmetric_icon_count_choices: Tuple[int, ...] = (2, 4, 6)
    patch_inner_margin_px: int = 8
    patch_min_gap_px: int = 6
    patch_sampling_attempts: int = 160


__all__ = ["MirrorGridDefaults"]
