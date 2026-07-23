"""Default parameter bundles for pair-grid icon scenes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Tuple

from ...shared.defaults import ICON_SHARED_DEFAULTS
from ...shared.icon_transform import NON_IDENTITY_TRANSFORM_IDS


@dataclass(frozen=True)
class PairGridTaskDefaults:
    """Shared scene defaults with task-level override slots."""

    option_count: int = 6
    canvas_width: int = 1104
    canvas_height: int = 640
    reference_panel_width_px: int = 296
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    scene_icon_size_min_px: int = 40
    scene_icon_size_max_px: int = 96
    reference_icon_size_px: int = 110
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    background_color_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.background_color_rgb
    panel_fill_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_fill_rgb
    panel_border_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_border_rgb
    header_text_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.header_text_rgb
    cell_border_rgb: Tuple[int, int, int] = (218, 223, 233)
    cell_label_color_rgb: Tuple[int, int, int] = (52, 60, 77)
    arrow_color_rgb: Tuple[int, int, int] = (84, 96, 118)
    cell_padding_px: int = 10
    pair_arrow_stroke_px: int = 4
    cell_label_font_size_px: int = 22
    pool_manifest: str = "all_icons.txt"
    transform_ids: Tuple[str, ...] = NON_IDENTITY_TRANSFORM_IDS
    transform_check_size_px: int = 72
    palette_size_min: int = 5
    palette_size_max: int = 5
    color_channel_min: int = 24
    color_channel_max: int = 220
    min_color_distance: float = 42.0
    color_distance_space: str = "lab"
    icon_noise_edit_types: Tuple[str, ...] = ICON_SHARED_DEFAULTS.icon_noise_edit_types
    icon_noise_edit_count_range: Tuple[int, int] = ICON_SHARED_DEFAULTS.icon_noise_edit_count_range
    icon_noise_value_ranges: Dict[str, Dict[str, Tuple[float, float]]] = field(
        default_factory=lambda: deepcopy(ICON_SHARED_DEFAULTS.icon_noise_value_ranges)
    )


__all__ = ["PairGridTaskDefaults"]
