"""Default parameter bundle for paired-canvas icon scenes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Tuple

from ...shared.defaults import ICON_SHARED_DEFAULTS


SCENE_ID = "paired_canvas"


@dataclass(frozen=True)
class PairedCanvasDefaults:
    """Shared scene defaults with task-level override slots."""

    object_count_min: int = 5
    object_count_max: int = 10
    target_count_min: int = 1
    target_count_max: int = 5
    distractor_count_min: int = 1
    distractor_count_max: int = 5
    distractor_margin_over_target: int = 0
    canvas_width: int = 1104
    canvas_height: int = 640
    reference_panel_width_px: int = 516
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    scene_icon_size_min_px: int = 48
    scene_icon_size_max_px: int = 78
    reference_icon_size_px: int = 78
    reference_icon_size_min_px: int = 48
    reference_icon_size_max_px: int = 78
    scene_max_overlap_fraction: float = 0.04
    scene_placement_max_attempts: int = 160
    scene_size_shrink_rounds: int = ICON_SHARED_DEFAULTS.scene_size_shrink_rounds
    scene_size_shrink_factor: float = ICON_SHARED_DEFAULTS.scene_size_shrink_factor
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    pool_manifest: str = "all_icons.txt"
    palette_size_min: int = 8
    palette_size_max: int = 12
    color_channel_min: int = 24
    color_channel_max: int = 220
    min_color_distance: float = 42.0
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
    size_scale_small: float = 0.74
    size_scale_large: float = 1.18
    movement_delta_min: float = 0.22
    movement_delta_max: float = 0.34
    min_center_gap_frac: float = 0.135
    rotation_candidates_degrees: Tuple[int, ...] = (0, 90, 180, 270)


__all__ = ["SCENE_ID", "PairedCanvasDefaults"]
