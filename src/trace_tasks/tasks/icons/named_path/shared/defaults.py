"""Defaults and constants for the named-path icons scene."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ....shared.labeling import LABEL_POOL_A_L
from ...shared.defaults import ICON_SHARED_DEFAULTS
from ...shared.procedural_named_icons import PROCEDURAL_NAMED_ICON_FILL_STYLES


SCENE_ID = "named_path"
OPTION_LABELS: Tuple[str, ...] = tuple(str(label) for label in LABEL_POOL_A_L[:6])


@dataclass(frozen=True)
class NamedPathDefaults:
    """Stable fallback defaults for named-path rendering and sampling."""

    candidate_count: int = 6
    distractor_count_min: int = 4
    distractor_count_max: int = 8
    target_occurrence_count_min: int = 2
    target_occurrence_count_max: int = 4
    canvas_width: int = 1280
    canvas_height: int = 720
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    scene_icon_size_min_px: int = 52
    scene_icon_size_max_px: int = 78
    reference_panel_width_px: int = ICON_SHARED_DEFAULTS.reference_panel_width_px
    reference_icon_size_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    reference_icon_size_min_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    reference_icon_size_max_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
    scene_max_overlap_fraction: float = 0.0
    scene_placement_max_attempts: int = 200
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
    path_stroke_width_px: int = 7
    path_line_alpha: int = 145
    path_stop_radius_px: int = 6
    path_horizontal_margin_px: int = 72
    path_vertical_margin_px: int = 92
    path_amplitude_min_px: int = 72
    path_amplitude_max_px: int = 136
    icon_collision_gap_px: int = 6
    candidate_label_font_size_px: int = 24
    candidate_label_padding_px: int = 5
    candidate_label_gap_px: int = 5
    candidate_label_color_rgb: Tuple[int, int, int] = (52, 60, 77)
    candidate_label_background_rgb: Tuple[int, int, int] = (255, 255, 255)
    candidate_label_border_rgb: Tuple[int, int, int] = (172, 183, 204)
    path_color_rgb: Tuple[int, int, int] = (92, 108, 134)
    stop_fill_rgb: Tuple[int, int, int] = (255, 255, 255)
    stop_outline_rgb: Tuple[int, int, int] = (92, 108, 134)
    endpoint_label_font_size_px: int = 18
    endpoint_label_color_rgb: Tuple[int, int, int] = (58, 68, 86)
    endpoint_label_background_rgb: Tuple[int, int, int] = (255, 255, 255)
