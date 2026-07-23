"""Defaults for named-ring icon scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Tuple

from ...shared.defaults import ICON_SHARED_DEFAULTS
from ...shared.procedural_named_icons import PROCEDURAL_NAMED_ICON_FILL_STYLES


SCENE_ID = "named_ring"
MARKER_LABELS: Tuple[str, str] = ("A", "B")


@dataclass(frozen=True)
class NamedRingDefaults:
    """Stable fallback defaults for named-ring rendering and sampling."""

    ring_icon_count_min: int = 12
    ring_icon_count_max: int = 22
    answer_count_min: int = 0
    answer_count_max: int = 6
    arc_span_min: int = 3
    arc_span_max: int = 12
    off_arc_target_count_min: int = 1
    off_arc_target_count_max: int = 4
    canvas_width: int = 880
    canvas_height: int = 680
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    reference_panel_width_px: int = ICON_SHARED_DEFAULTS.reference_panel_width_px
    reference_icon_size_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    reference_icon_size_min_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    reference_icon_size_max_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
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
    ring_margin_px: int = 86
    ring_stroke_width_px: int = 4
    ring_outline_rgb: Tuple[int, int, int] = (142, 154, 178)
    ring_stop_radius_px: int = 4
    ring_stop_fill_rgb: Tuple[int, int, int] = (255, 255, 255)
    ring_stop_outline_rgb: Tuple[int, int, int] = (122, 136, 164)
    marker_label_font_size_px: int = 24
    marker_label_radius_px: int = 18
    marker_label_gap_px: int = 8
    marker_label_background_rgb: Tuple[int, int, int] = (255, 255, 255)
    marker_label_border_rgb: Tuple[int, int, int] = (56, 70, 98)
    marker_label_color_rgb: Tuple[int, int, int] = (39, 50, 72)
