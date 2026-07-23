"""Shared constants and fallback defaults for wallpaper-panel icon scenes."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Dict, Tuple

from ...shared.defaults import ICON_SHARED_DEFAULTS

DOMAIN = "icons"
SCENE_ID = "wallpaper_panels"
REFERENCE_LABEL = "Reference"
OPTION_LABELS: Tuple[str, ...] = tuple("ABCD")
LATTICE_ROWS = 4
LATTICE_COLS = 4
WALLPAPER_GROUP_IDS: Tuple[str, ...] = ("p1", "p2", "pm", "pg", "cm", "pmm", "p4", "p3")


@dataclass(frozen=True)
class WallpaperPanelDefaults:
    """Stable fallback defaults shared by wallpaper-panel tasks."""

    option_count_choices: Tuple[int, ...] = (4,)
    match_count_choices: Tuple[int, ...] = (1, 2, 3)
    lattice_rows: int = LATTICE_ROWS
    lattice_cols: int = LATTICE_COLS
    wallpaper_group_ids: Tuple[str, ...] = WALLPAPER_GROUP_IDS
    canvas_width: int = 1104
    canvas_height: int = 640
    reference_panel_width_px: int = 300
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = 16
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    option_panel_gap_px: int = 18
    scene_icon_size_min_px: int = 40
    scene_icon_size_max_px: int = 44
    cell_box_width_min_px: int = 0
    cell_box_width_max_px: int = 0
    cell_box_height_min_px: int = 0
    cell_box_height_max_px: int = 0
    scene_max_overlap_fraction: float = 0.12
    scene_placement_max_attempts: int = 1
    scene_size_shrink_rounds: int = ICON_SHARED_DEFAULTS.scene_size_shrink_rounds
    scene_size_shrink_factor: float = ICON_SHARED_DEFAULTS.scene_size_shrink_factor
    panel_title_font_size_px: int = 22
    pool_manifest: str = "non_symmetry.txt"
    palette_size_min: int = 1
    palette_size_max: int = 1
    color_channel_min: int = 24
    color_channel_max: int = 220
    min_color_distance: float = 40.0
    color_distance_space: str = "lab"
    background_color_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.background_color_rgb
    panel_fill_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_fill_rgb
    panel_border_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_border_rgb
    header_text_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.header_text_rgb
    cell_padding_px: int = 0
    cell_icon_padding_px: int = 0
    cell_corner_radius_px: int = 0
    cell_border_rgb: Tuple[int, int, int] = (218, 223, 233)
    cell_label_font_size_px: int = 22
    cell_label_color_rgb: Tuple[int, int, int] = (52, 60, 77)
    scene_content_side_padding_px: int = 0
    scene_content_bottom_padding_px: int = 0
    scene_content_top_offset_px: int = 0
    icon_noise_edit_types: Tuple[str, ...] = ICON_SHARED_DEFAULTS.icon_noise_edit_types
    icon_noise_edit_count_range: Tuple[int, int] = (0, 0)
    icon_noise_value_ranges: Dict[str, Dict[str, Tuple[float, float]]] = field(
        default_factory=lambda: deepcopy(ICON_SHARED_DEFAULTS.icon_noise_value_ranges)
    )


__all__ = [
    "DOMAIN",
    "LATTICE_COLS",
    "LATTICE_ROWS",
    "OPTION_LABELS",
    "REFERENCE_LABEL",
    "SCENE_ID",
    "WALLPAPER_GROUP_IDS",
    "WallpaperPanelDefaults",
]
