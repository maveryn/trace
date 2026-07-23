"""Fallback defaults shared across icon scenes."""

from __future__ import annotations

from dataclasses import dataclass, field

from .icon_noise import default_icon_noise_value_ranges


@dataclass(frozen=True)
class IconSharedDefaults:
    """Stable fallback values for two-panel icon scenes."""

    canvas_width: int = 960
    canvas_height: int = 544
    reference_panel_width_px: int = 248
    panel_gap_px: int = 24
    outer_margin_px: int = 24
    panel_padding_px: int = 20
    panel_corner_radius_px: int = 18
    scene_icon_size_min_px: int = 40
    scene_icon_size_max_px: int = 96
    reference_icon_size_px: int = 136
    object_count_min: int = 1
    object_count_max: int = 20
    target_count_min: int = 0
    target_count_max: int = 10
    distractor_count_min: int = 1
    distractor_count_max: int = 10
    scene_max_overlap_fraction: float = 0.10
    scene_placement_max_attempts: int = 120
    scene_size_shrink_rounds: int = 6
    scene_size_shrink_factor: float = 0.90
    background_color_rgb: tuple[int, int, int] = (247, 248, 251)
    panel_fill_rgb: tuple[int, int, int] = (255, 255, 255)
    panel_border_rgb: tuple[int, int, int] = (205, 212, 224)
    header_text_rgb: tuple[int, int, int] = (70, 78, 96)
    panel_title_font_size_px: int = 24
    icon_noise_edit_types: tuple[str, ...] = ("blur", "downsample", "jpeg", "noise")
    icon_noise_edit_count_range: tuple[int, int] = (0, 2)
    icon_noise_value_ranges: dict[str, dict[str, tuple[float, float]]] = field(
        default_factory=default_icon_noise_value_ranges
    )


ICON_SHARED_DEFAULTS = IconSharedDefaults()


__all__ = ["ICON_SHARED_DEFAULTS", "IconSharedDefaults"]
