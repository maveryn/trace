"""Scene-local defaults for named-field icon tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from ...shared.defaults import ICON_SHARED_DEFAULTS
from ...shared.procedural_named_icons import PROCEDURAL_NAMED_ICON_FILL_STYLES


NON_STACK_LAYOUT_MODES: Tuple[str, ...] = (
    "jittered_grid",
    "ordered_grid",
    "shelf_rows",
    "free_scatter",
)

SHAPE_COUNT_ARRANGEMENT_PROFILES = {
    "jittered_grid": {"target_count_min": 1, "target_count_max": 6, "object_count_min": 14, "object_count_max": 28},
    "ordered_grid": {"target_count_min": 1, "target_count_max": 6, "object_count_min": 14, "object_count_max": 28},
    "shelf_rows": {"target_count_min": 1, "target_count_max": 6, "object_count_min": 14, "object_count_max": 28},
    "free_scatter": {"target_count_min": 1, "target_count_max": 5, "object_count_min": 12, "object_count_max": 22},
    "clustered_by_shape": {"target_count_min": 1, "target_count_max": 6, "object_count_min": 14, "object_count_max": 30},
    "shape_stacks": {"target_count_min": 6, "target_count_max": 14, "object_count_min": 20, "object_count_max": 36},
    "target_stack_with_oddballs": {"target_count_min": 6, "target_count_max": 12, "object_count_min": 7, "object_count_max": 13},
    "mixed_stacks": {"target_count_min": 4, "target_count_max": 12, "object_count_min": 18, "object_count_max": 34},
}


@dataclass(frozen=True)
class ShapeCountNamedFieldDefaults:
    """Fallback defaults for direct named-shape count tasks."""

    object_count_min: int = 12
    object_count_max: int = 36
    target_count_min: int = 1
    target_count_max: int = 14
    canvas_width: int = 800
    canvas_height: int = 480
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    scene_icon_size_min_px: int = 48
    scene_icon_size_max_px: int = 96
    scene_max_overlap_fraction: float = 0.0
    scene_placement_max_attempts: int = ICON_SHARED_DEFAULTS.scene_placement_max_attempts
    scene_size_shrink_rounds: int = ICON_SHARED_DEFAULTS.scene_size_shrink_rounds
    scene_size_shrink_factor: float = ICON_SHARED_DEFAULTS.scene_size_shrink_factor
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    reference_panel_width_px: int = ICON_SHARED_DEFAULTS.reference_panel_width_px
    reference_icon_size_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
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
    named_icon_layout_modes: Tuple[str, ...] = (
        "jittered_grid",
        "ordered_grid",
        "shelf_rows",
        "free_scatter",
        "target_stack_with_oddballs",
    )
    named_icon_slot_padding_px: int = 6
    named_icon_slot_jitter_px: int = 8
    named_icon_stack_gap_px: int = 1
    named_icon_stack_distractor_group_min: int = 3
    named_icon_stack_distractor_group_max: int = 6
    named_icon_stack_oddball_count_min: int = 1
    named_icon_stack_oddball_count_max: int = 1
    named_icon_fill_style_support: Tuple[str, ...] = PROCEDURAL_NAMED_ICON_FILL_STYLES


@dataclass(frozen=True)
class PairArithmeticNamedFieldDefaults:
    """Fallback defaults for two-operand named-field arithmetic counts."""

    operand_count_min: int = 1
    operand_count_max: int = 6
    total_answer_min: int = 2
    total_answer_max: int = 10
    difference_answer_min: int = 0
    difference_answer_max: int = 5
    distractor_count_min: int = 4
    distractor_count_max: int = 8
    canvas_width: int = 800
    canvas_height: int = 480
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    scene_icon_size_min_px: int = 48
    scene_icon_size_max_px: int = 96
    scene_max_overlap_fraction: float = 0.0
    scene_placement_max_attempts: int = ICON_SHARED_DEFAULTS.scene_placement_max_attempts
    scene_size_shrink_rounds: int = ICON_SHARED_DEFAULTS.scene_size_shrink_rounds
    scene_size_shrink_factor: float = ICON_SHARED_DEFAULTS.scene_size_shrink_factor
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    reference_panel_width_px: int = ICON_SHARED_DEFAULTS.reference_panel_width_px
    reference_icon_size_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
    palette_size_min: int = 8
    palette_size_max: int = 10
    color_channel_min: int = 24
    color_channel_max: int = 230
    min_color_distance: float = 40.0
    color_distance_space: str = "lab"
    background_color_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.background_color_rgb
    panel_fill_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_fill_rgb
    panel_border_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_border_rgb
    header_text_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.header_text_rgb
    icon_noise_edit_types: Tuple[str, ...] = ICON_SHARED_DEFAULTS.icon_noise_edit_types
    icon_noise_edit_count_range: Tuple[int, int] = ICON_SHARED_DEFAULTS.icon_noise_edit_count_range
    named_icon_layout_modes: Tuple[str, ...] = NON_STACK_LAYOUT_MODES
    named_icon_slot_padding_px: int = 6
    named_icon_slot_jitter_px: int = 8
    named_icon_stack_gap_px: int = 1
    named_icon_fill_style_support: Tuple[str, ...] = PROCEDURAL_NAMED_ICON_FILL_STYLES


@dataclass(frozen=True)
class BooleanNamedFieldDefaults:
    """Fallback defaults for Boolean predicate named-field count tasks."""

    object_count_min: int = 8
    object_count_max: int = 20
    object_count_max_answer_offset: int = 10
    target_count_min: int = 1
    target_count_max: int = 6
    canvas_width: int = 800
    canvas_height: int = 480
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    scene_icon_size_min_px: int = 48
    scene_icon_size_max_px: int = 96
    scene_max_overlap_fraction: float = 0.0
    scene_placement_max_attempts: int = ICON_SHARED_DEFAULTS.scene_placement_max_attempts
    scene_size_shrink_rounds: int = ICON_SHARED_DEFAULTS.scene_size_shrink_rounds
    scene_size_shrink_factor: float = ICON_SHARED_DEFAULTS.scene_size_shrink_factor
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    reference_panel_width_px: int = ICON_SHARED_DEFAULTS.reference_panel_width_px
    reference_icon_size_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
    palette_size_min: int = 8
    palette_size_max: int = 10
    color_channel_min: int = 24
    color_channel_max: int = 230
    min_color_distance: float = 40.0
    color_distance_space: str = "lab"
    background_color_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.background_color_rgb
    panel_fill_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_fill_rgb
    panel_border_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.panel_border_rgb
    header_text_rgb: Tuple[int, int, int] = ICON_SHARED_DEFAULTS.header_text_rgb
    icon_noise_edit_types: Tuple[str, ...] = ICON_SHARED_DEFAULTS.icon_noise_edit_types
    icon_noise_edit_count_range: Tuple[int, int] = ICON_SHARED_DEFAULTS.icon_noise_edit_count_range
    named_icon_layout_modes: Tuple[str, ...] = NON_STACK_LAYOUT_MODES
    named_icon_slot_padding_px: int = 6
    named_icon_slot_jitter_px: int = 8
    named_icon_stack_gap_px: int = 1
    named_icon_fill_style_support: Tuple[str, ...] = PROCEDURAL_NAMED_ICON_FILL_STYLES


@dataclass(frozen=True)
class CounterfactualNamedFieldDefaults:
    """Fallback defaults for counterfactual named-field count tasks."""

    target_count_min: int = 1
    target_count_max: int = 6
    removal_count_min: int = 1
    removal_count_max: int = 4
    distractor_count_min: int = 2
    distractor_count_max: int = 5
    canvas_width: int = 800
    canvas_height: int = 480
    outer_margin_px: int = ICON_SHARED_DEFAULTS.outer_margin_px
    panel_padding_px: int = ICON_SHARED_DEFAULTS.panel_padding_px
    panel_corner_radius_px: int = ICON_SHARED_DEFAULTS.panel_corner_radius_px
    scene_icon_size_min_px: int = 48
    scene_icon_size_max_px: int = 96
    scene_max_overlap_fraction: float = 0.0
    scene_placement_max_attempts: int = ICON_SHARED_DEFAULTS.scene_placement_max_attempts
    scene_size_shrink_rounds: int = ICON_SHARED_DEFAULTS.scene_size_shrink_rounds
    scene_size_shrink_factor: float = ICON_SHARED_DEFAULTS.scene_size_shrink_factor
    panel_title_font_size_px: int = ICON_SHARED_DEFAULTS.panel_title_font_size_px
    reference_panel_width_px: int = ICON_SHARED_DEFAULTS.reference_panel_width_px
    reference_icon_size_px: int = ICON_SHARED_DEFAULTS.reference_icon_size_px
    panel_gap_px: int = ICON_SHARED_DEFAULTS.panel_gap_px
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
    named_icon_layout_modes: Tuple[str, ...] = NON_STACK_LAYOUT_MODES
    named_icon_slot_padding_px: int = 6
    named_icon_slot_jitter_px: int = 8
    named_icon_stack_gap_px: int = 1
    named_icon_fill_style_support: Tuple[str, ...] = PROCEDURAL_NAMED_ICON_FILL_STYLES


BOOLEAN_DEFAULTS = BooleanNamedFieldDefaults()
COUNTERFACTUAL_DEFAULTS = CounterfactualNamedFieldDefaults()
PAIR_ARITHMETIC_DEFAULTS = PairArithmeticNamedFieldDefaults()
SHAPE_COUNT_DEFAULTS = ShapeCountNamedFieldDefaults()


__all__ = [
    "BOOLEAN_DEFAULTS",
    "COUNTERFACTUAL_DEFAULTS",
    "SHAPE_COUNT_ARRANGEMENT_PROFILES",
    "SHAPE_COUNT_DEFAULTS",
    "BooleanNamedFieldDefaults",
    "CounterfactualNamedFieldDefaults",
    "NON_STACK_LAYOUT_MODES",
    "PAIR_ARITHMETIC_DEFAULTS",
    "PairArithmeticNamedFieldDefaults",
    "ShapeCountNamedFieldDefaults",
]
