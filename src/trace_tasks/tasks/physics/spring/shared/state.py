"""State objects for the spring physics scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "spring"
SCENE_PROMPT_KEY = "spring_diagram"
SPRING_SEMANTIC_COLORS = ((255, 231, 231), (187, 56, 56), (167, 38, 38))
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "paired_springs",
    "staggered_springs",
    "textured_spring",
)
SPRING_MODE_MISSING_WEIGHT = "missing_weight"
SPRING_MODE_MISSING_EXTENSION = "missing_extension"
SPRING_MODE_DIFFERENCE = "difference"
SPRING_MISSING_MODES = (SPRING_MODE_MISSING_WEIGHT, SPRING_MODE_MISSING_EXTENSION)
ALL_SPRING_MODES = (*SPRING_MISSING_MODES, SPRING_MODE_DIFFERENCE)


@dataclass(frozen=True)
class SpringTaskDefaults:
    """Stable fallback defaults for spring-extension scenes."""

    canvas_width: int = 980
    canvas_height: int = 660
    card_width_px: int = 304
    card_height_px: int = 470
    card_left_px: int = 86
    card_top_px: int = 108
    card_gap_px: int = 108
    stagger_offset_y_px: int = 28
    card_corner_radius_px: int = 24
    card_outline_width_px: int = 4
    support_width_px: int = 164
    support_height_px: int = 16
    support_corner_radius_px: int = 8
    anchor_y_gap_px: int = 22
    hanger_line_width_px: int = 4
    ruler_top_gap_px: int = 18
    ruler_right_gap_px: int = 42
    ruler_value_max: int = 14
    ruler_unit_px: int = 25
    ruler_width_px: int = 4
    ruler_tick_long_px: int = 20
    ruler_tick_short_px: int = 12
    ruler_font_size_px: int = 19
    spring_neutral_units: int = 2
    spring_line_width_px: int = 5
    spring_half_width_px: int = 16
    spring_turn_count: int = 8
    weight_box_width_px: int = 78
    weight_box_height_px: int = 54
    weight_font_size_px: int = 28
    marker_height_px: int = 22
    marker_width_px: int = 52
    missing_tag_width_px: int = 52
    missing_tag_height_px: int = 34
    missing_tag_top_gap_px: int = 22
    label_stroke_width_px: int = 3
    texture_spacing_px: int = 18
    texture_line_width_px: int = 2
    scale_factor_support: Tuple[int, ...] = (1, 2, 3)
    weight_value_min: int = 1
    weight_value_max: int = 12
    extension_value_max: int = 14
    missing_weight_support: Tuple[int, ...] = tuple(range(1, 9))
    missing_extension_support: Tuple[int, ...] = tuple(range(1, 13))
    extension_difference_scale_factor_support: Tuple[int, ...] = (2,)
    extension_difference_support: Tuple[int, ...] = (2, 4, 8, 10, 12)


@dataclass(frozen=True)
class SpringAxes:
    """Resolved scene axes and answer support for one instance."""

    scene_variant: str
    spring_mode: str
    public_branch: str
    internal_branch: str
    solve_for: str | None
    accent_color_name: str
    target_answer: int
    scene_variant_probabilities: Dict[str, float]
    public_branch_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class SpringColumnSpec:
    """One spring card's symbolic measurement state."""

    column_id: str
    shown_weight_value: int | None
    true_weight_value: int
    shown_extension_value: int | None
    true_extension_value: int
    missing_weight: bool
    missing_extension: bool
    detached_weight: bool


@dataclass(frozen=True)
class SpringSceneSpec:
    """Resolved symbolic scene specification for one instance."""

    scene_variant: str
    spring_mode: str
    scale_factor: int
    left: SpringColumnSpec
    right: SpringColumnSpec
    target_answer: int
    annotation_entity_ids: Tuple[str, ...]


@dataclass(frozen=True)
class RenderedSpringScene:
    """Rendered spring-extension scene plus prompt-facing annotation metadata."""

    image: Image.Image
    annotation_bboxes: List[List[float]]
    annotation_entity_ids: List[str]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    shown_measurement_count: int
