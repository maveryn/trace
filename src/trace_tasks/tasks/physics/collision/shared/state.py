"""State objects for the physics collision scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "collision"
SCENE_NAMESPACE = "physics_collision"
SINGLE_PUBLIC_BRANCH = "single"
STICKY_DIRECTION_OPERATION = "direction_choice"
STICKY_SPEED_OPERATION = "speed_value"

OPTION_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
STICKY_DIRECTION_OPTION_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D")
DIRECTION_NAMES: Tuple[str, ...] = (
    "east",
    "northeast",
    "north",
    "northwest",
    "west",
    "southwest",
    "south",
    "southeast",
)
DIRECTION_ANGLE_DEGREES: Dict[str, float] = {
    "east": 0.0,
    "northeast": 45.0,
    "north": 90.0,
    "northwest": 135.0,
    "west": 180.0,
    "southwest": 225.0,
    "south": 270.0,
    "southeast": 315.0,
}
SUPPORTED_AFTER_EFFECT_VARIANTS: Tuple[str, ...] = (
    "aftermath_table",
    "aftermath_gridded_table",
    "aftermath_compact_table",
)
SUPPORTED_STICKY_VARIANTS: Tuple[str, ...] = (
    "wide_table",
    "compact_table",
    "gridded_table",
)
ANNOTATION_ENTITY_KEY_BY_ID: Dict[str, str] = {
    "horizontal_puck": "A",
    "vertical_puck": "B",
    "stuck_pucks": "A+B",
}
ANNOTATION_ENTITY_IDS: Tuple[str, ...] = tuple(ANNOTATION_ENTITY_KEY_BY_ID.keys())
INPUT_MOTION_ARROW_ENTITY_IDS: Tuple[str, ...] = (
    "horizontal_motion_arrow",
    "vertical_motion_arrow",
)


@dataclass(frozen=True)
class AftermathRenderDefaults:
    """Stable fallback rendering defaults for collision aftermath diagrams."""

    canvas_width: int = 1180
    canvas_height: int = 760
    table_left_px: int = 52
    table_top_px: int = 52
    table_width_px: int = 1076
    table_height_px: int = 640
    table_corner_radius_px: int = 22
    impact_center_x_px: int = 590
    impact_center_y_px: int = 380
    compact_impact_center_y_px: int = 364
    puck_radius_px: int = 38
    impact_radius_px: int = 16
    target_distance_px: int = 156
    trail_start_gap_px: int = 18
    incoming_arrow_inner_gap_px: int = 58
    incoming_arrow_length_px: int = 210
    candidate_label_offset_px: int = 38
    motion_arrow_width_px: int = 8
    candidate_arrow_width_px: int = 6
    arrow_head_length_px: int = 23
    arrow_head_width_px: int = 19
    candidate_arrow_head_length_px: int = 20
    candidate_arrow_head_width_px: int = 17
    title_font_size_px: int = 27
    label_font_size_px: int = 23
    puck_font_size_px: int = 23
    label_stroke_width_px: int = 2
    grid_spacing_px: int = 44


@dataclass(frozen=True)
class StickyRenderDefaults:
    """Stable fallback rendering and sampling defaults for sticky collisions."""

    canvas_width: int = 1180
    canvas_height: int = 760
    table_left_px: int = 52
    table_top_px: int = 48
    table_width_px: int = 1076
    table_height_px: int = 506
    table_corner_radius_px: int = 22
    collision_center_x_px: int = 436
    collision_center_y_px: int = 304
    compact_collision_center_x_px: int = 420
    compact_collision_center_y_px: int = 292
    horizontal_start_distance_px: int = 300
    vertical_start_distance_px: int = 200
    compact_start_distance_delta_px: int = -20
    puck_radius_px: int = 42
    stuck_radius_px: int = 50
    motion_arrow_width_px: int = 8
    final_arrow_width_px: int = 8
    arrow_head_length_px: int = 24
    arrow_head_width_px: int = 22
    label_font_size_px: int = 23
    puck_font_size_px: int = 30
    option_font_size_px: int = 24
    label_stroke_width_px: int = 2
    option_panel_top_px: int = 584
    option_cell_left_px: int = 82
    option_cell_width_px: int = 164
    option_arrow_length_px: int = 74
    option_arrow_width_px: int = 7
    option_arrow_head_length_px: int = 20
    option_arrow_head_width_px: int = 18
    grid_spacing_px: int = 44
    show_candidate_options: bool = True
    mass_support: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)
    speed_support: Tuple[int, ...] = (2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)
    component_answer_support: Tuple[int, ...] = (
        -6,
        -5,
        -4,
        -3,
        -2,
        -1,
        1,
        2,
        3,
        4,
        5,
        6,
    )
    speed_answer_tenths_support: Tuple[int, ...] = (
        14,
        22,
        28,
        32,
        36,
        41,
        42,
        45,
        50,
        51,
        54,
        57,
        58,
        61,
        63,
        64,
        67,
        71,
        72,
        78,
        85,
    )


@dataclass(frozen=True)
class AftermathAxes:
    """Resolved non-public axes for one aftermath instance."""

    scene_variant: str
    final_motion_direction: str
    correct_option_letter: str
    accent_color_name: str
    scene_variant_probabilities: Dict[str, float]
    final_motion_direction_probabilities: Dict[str, float]
    correct_option_letter_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]


@dataclass(frozen=True)
class CollisionAftermathSpec:
    """Symbolic collision aftermath scene specification."""

    scene_variant: str
    final_motion_direction: str
    correct_option_letter: str
    option_directions: Dict[str, str]
    option_angles_degrees: Dict[str, float]


@dataclass(frozen=True)
class StickyAxes:
    """Resolved non-public axes for one sticky collision instance."""

    scene_variant: str
    component_axis: str | None
    target_speed_tenths: int | None
    accent_color_name: str
    target_answer: int | float | str
    correct_option_letter: str
    option_letters: Tuple[str, ...]
    scene_variant_probabilities: Dict[str, float]
    component_axis_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]
    correct_option_letter_probabilities: Dict[str, float]


@dataclass(frozen=True)
class StickyScenario:
    """One symbolic perpendicular sticky-collision setup."""

    horizontal_mass: int
    vertical_mass: int
    horizontal_speed: int
    vertical_speed: int
    horizontal_sign: int
    vertical_sign: int
    final_vx: int
    final_vy: int
    total_mass: int


@dataclass(frozen=True)
class StickySceneSpec:
    """Resolved symbolic sticky-collision scene."""

    scene_variant: str
    component_axis: str | None
    scenario: StickyScenario
    correct_option_letter: str
    option_letters: Tuple[str, ...]
    option_angles_degrees: Dict[str, float]
    direction_label: str
    target_answer: int | str
    annotation_entity_ids: Tuple[str, ...]


@dataclass(frozen=True)
class CollisionAftermathRenderedScene:
    """Rendered aftermath diagram and projected annotation metadata."""

    image: Image.Image
    annotation_bbox_map: Dict[str, List[float]]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class StickyRenderedScene:
    """Rendered sticky-collision diagram and projected annotation metadata."""

    image: Image.Image
    annotation_points: List[List[float]]
    annotation_point_map: Dict[str, List[float]]
    annotation_entity_ids: List[str]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class PreparedRender:
    """Final image plus render metadata used by public task files."""

    image: Image.Image
    annotation_bbox_map: Dict[str, List[float]]
    annotation_point_map: Dict[str, List[float]]
    annotation_entity_ids: List[str]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    background_meta: Dict[str, Any]
    diagram_style_meta: Dict[str, Any]
    layout_placement_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    font_family: str
