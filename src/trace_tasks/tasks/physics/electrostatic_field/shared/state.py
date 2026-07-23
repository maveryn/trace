"""State containers and constants for electrostatic-field diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from trace_tasks.tasks.physics.shared.vector_arrows import SEMANTIC_DIRECTION_VECTORS


SCENE_ID = "electrostatic_field"
SCENE_NAMESPACE = "physics_electrostatic_field"
SCENE_PROMPT_KEY = "electrostatic_field_map"
SCENE_STYLE_VARIANTS: Tuple[str, ...] = ("clean_grid", "paper_grid", "dense_grid")
SCENE_MODE_DIRECTION = "direction"
SCENE_MODE_ZERO_FIELD = "zero_field"
SCENE_MODE_POTENTIAL = "potential"
DIRECTION_MODE_FIELD = "field"
DIRECTION_MODE_POSITIVE_FORCE = "positive_force"
DIRECTION_MODE_NEGATIVE_FORCE = "negative_force"
SUPPORTED_DIRECTION_MODES: Tuple[str, ...] = (
    DIRECTION_MODE_FIELD,
    DIRECTION_MODE_POSITIVE_FORCE,
    DIRECTION_MODE_NEGATIVE_FORCE,
)
SUPPORTED_DIRECTIONS: Tuple[str, ...] = (
    "east",
    "northeast",
    "north",
    "northwest",
    "west",
    "southwest",
    "south",
    "southeast",
)
DIRECTION_VECTORS: Dict[str, Tuple[int, int]] = dict(SEMANTIC_DIRECTION_VECTORS)
OPTION_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F", "G", "H")
POINT_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
POTENTIAL_DISTANCE_UNITS: Tuple[int, ...] = (2, 3, 4)
POTENTIAL_CHARGE_COORDS: Tuple[Tuple[int, int], ...] = ((2, 0), (0, 3), (-4, 0))
POTENTIAL_ANSWER_SUPPORT: Tuple[int, ...] = tuple(range(-9, 10))
POTENTIAL_CONTRIBUTION_SUPPORT: Tuple[int, ...] = (-3, -2, -1, 1, 2, 3)


@dataclass(frozen=True)
class RenderDefaults:
    """Stable fallback defaults for electrostatic field-map scenes."""

    canvas_width: int = 1180
    canvas_height: int = 760
    board_left_px: int = 58
    board_top_px: int = 58
    board_width_px: int = 760
    board_height_px: int = 560
    coord_extent: int = 5
    grid_line_width_px: int = 1
    dense_grid_line_width_px: int = 2
    axis_width_px: int = 4
    charge_radius_px: int = 30
    point_radius_px: int = 14
    candidate_point_radius_px: int = 10
    label_font_size_px: int = 22
    option_font_size_px: int = 25
    note_font_size_px: int = 21
    charge_font_size_px: int = 25
    option_panel_left_px: int = 850
    option_panel_top_px: int = 82
    option_cell_width_px: int = 140
    option_cell_height_px: int = 118
    option_cell_gap_x_px: int = 22
    option_cell_gap_y_px: int = 22
    option_arrow_length_px: int = 64
    option_arrow_width_px: int = 7
    option_arrow_head_length_px: int = 20
    option_arrow_head_width_px: int = 18


@dataclass(frozen=True)
class Charge:
    """One fixed point charge in grid coordinates."""

    charge_id: str
    charge_value: int
    x: int
    y: int


@dataclass(frozen=True)
class CandidatePoint:
    """One labeled candidate point in grid coordinates."""

    letter: str
    x: int
    y: int
    is_correct: bool


@dataclass(frozen=True)
class PotentialCharge:
    """One visible charge with an exact displayed distance to the query point."""

    charge_id: str
    charge_value: int
    contribution: int
    distance_units: int
    x: int
    y: int


@dataclass(frozen=True)
class DirectionScenario:
    """One symbolic field-direction scenario."""

    charges: Tuple[Charge, ...]
    point_x: int
    point_y: int
    direction_mode: str
    test_charge_sign: str | None
    field_direction: str
    requested_direction: str
    option_directions: Dict[str, str]


@dataclass(frozen=True)
class ZeroFieldScenario:
    """One symbolic zero-field candidate-point scenario."""

    charges: Tuple[Charge, ...]
    candidate_points: Tuple[CandidatePoint, ...]
    correct_option_letter: str
    symmetry_axis: str


@dataclass(frozen=True)
class PotentialScenario:
    """One symbolic electric-potential scenario."""

    charges: Tuple[PotentialCharge, ...]
    point_x: int
    point_y: int
    potential_value: int


@dataclass(frozen=True)
class SceneAxes:
    """Scene-level style axes shared by the three electrostatic tasks."""

    scene_variant: str
    accent_color_name: str
    scene_variant_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]


@dataclass(frozen=True)
class DirectionAxes:
    """Task-owned field/force direction sampling axes."""

    direction_mode: str
    target_direction: str
    correct_option_letter: str
    direction_mode_probabilities: Dict[str, float]
    target_direction_probabilities: Dict[str, float]
    correct_option_letter_probabilities: Dict[str, float]


@dataclass(frozen=True)
class OptionLetterAxes:
    """Task-owned option-letter sampling axes."""

    correct_option_letter: str
    correct_option_letter_probabilities: Dict[str, float]


@dataclass(frozen=True)
class PotentialAxes:
    """Task-owned exact potential answer sampling axes."""

    target_answer: int
    target_answer_probabilities: Dict[str, float]
    answer_support: Tuple[int, ...]
    contribution_support: Tuple[int, ...]


@dataclass(frozen=True)
class SceneSpec:
    """Resolved symbolic electrostatic scene."""

    scene_mode: str
    scene_variant: str
    target_answer: int | str
    direction_scenario: DirectionScenario | None
    zero_field_scenario: ZeroFieldScenario | None
    potential_scenario: PotentialScenario | None
    annotation_entity_ids: Tuple[str, ...]


@dataclass(frozen=True)
class RenderedElectrostaticCore:
    """Rendered electrostatic scene plus prompt-facing annotation metadata."""

    image: Any
    annotation_bboxes: List[List[float]]
    annotation_points: List[List[float]]
    annotation_point_map: Dict[str, List[float]]
    annotation_entity_ids: List[str]
    annotation_key_by_entity_id: Dict[str, str]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class RenderedElectrostaticScene(RenderedElectrostaticCore):
    """Final rendered scene with style, font, and post-processing metadata."""

    font_family: str
    diagram_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    layout_placement_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
