"""State and dataclasses for two-source wave-interference diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "wave_interference"
SCENE_NAMESPACE = "physics_wave_interference"
SCENE_PROMPT_KEY = "wave_interference_tank"
SOURCE_SEPARATION_STEPS = 8

SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "clean_tank",
    "grid_tank",
    "lab_sheet",
)
SUPPORTED_PHASE_RELATIONS: Tuple[str, ...] = ("in_phase", "opposite_phase")
SUPPORTED_TARGET_CONDITIONS: Tuple[str, ...] = ("constructive", "destructive")
OPTION_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D", "E")


@dataclass(frozen=True)
class WaveInterferenceDefaults:
    """Stable fallback defaults for wave-interference scenes."""

    canvas_width: int = 1180
    canvas_height: int = 760
    board_left_px: int = 70
    board_top_px: int = 50
    board_width_px: int = 980
    board_height_px: int = 620
    half_wavelength_px: int = 50
    ring_count: int = 10
    grid_line_width_px: int = 1
    source_radius_px: int = 24
    candidate_radius_px: int = 15
    point_radius_px: int = 18
    wavefront_width_px: int = 2
    guide_width_px: int = 4
    label_font_size_px: int = 22
    source_font_size_px: int = 21
    candidate_font_size_px: int = 23
    note_font_size_px: int = 20
    path_difference_step_support: Tuple[int, ...] = (1, 2, 3, 4, 5)


@dataclass(frozen=True)
class CommonAxes:
    """Resolved scene axes shared by both wave-interference objectives."""

    scene_variant: str
    phase_relation: str
    accent_color_name: str
    scene_variant_probabilities: Dict[str, float]
    phase_relation_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]


@dataclass(frozen=True)
class ChoiceAxes:
    """Resolved axes for the candidate-point choice objective."""

    target_condition: str
    correct_option_letter: str
    target_condition_probabilities: Dict[str, float]
    correct_option_letter_probabilities: Dict[str, float]


@dataclass(frozen=True)
class PathDifferenceAxes:
    """Resolved axes for the path-difference objective."""

    path_difference_steps: int
    path_difference_steps_probabilities: Dict[str, float]


@dataclass(frozen=True)
class PointTemplate:
    """One exact point described by half-wavelength source distances."""

    r1_steps: int
    r2_steps: int
    sign_y: int
    x_steps: float
    y_steps: float


@dataclass(frozen=True)
class CandidatePoint:
    """One labeled interference candidate point."""

    letter: str
    x_steps: float
    y_steps: float
    r1_steps: int
    r2_steps: int
    condition: str
    is_correct: bool


@dataclass(frozen=True)
class ChoiceScenario:
    """One two-source interference point-choice scenario."""

    phase_relation: str
    target_condition: str
    candidates: Tuple[CandidatePoint, ...]
    correct_option_letter: str


@dataclass(frozen=True)
class PathDifferenceScenario:
    """One two-source path-difference value scenario."""

    phase_relation: str
    point_x_steps: float
    point_y_steps: float
    r1_steps: int
    r2_steps: int
    path_difference_steps: int


@dataclass(frozen=True)
class SceneSpec:
    """Resolved symbolic wave-interference scene."""

    scene_variant: str
    phase_relation: str
    choice_scenario: ChoiceScenario | None
    path_scenario: PathDifferenceScenario | None
    target_answer: int | str
    annotation_entity_ids: Tuple[str, ...]


@dataclass(frozen=True)
class RenderedWaveInterferenceScene:
    """Rendered wave scene plus prompt-facing annotation metadata."""

    image: Image.Image
    annotation_type: str
    annotation_point: List[float] | None
    annotation_segments: List[List[List[float]]]
    annotation_entity_ids: List[str]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]


__all__ = [
    "ChoiceAxes",
    "ChoiceScenario",
    "CommonAxes",
    "CandidatePoint",
    "OPTION_LETTERS",
    "PathDifferenceAxes",
    "PathDifferenceScenario",
    "PointTemplate",
    "RenderedWaveInterferenceScene",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_PROMPT_KEY",
    "SOURCE_SEPARATION_STEPS",
    "SUPPORTED_PHASE_RELATIONS",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_TARGET_CONDITIONS",
    "SceneSpec",
    "WaveInterferenceDefaults",
]
