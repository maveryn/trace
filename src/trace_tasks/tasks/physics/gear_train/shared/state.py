"""State objects for physics gear-train diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "gear_train"
SCENE_NAMESPACE = "physics_gear_train"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = ("straight_chain", "staggered_chain", "arc_chain")
SUPPORTED_DIRECTIONS: Tuple[str, ...] = ("clockwise", "counterclockwise")
SUPPORTED_SPEED_RELATIONS: Tuple[str, ...] = ("faster", "slower")
DIRECTION_OPTION_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D")


@dataclass(frozen=True)
class GearTrainDefaults:
    """Stable fallback defaults for gear-train diagrams."""

    canvas_width: int = 1040
    canvas_height: int = 680
    panel_margin_x_px: int = 54
    panel_margin_top_px: int = 52
    panel_margin_bottom_px: int = 58
    gear_count_support: Tuple[int, ...] = (2, 3, 4, 5, 6)
    gear_radius_px_support: Tuple[int, ...] = (42, 48, 54, 60)
    label_font_size_px: int = 23
    title_font_size_px: int = 27
    tooth_count_support: Tuple[int, ...] = (12, 16, 18, 20, 24, 30, 36, 40, 48)
    input_rpm_support: Tuple[int, ...] = (30, 40, 45, 60, 72, 80, 90, 100, 120, 144, 150, 180)


@dataclass(frozen=True)
class GearDirectionScenario:
    """Resolved symbolic gear-direction scenario."""

    scene_variant: str
    gear_count: int
    input_direction: str
    output_direction: str
    radii_px: Tuple[float, ...]
    scene_variant_probabilities: Dict[str, float]
    gear_count_probabilities: Dict[str, float]
    input_direction_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class GearDirectionChoiceScenario:
    """Resolved four-panel gear-direction option scenario."""

    target_direction: str
    correct_option_letter: str
    panel_scenarios: Dict[str, GearDirectionScenario]
    target_direction_probabilities: Dict[str, float]
    correct_option_letter_probabilities: Dict[str, float]
    scene_variant_probabilities: Dict[str, float]
    gear_count_probabilities: Dict[str, float]


@dataclass(frozen=True)
class GearSpeedScenario:
    """Resolved symbolic gear-speed scenario."""

    scene_variant: str
    gear_count: int
    input_teeth: int
    output_teeth: int
    idler_teeth: Tuple[int, ...]
    input_rpm: int
    output_rpm: int
    speed_relation: str
    radii_px: Tuple[float, ...]
    scene_variant_probabilities: Dict[str, float]
    gear_count_probabilities: Dict[str, float]
    speed_relation_probabilities: Dict[str, float]
    input_rpm_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedGearTrainScene:
    """Rendered gear-train scene and projected annotation."""

    image: Image.Image
    annotation_bbox_map: Dict[str, List[float]]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
    font_family: str
    diagram_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


__all__ = [
    "DIRECTION_OPTION_LETTERS",
    "GearDirectionChoiceScenario",
    "GearDirectionScenario",
    "GearSpeedScenario",
    "GearTrainDefaults",
    "RenderedGearTrainScene",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SUPPORTED_DIRECTIONS",
    "SUPPORTED_SCENE_VARIANTS",
    "SUPPORTED_SPEED_RELATIONS",
]
