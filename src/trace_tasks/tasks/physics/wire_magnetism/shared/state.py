"""State and dataclasses for wire-magnetism diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "wire_magnetism"
SCENE_NAMESPACE = "physics_wire_magnetism"
SCENE_PROMPT_KEY = "wire_magnetism_diagram"
OPTION_LABELS: Tuple[str, ...] = ("A", "B", "C", "D")
SUPPORTED_CURRENT_DIRECTIONS: Tuple[str, ...] = ("out_of_page", "into_page")
SUPPORTED_POINT_POSITIONS: Tuple[str, ...] = ("north", "south", "east", "west")


@dataclass(frozen=True)
class WireMagnetismDefaults:
    """Stable fallback defaults for wire-magnetism diagrams."""

    canvas_width: int = 1080
    canvas_height: int = 700


@dataclass(frozen=True)
class WireScenario:
    """Resolved physical setup and answer binding for one wire diagram."""

    current_direction: str
    current_z_sign: int
    point_position: str
    point_offset_phys: Tuple[int, int]
    field_direction: str
    option_map: Dict[str, str]
    correct_label: str
    current_direction_probabilities: Dict[str, float]
    point_position_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class RenderedWireMagnetismScene:
    """Rendered wire-magnetism scene plus verifier-facing metadata."""

    image: Image.Image
    annotation_bboxes: Dict[str, List[float]]
    annotation_entity_ids: List[str]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]


__all__ = [
    "OPTION_LABELS",
    "RenderedWireMagnetismScene",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_PROMPT_KEY",
    "SUPPORTED_CURRENT_DIRECTIONS",
    "SUPPORTED_POINT_POSITIONS",
    "WireMagnetismDefaults",
    "WireScenario",
]
