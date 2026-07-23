"""Passive state for the shadow-cause scene."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "shadow_cause"
SCENE_NAMESPACE = "physics_shadow_cause"
OPTION_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
SHADOW_DIRECTIONS: Tuple[str, ...] = (
    "east",
    "northeast",
    "north",
    "northwest",
    "west",
    "southwest",
    "south",
    "southeast",
)
OBJECT_SHAPES: Tuple[str, ...] = ("block", "cylinder", "sphere")
DIRECTION_VECTORS: Dict[str, Tuple[float, float]] = {
    "east": (1.0, 0.0),
    "northeast": (math.sqrt(0.5), -math.sqrt(0.5)),
    "north": (0.0, -1.0),
    "northwest": (-math.sqrt(0.5), -math.sqrt(0.5)),
    "west": (-1.0, 0.0),
    "southwest": (-math.sqrt(0.5), math.sqrt(0.5)),
    "south": (0.0, 1.0),
    "southeast": (math.sqrt(0.5), math.sqrt(0.5)),
}
OPPOSITE_DIRECTION: Dict[str, str] = {
    "east": "west",
    "northeast": "southwest",
    "north": "south",
    "northwest": "southeast",
    "west": "east",
    "southwest": "northeast",
    "south": "north",
    "southeast": "northwest",
}


@dataclass(frozen=True)
class ShadowCauseTaskDefaults:
    """Fallback rendering constants for one shadow-cause diagram."""

    canvas_width: int = 1120
    canvas_height: int = 760
    floor_left_px: int = 48
    floor_top_px: int = 50
    floor_right_margin_px: int = 48
    floor_bottom_margin_px: int = 48
    object_center_x_px: int = 560
    object_base_y_px: int = 430
    lamp_radius_x_px: int = 355
    lamp_radius_y_px: int = 238
    lamp_bulb_radius_px: int = 19
    lamp_label_font_size_px: int = 29
    label_stroke_width_px: int = 2
    title_font_size_px: int = 25
    shadow_length_px: int = 178
    shadow_length_px_min: int = 158
    shadow_length_px_max: int = 204
    shadow_base_width_px: int = 38
    shadow_tip_width_px: int = 78
    object_size_px: int = 88


@dataclass(frozen=True)
class ShadowCauseAxes:
    """Resolved sampling axes for one shadow-cause instance."""

    correct_option_letter: str
    shadow_direction: str
    object_shape: str
    correct_option_letter_probabilities: Dict[str, float]
    shadow_direction_probabilities: Dict[str, float]
    object_shape_probabilities: Dict[str, float]


@dataclass(frozen=True)
class LampSpec:
    """Candidate light-source geometry and option identity."""

    label: str
    direction: str
    center_px: Tuple[float, float]


@dataclass(frozen=True)
class ShadowSceneSpec:
    """Resolved physical and answer state for the diagram."""

    correct_option_letter: str
    shadow_direction: str
    source_direction: str
    object_shape: str
    object_fill_rgb: Tuple[int, int, int]
    lamps: Tuple[LampSpec, ...]


@dataclass(frozen=True)
class RenderedShadowCauseScene:
    """Rendered image plus projected verifier geometry."""

    image: Image.Image
    context_bbox_map: Dict[str, List[float]]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]
