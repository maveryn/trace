"""Passive state and defaults for lever-balance diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "lever"
SCENE_NAMESPACE = "physics_lever"
SCENE_PROMPT_KEY = "lever_balance_diagram"
SUPPORTED_SCENE_VARIANTS: Tuple[str, ...] = (
    "center_fulcrum",
    "offset_fulcrum",
    "textured_beam",
)
MISSING_WEIGHT_SCENE_VARIANTS: Tuple[str, ...] = (
    "center_fulcrum",
    "textured_beam",
)
TORQUE_SIDES: Tuple[str, ...] = ("left", "right")
LEVER_SEMANTIC_COLORS: Tuple[Tuple[int, int, int], ...] = (
    (255, 238, 240),
    (192, 62, 84),
    (196, 56, 79),
)


@dataclass(frozen=True)
class LeverTaskDefaults:
    """Stable fallback defaults for lever-balance scenes."""

    canvas_width: int = 1280
    canvas_height: int = 640
    beam_width_px: int = 1140
    beam_height_px: int = 22
    beam_corner_radius_px: int = 10
    beam_center_y_px: int = 350
    fulcrum_width_px: int = 110
    fulcrum_height_px: int = 84
    fulcrum_offset_px: int = 76
    slot_spacing_px: int = 66
    distance_support: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)
    weight_box_width_px: int = 58
    weight_box_height_px: int = 52
    weight_box_gap_px: int = 8
    weight_font_size_px: int = 26
    distance_font_size_px: int = 22
    label_stroke_width_px: int = 3
    texture_line_width_px: int = 2
    texture_spacing_px: int = 18
    torque_answer_support: Tuple[int, ...] = tuple(range(2, 25))
    missing_weight_support: Tuple[int, ...] = tuple(range(1, 7))
    weight_value_min: int = 1
    weight_value_max: int = 9
    max_side_weights: int = 4
    missing_weight_max_side_weights: int = 2


@dataclass(frozen=True)
class LeverSideTorqueAxes:
    """Resolved generation axes for one side-torque instance."""

    public_branch: str
    scene_variant: str
    torque_side: str
    accent_color_name: str
    target_answer: int
    public_branch_probabilities: Dict[str, float]
    scene_variant_probabilities: Dict[str, float]
    torque_side_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class LeverMissingWeightAxes:
    """Resolved generation axes for one missing-weight instance."""

    public_branch: str
    scene_variant: str
    accent_color_name: str
    target_answer: int
    public_branch_probabilities: Dict[str, float]
    scene_variant_probabilities: Dict[str, float]
    accent_color_name_probabilities: Dict[str, float]
    target_answer_probabilities: Dict[str, float]


@dataclass(frozen=True)
class LeverWeightSlot:
    """One symbolic weight or missing-weight placeholder before rendering."""

    side: str
    distance_units: int
    value: int | None
    missing: bool
    relevant: bool


@dataclass(frozen=True)
class LeverWeightPlacement:
    """One rendered weight or placeholder attached to the lever."""

    weight_id: str
    side: str
    distance_units: int
    value: int | None
    missing: bool
    relevant: bool
    bbox_px: List[float]


@dataclass(frozen=True)
class RenderedLeverScene:
    """Rendered lever-balance scene plus prompt-facing annotation metadata."""

    image: Image.Image
    beam_bbox_px: List[float]
    fulcrum_bbox_px: List[float]
    weight_specs: List[LeverWeightPlacement]
    placeholder_bbox_px: List[float] | None
    relevant_weight_bboxes: List[List[float]]
    relevant_weight_ids: List[str]
    known_weight_bboxes: List[List[float]]
    target_weight_bboxes: List[List[float]]
    render_map: Dict[str, Any]
    scene_entities: List[Dict[str, Any]]
    max_distance_units: int
