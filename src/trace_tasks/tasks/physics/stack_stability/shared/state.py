"""Scene state for stack-stability brick-stack diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


SCENE_ID = "stack_stability"
SCENE_PROMPT_KEY = "stack_stability_diagram"
STATUS_STABLE = "stable"
STATUS_TIPPING = "tipping"
OPTION_LETTERS: Tuple[str, ...] = ("A", "B", "C", "D", "E", "F")
TIP_DIRECTIONS: Tuple[str, ...] = ("left", "right")
SCENE_NAMESPACE = "physics_stack_stability"


@dataclass(frozen=True)
class StackTaskDefaults:
    canvas_width: int = 1180
    canvas_height: int = 780
    board_left_px: int = 48
    board_top_px: int = 48
    board_right_margin_px: int = 48
    board_bottom_margin_px: int = 48
    cell_gap_x_px: int = 26
    cell_gap_y_px: int = 28
    brick_width_px: int = 80
    brick_height_px: int = 32
    brick_gap_px: int = 3
    label_font_size_px: int = 31
    title_font_size_px: int = 25
    small_font_size_px: int = 18
    label_stroke_width_px: int = 2
    support_width_px: int = 5
    projection_width_px: int = 4
    com_radius_px: int = 8


@dataclass(frozen=True)
class StackProfile:
    status: str
    tip_direction: str | None
    row_offsets: Tuple[float, ...]


@dataclass(frozen=True)
class StackCandidateSpec:
    label: str
    status: str
    tip_direction: str | None
    row_offsets: Tuple[float, ...]
    brick_fill_rgb: Tuple[int, int, int]
    brick_outline_rgb: Tuple[int, int, int]


@dataclass(frozen=True)
class StackSceneSpec:
    target_status: str
    correct_option_letter: str
    candidates: Tuple[StackCandidateSpec, ...]


@dataclass(frozen=True)
class RenderedStack:
    label: str
    status: str
    tip_direction: str | None
    brick_bboxes_px: Tuple[List[float], ...]
    stack_bbox_px: List[float]
    support_bbox_px: List[float]
    center_of_mass_point_px: List[float]
    center_of_mass_bbox_px: List[float]
    projection_point_px: List[float]
    projection_bbox_px: List[float]
    support_left_px: float
    support_right_px: float
    com_offset_units: float


@dataclass(frozen=True)
class RenderedStackScene:
    image: Image.Image
    annotation_bbox_px: List[float]
    scene_entities: List[Dict[str, Any]]
    render_map: Dict[str, Any]


__all__ = [
    "OPTION_LETTERS",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "SCENE_PROMPT_KEY",
    "STATUS_STABLE",
    "STATUS_TIPPING",
    "TIP_DIRECTIONS",
    "RenderedStack",
    "RenderedStackScene",
    "StackCandidateSpec",
    "StackProfile",
    "StackSceneSpec",
    "StackTaskDefaults",
]
