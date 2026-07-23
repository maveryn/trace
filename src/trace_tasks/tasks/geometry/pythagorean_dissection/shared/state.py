"""State containers for Pythagorean dissection diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class PythagoreanLegCase:
    """One integer right-triangle leg pair used by the dissection."""

    leg_a: int
    leg_b: int

    @property
    def central_square_area(self) -> int:
        return int(self.leg_a) * int(self.leg_a) + int(self.leg_b) * int(self.leg_b)

    @property
    def central_square_side(self) -> float:
        return float(self.central_square_area) ** 0.5


@dataclass(frozen=True)
class PythagoreanDissectionPlan:
    """Task-bound construction values before pixel rendering."""

    answer: int
    leg_a: int
    leg_b: int
    outer_square_side: int
    vertical_square_area: int
    horizontal_square_area: int
    central_square_side: float
    case_index: int
    answer_support: Tuple[int, ...]
    witness: Dict[str, Any]


@dataclass
class RenderContext:
    """Styled PIL render context for one dissection."""

    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    line_color: Color
    secondary_color: Color
    label_color: Color
    label_stroke_color: Color
    leg_fill_color: Color
    other_leg_fill_color: Color
    central_fill_color: Color
    line_width: int
    label_stroke_width: int
    font: Any
    small_font: Any
    readout_text_metadata: Dict[str, Any]
    orientation_key: str
    orientation_sign_x: int
    orientation_sign_y: int
    diagram_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    scene_transform: LazySceneTransform


@dataclass(frozen=True)
class RenderedPythagoreanDissectionScene:
    """Rendered dissection plus projected verifier fragments."""

    image: Image.Image
    annotation_roles: Tuple[str, ...]
    annotation_keyed_points: Mapping[str, Point]
    label_bboxes: Dict[str, BBox]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    witness: Dict[str, Any]


__all__ = [
    "BBox",
    "Color",
    "Point",
    "PythagoreanDissectionPlan",
    "PythagoreanLegCase",
    "RenderContext",
    "RenderedPythagoreanDissectionScene",
]
