"""Passive state for rectangular-solid scene construction and rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image, ImageDraw

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class CuboidMeasureProblem:
    """Resolved rectangular-prism measurement problem."""

    target_role: str
    length: int
    width: int
    height: int
    volume: int
    surface_area: int
    answer: int
    formula_family: str
    formula: str
    case_probabilities: Dict[str, float]
    answer_support_probabilities: Dict[str, float]


@dataclass(frozen=True)
class CubeFrameProblem:
    """Resolved cube wire-frame edge problem."""

    frame_mode: str
    cube_edge: int
    visible_frame_edge_count: int
    frame_length: int
    answer: int
    formula_family: str
    formula: str
    case_probabilities: Dict[str, float]
    answer_support_probabilities: Dict[str, float]


@dataclass(frozen=True)
class OpenBoxNetProblem:
    """Resolved corner-cut open-box net problem."""

    target_role: str
    sheet_length: int
    sheet_width: int
    cut_size: int
    base_length: int
    base_width: int
    open_box_volume: int
    answer: int
    formula_family: str
    formula: str
    case_probabilities: Dict[str, float]
    answer_support_probabilities: Dict[str, float]


@dataclass
class RenderContext:
    """Mutable drawing context with sampled style state."""

    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    line_color: Color
    secondary_color: Color
    label_color: Color
    label_stroke_color: Color
    label_backing_color: Color
    face_front: Color
    face_side: Color
    face_top: Color
    accent_color: Color
    muted_color: Color
    line_width: int
    label_stroke_width: int
    font: Any
    small_font: Any
    diagram_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]


@dataclass(frozen=True)
class RenderedRectangularSolidScene:
    """Rendered image and projected scene witnesses."""

    image: Image.Image
    annotation_type: str
    annotation_keyed_points: Dict[str, Point]
    annotation_keyed_bboxes: Dict[str, BBox]
    annotation_roles: Tuple[str, ...]
    label_bboxes: Dict[str, BBox]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


__all__ = [
    "BBox",
    "Color",
    "CubeFrameProblem",
    "CuboidMeasureProblem",
    "OpenBoxNetProblem",
    "Point",
    "RenderContext",
    "RenderedRectangularSolidScene",
]
