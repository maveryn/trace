"""State contracts for the circle-polygon-composite geometry scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

SCENE_ID = "circle_polygon_composite"

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]

SIDE_KEYS: Tuple[str, ...] = ("AB", "BC", "CD", "DA")
CONSTRUCTION_KINDS: Tuple[str, ...] = ("incircle", "semicircle")

TANGENTIAL_ANNOTATION_KEYS: Tuple[str, ...] = (
    "A",
    "B",
    "C",
    "D",
)

ANGLE_ANNOTATION_KEYS: Tuple[str, ...] = (
    "A",
    "B",
    "C",
    "D",
    "O",
    "T",
)


@dataclass(frozen=True)
class TangentialDiagramSpec:
    """Visual grammar inputs for a tangential-quadrilateral diagram."""

    vertex_tangents: Dict[str, int]
    side_lengths: Dict[str, int]
    unknown_sides: Tuple[str, ...]
    annotation_roles: Tuple[str, ...] = TANGENTIAL_ANNOTATION_KEYS


@dataclass(frozen=True)
class AngleDiagramSpec:
    """Visual grammar inputs for a square/incircle or rectangle/semicircle tangent-angle diagram."""

    construction_kind: str
    angle_degrees: int
    side_sign: int
    annotation_roles: Tuple[str, ...] = ANGLE_ANNOTATION_KEYS


@dataclass
class CirclePolygonRenderContext:
    """Mutable PIL drawing context with resolved scene style."""

    image: Any
    draw: Any
    width: int
    height: int
    line_color: Color
    secondary_color: Color
    label_color: Color
    label_stroke_color: Color
    label_backing_color: Color
    polygon_fill: Color
    circle_fill: Color
    accent_color: Color
    line_width: int
    label_stroke_width: int
    font: Any
    small_font: Any
    diagram_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]
    scene_transform: Any


@dataclass(frozen=True)
class RenderedTangentialScene:
    """Rendered tangential quadrilateral and projected witness positions."""

    image: Any
    annotation_keyed_points: Dict[str, Point]
    annotation_roles: Tuple[str, ...]
    vertices: Dict[str, Point]
    tangency_points: Dict[str, Point]
    label_bboxes: Dict[str, BBox]
    render_map: Dict[str, Any]


@dataclass(frozen=True)
class RenderedAngleScene:
    """Rendered tangent-angle construction and projected witness positions."""

    image: Any
    annotation_keyed_points: Dict[str, Point]
    annotation_roles: Tuple[str, ...]
    label_bboxes: Dict[str, BBox]
    render_map: Dict[str, Any]


__all__ = [
    "ANGLE_ANNOTATION_KEYS",
    "AngleDiagramSpec",
    "BBox",
    "CONSTRUCTION_KINDS",
    "CirclePolygonRenderContext",
    "Color",
    "Point",
    "RenderedAngleScene",
    "RenderedTangentialScene",
    "SCENE_ID",
    "SIDE_KEYS",
    "TANGENTIAL_ANNOTATION_KEYS",
    "TangentialDiagramSpec",
]
