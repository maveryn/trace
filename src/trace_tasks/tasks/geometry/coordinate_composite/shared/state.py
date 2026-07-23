"""State contracts for coordinate-composite diagrams."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Tuple

from PIL import Image

GraphPoint = Tuple[float, float]
PixelPoint = Tuple[float, float]
Color = Tuple[int, int, int]


class PairFilter(Enum):
    """Semantic object-pair family used by the public task query mapping."""

    LINE_CIRCLE = "line_circle"
    CIRCLE_CIRCLE = "circle_circle"
    LINE_POLYGON = "line_polygon"
    CIRCLE_POLYGON = "circle_polygon"
    ALL = "all"


@dataclass(frozen=True)
class LineObject:
    object_id: str
    p0: GraphPoint
    p1: GraphPoint


@dataclass(frozen=True)
class CircleObject:
    object_id: str
    center: GraphPoint
    radius: float


@dataclass(frozen=True)
class PolygonObject:
    object_id: str
    vertices: Tuple[GraphPoint, ...]


SceneObject = LineObject | CircleObject | PolygonObject


@dataclass(frozen=True)
class RenderedScene:
    """Rendered diagram plus projected intersection witnesses."""

    image: Image.Image
    intersection_points_px: Tuple[PixelPoint, ...]
    intersection_points_graph: Tuple[GraphPoint, ...]
    object_specs: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    render_spec_extra: Dict[str, Any]
    candidate_points_px: Tuple[PixelPoint, ...] = tuple()
    candidate_points_graph: Tuple[GraphPoint, ...] = tuple()
    candidate_point_labels: Tuple[str, ...] = tuple()
    candidate_marker_bboxes: Dict[str, Any] = field(default_factory=dict)
    candidate_label_bboxes: Dict[str, Any] = field(default_factory=dict)


__all__ = [
    "CircleObject",
    "Color",
    "GraphPoint",
    "LineObject",
    "PairFilter",
    "PixelPoint",
    "PolygonObject",
    "RenderedScene",
    "SceneObject",
]
