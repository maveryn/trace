"""State contracts for the geometry angle-relations scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Tuple

from PIL import Image

from .spatial_primitives import (
    BBox,
    Point,
    RenderContext,
)

SCENE_ID = "angle_relations"
DOMAIN = "geometry"

@dataclass(frozen=True)
class RenderedAngleRelationScene:
    """Rendered angle-relations diagram plus scene-local metadata."""

    image: Image.Image
    answer: int
    annotation_bboxes: Tuple[BBox, ...]
    annotation_roles: Tuple[str, ...]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    witness: Dict[str, Any]
    reasoning_steps: int
    annotation_keyed_points: Mapping[str, Point] | None = None
    annotation_keyed_bboxes: Mapping[str, BBox] | None = None


@dataclass(frozen=True)
class AngleRelationCase:
    """One constructively valid angle-relations diagram case."""

    answer: int
    build: Callable[[RenderContext], RenderedAngleRelationScene]


ANGLE_ABC = "ABC"
ANGLE_BAC = "BAC"
ANGLE_BCD = "BCD"
ANGLE_AEF = "AEF"
ANGLE_CFE = "CFE"
POINT_A = "A"
POINT_B = "B"
POINT_C = "C"
POINT_D = "D"

__all__ = [
    "ANGLE_AEF",
    "ANGLE_ABC",
    "ANGLE_BAC",
    "ANGLE_BCD",
    "ANGLE_CFE",
    "AngleRelationCase",
    "BBox",
    "DOMAIN",
    "Point",
    "POINT_A",
    "POINT_B",
    "POINT_C",
    "POINT_D",
    "RenderedAngleRelationScene",
    "SCENE_ID",
]
