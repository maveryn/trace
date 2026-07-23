"""State contracts for the circle-pair-tangents geometry scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

SCENE_ID = "circle_pair_tangents"

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]

ANNOTATION_KEYS: tuple[str, str, str, str] = ("C", "D", "A", "B")
LARGER_CIRCLE_SIDES: tuple[str, str] = ("left", "right")
TANGENT_SIDES: tuple[str, str] = ("above", "below")


@dataclass(frozen=True)
class TangentCase:
    """One integer external-common-tangent right-triangle case."""

    small_radius: int
    large_radius: int
    center_distance: int
    tangent_length: int

    @property
    def radius_difference(self) -> int:
        return int(self.large_radius) - int(self.small_radius)

    @property
    def key(self) -> str:
        return (
            f"r{int(self.small_radius)}-{int(self.large_radius)}"
            f"_d{int(self.center_distance)}_t{int(self.tangent_length)}"
        )


@dataclass(frozen=True)
class PairTangentDiagramSpec:
    """Task-owned semantic request passed into neutral scene rendering."""

    radius_o1: int
    radius_o2: int
    center_distance: int
    tangent_length: int
    larger_circle_side: str
    tangent_side: str
    center_segment_label: str
    tangent_segment_label: str
    annotation_roles: tuple[str, ...] = ANNOTATION_KEYS


@dataclass(frozen=True)
class RenderedPairTangentScene:
    """Rendered tangent diagram geometry and projected witness positions."""

    image: object
    annotation_keyed_points: Dict[str, Point]
    annotation_roles: tuple[str, ...]
    label_bboxes: Dict[str, BBox]
    scene_entities: tuple[dict[str, object], ...]
    render_map: Dict[str, object]


@dataclass
class PairTangentRenderContext:
    """Mutable PIL drawing context with resolved scene style."""

    image: object
    draw: object
    width: int
    height: int
    line_color: Color
    secondary_color: Color
    label_color: Color
    label_stroke_color: Color
    label_backing_color: Color
    accent_color: Color
    line_width: int
    label_stroke_width: int
    font: object
    small_font: object
    diagram_style_meta: Dict[str, object]
    background_meta: Dict[str, object]
    scene_transform: object


__all__ = [
    "ANNOTATION_KEYS",
    "BBox",
    "Color",
    "LARGER_CIRCLE_SIDES",
    "PairTangentDiagramSpec",
    "PairTangentRenderContext",
    "Point",
    "RenderedPairTangentScene",
    "SCENE_ID",
    "TANGENT_SIDES",
    "TangentCase",
]
