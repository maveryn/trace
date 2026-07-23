"""State contracts for the circle-centerline-overlap geometry scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple


SCENE_ID = "circle_centerline_overlap"

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]

CENTER_LABELS: tuple[str, str, str] = ("A", "B", "C")
BOUNDARY_POINT_LABELS: tuple[str, str, str, str] = ("P", "Q", "R", "S")
LABEL_MODES: tuple[str, ...] = ("radius",)
SUPPORTED_CIRCLE_COUNTS: tuple[int, int] = (2, 3)
DEFAULT_CIRCLE_COUNT_WEIGHTS: dict[str, float] = {"2": 0.75, "3": 0.25}
BOUNDARY_PAIRS: tuple[str, str] = ("AB", "BC")
BOUNDARY_TARGET_ROLES: tuple[str, str] = (
    "left_center_to_right_boundary",
    "left_boundary_to_right_center",
)


@dataclass(frozen=True)
class CircleOverlapCase:
    """One valid chain of two or three collinear overlapping circles."""

    radius_a: int
    radius_b: int
    radius_c: int
    overlap_ab: int
    overlap_bc: int

    @property
    def circle_count(self) -> int:
        return 2 if int(self.radius_c) <= 0 and int(self.overlap_bc) <= 0 else 3

    @property
    def distance_ab(self) -> int:
        return int(self.radius_a) + int(self.radius_b) - int(self.overlap_ab)

    @property
    def distance_bc(self) -> int:
        if self.circle_count == 2:
            return 0
        return int(self.radius_b) + int(self.radius_c) - int(self.overlap_bc)

    @property
    def distance_ac(self) -> int:
        if self.circle_count == 2:
            return int(self.distance_ab)
        return int(self.distance_ab) + int(self.distance_bc)

    @property
    def key(self) -> str:
        if self.circle_count == 2:
            return f"ra{self.radius_a}_rb{self.radius_b}_oab{self.overlap_ab}"
        return (
            f"ra{self.radius_a}_rb{self.radius_b}_rc{self.radius_c}"
            f"_oab{self.overlap_ab}_obc{self.overlap_bc}"
        )


@dataclass(frozen=True)
class CenterlineOverlapDiagramSpec:
    """Task-owned semantic request passed into neutral scene rendering."""

    case: CircleOverlapCase
    label_mode: str
    target_name: str
    known_segment_name: str
    known_segment_value: int
    target_segment_points: tuple[str, str]
    known_segment_points: tuple[str, str]
    show_overlap_dimensions: bool
    annotation_roles: tuple[str, ...]


@dataclass(frozen=True)
class RenderedCenterlineOverlapScene:
    """Rendered scene geometry and projected witness positions."""

    image: object
    annotation_keyed_points: Dict[str, Point]
    annotation_roles: tuple[str, ...]
    label_bboxes: Dict[str, BBox]
    scene_entities: tuple[dict[str, object], ...]
    render_map: Dict[str, object]


@dataclass
class CenterlineOverlapRenderContext:
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
    fill_colors: tuple[Color, Color, Color]
    accent_color: Color
    line_width: int
    label_stroke_width: int
    font: object
    small_font: object
    diagram_style_meta: Dict[str, object]
    background_meta: Dict[str, object]
    scene_transform: object


__all__ = [
    "BOUNDARY_PAIRS",
    "BOUNDARY_POINT_LABELS",
    "BOUNDARY_TARGET_ROLES",
    "BBox",
    "CENTER_LABELS",
    "CenterlineOverlapDiagramSpec",
    "CenterlineOverlapRenderContext",
    "CircleOverlapCase",
    "Color",
    "DEFAULT_CIRCLE_COUNT_WEIGHTS",
    "LABEL_MODES",
    "Point",
    "RenderedCenterlineOverlapScene",
    "SCENE_ID",
    "SUPPORTED_CIRCLE_COUNTS",
]
