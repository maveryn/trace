"""Passive state containers for Pythagorean tree rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform

Point = tuple[float, float]
BBox = tuple[float, float, float, float]
Color = tuple[int, int, int]
Polygon = tuple[Point, ...]


@dataclass(frozen=True)
class PythagoreanTreeTriple:
    """One integer right-triangle case for attached-square area problems."""

    leg_a: int
    leg_b: int
    hypotenuse: int

    @property
    def leg_square_1_area(self) -> int:
        return int(self.leg_a) * int(self.leg_a)

    @property
    def leg_square_2_area(self) -> int:
        return int(self.leg_b) * int(self.leg_b)

    @property
    def hypotenuse_square_area(self) -> int:
        return int(self.hypotenuse) * int(self.hypotenuse)


@dataclass(frozen=True)
class PythagoreanTreePlan:
    """Task-bound semantic construction passed to rendering."""

    triple: PythagoreanTreeTriple
    target_role: str
    answer: int
    known_area_labels: dict[str, str]
    witness: dict[str, Any]


@dataclass
class RenderContext:
    """Resolved canvas, style, and font state for one render attempt."""

    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    line_color: Color
    secondary_color: Color
    label_color: Color
    label_stroke_color: Color
    triangle_fill: Color
    leg_square_fill: Color
    other_leg_square_fill: Color
    hypotenuse_square_fill: Color
    accent_color: Color
    line_width: int
    label_stroke_width: int
    font: Any
    small_font: Any
    readout_text_metadata: dict[str, Any]
    diagram_style_meta: dict[str, Any]
    background_meta: dict[str, Any]
    font_meta: dict[str, Any]
    scene_transform: LazySceneTransform


@dataclass(frozen=True)
class RenderedPythagoreanTreeScene:
    """Pixel-space output from scene rendering before public annotation binding."""

    image: Image.Image
    square_polygons: dict[str, Polygon]
    square_bboxes: dict[str, BBox]
    triangle_vertices: dict[str, Point]
    label_bboxes: dict[str, BBox]
    marker_bboxes: dict[str, BBox]
    render_map: dict[str, Any]
    witness: dict[str, Any]
    scene_entities: tuple[dict[str, Any], ...]


__all__ = [
    "BBox",
    "Color",
    "Point",
    "Polygon",
    "PythagoreanTreePlan",
    "PythagoreanTreeTriple",
    "RenderContext",
    "RenderedPythagoreanTreeScene",
]
