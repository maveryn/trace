"""State records for polygon equation diagrams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from PIL import Image, ImageDraw

Point = tuple[float, float]
BBox = tuple[float, float, float, float]
Color = tuple[int, int, int]


@dataclass(frozen=True)
class PolygonEquationCase:
    """One task-bound algebraic polygon measurement construction."""

    side_count: int
    answer: int
    target_name: str
    variable_name: str
    formula_schema: str
    relation: str
    output_role: str
    side_labels: Mapping[str, str] = field(default_factory=dict)
    angle_labels: Mapping[str, str] = field(default_factory=dict)
    side_mark_counts: Mapping[str, int] = field(default_factory=dict)
    angle_mark_counts: Mapping[str, int] = field(default_factory=dict)
    equal_sides: tuple[str, ...] = ()
    equal_angles: tuple[str, ...] = ()
    target_side: str = ""
    target_angle: str = ""
    witness: Mapping[str, Any] = field(default_factory=dict)

    def vertex_labels(self) -> tuple[str, ...]:
        return tuple(chr(ord("A") + index) for index in range(int(self.side_count)))

    def trace_fields(self) -> dict[str, Any]:
        return {
            "side_count": int(self.side_count),
            "polygon_kind": polygon_kind(int(self.side_count)),
            "interior_angle_sum": int((int(self.side_count) - 2) * 180),
            "target_name": str(self.target_name),
            "variable_name": str(self.variable_name),
            "formula_schema": str(self.formula_schema),
            "relation": str(self.relation),
            "output_role": str(self.output_role),
            "side_labels": dict(self.side_labels),
            "angle_labels": dict(self.angle_labels),
            "side_mark_counts": {str(key): int(value) for key, value in self.side_mark_counts.items()},
            "angle_mark_counts": {str(key): int(value) for key, value in self.angle_mark_counts.items()},
            "equal_sides": list(self.equal_sides),
            "equal_angles": list(self.equal_angles),
            "target_side": str(self.target_side),
            "target_angle": str(self.target_angle),
            "answer_value": int(self.answer),
            **dict(self.witness),
        }


@dataclass
class RenderContext:
    """Resolved style and PIL drawing state for one diagram."""

    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    line_color: Color
    secondary_color: Color
    label_color: Color
    label_stroke_color: Color
    fill_color: Color
    accent_color: Color
    line_width: int
    label_stroke_width: int
    font: Any
    small_font: Any
    diagram_style_meta: dict[str, Any]
    background_meta: dict[str, Any]
    scene_transform: Any


@dataclass(frozen=True)
class RenderedPolygonEquationScene:
    """Rendered polygon equation diagram and projected witnesses."""

    image: Image.Image
    annotation_keyed_points: dict[str, Point]
    annotation_roles: tuple[str, ...]
    vertex_points: dict[str, Point]
    point_label_bboxes: dict[str, BBox]
    side_label_bboxes: dict[str, BBox]
    angle_label_bboxes: dict[str, BBox]
    marker_bboxes: dict[str, BBox]
    vertices: tuple[Point, ...]


def polygon_kind(side_count: int) -> str:
    """Return a prompt-friendly polygon kind for a side count."""

    names = {
        3: "triangle",
        4: "quadrilateral",
        5: "pentagon",
        6: "hexagon",
    }
    return names.get(int(side_count), f"{int(side_count)}-gon")


__all__ = [
    "BBox",
    "Color",
    "Point",
    "PolygonEquationCase",
    "RenderContext",
    "RenderedPolygonEquationScene",
    "polygon_kind",
]
