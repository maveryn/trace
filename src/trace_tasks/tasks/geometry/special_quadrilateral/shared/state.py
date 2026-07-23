"""State types for special-quadrilateral theorem diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform

DOMAIN = "geometry"
SCENE_ID = "special_quadrilateral"
DEGREE_SYMBOL = chr(176)

Point = tuple[float, float]
BBox = tuple[float, float, float, float]
Color = tuple[int, int, int]


@dataclass(frozen=True)
class LinearExpression:
    """Linear expression printed on one visible angle or segment label."""

    coefficient: int
    constant: int

    def evaluate(self, x_value: int) -> int:
        return int(self.coefficient) * int(x_value) + int(self.constant)

    def display(self, *, degree: bool = False) -> str:
        coefficient = int(self.coefficient)
        constant = int(self.constant)
        if coefficient == 1:
            body = "x"
        elif coefficient == -1:
            body = "-x"
        else:
            body = f"{coefficient}x"
        if constant > 0:
            body = f"{body}+{constant}"
        elif constant < 0:
            body = f"{body}{constant}"
        if degree:
            return f"({body}){DEGREE_SYMBOL}"
        return body


@dataclass(frozen=True)
class QuadrilateralCase:
    """One task-bound theorem case, independent of public task identity."""

    render_kind: str
    shape_kind: str
    answer: int
    target_name: str
    target_label: str
    support_label: str
    theorem: str
    x_value: int | None = None
    target_expression: LinearExpression | None = None
    support_expression: LinearExpression | None = None


@dataclass(frozen=True)
class SpecialQuadrilateralProblem:
    """Resolved construction case and layout seed for one generated diagram."""

    case: QuadrilateralCase
    case_index: int
    layout_seed: int


@dataclass
class RenderContext:
    """Pillow drawing context plus sampled geometry style and transform."""

    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    line_color: Color
    secondary_color: Color
    label_color: Color
    label_stroke_color: Color
    accent_color: Color
    muted_color: Color
    fill_color: Color
    line_width: int
    label_stroke_width: int
    font: Any
    small_font: Any
    diagram_style_meta: dict[str, Any]
    background_meta: dict[str, Any]
    scene_transform: LazySceneTransform


@dataclass(frozen=True)
class RenderedSpecialQuadrilateralScene:
    """Rendered image and projected point/bbox metadata."""

    image: Image.Image
    vertices: dict[str, Point]
    annotation_points: dict[str, Point]
    point_label_bboxes: dict[str, BBox]
    readout_bboxes: dict[str, BBox]
    construction_bboxes: dict[str, BBox]
    render_map: dict[str, Any]


__all__ = [
    "BBox",
    "Color",
    "DEGREE_SYMBOL",
    "DOMAIN",
    "LinearExpression",
    "Point",
    "QuadrilateralCase",
    "RenderContext",
    "RenderedSpecialQuadrilateralScene",
    "SCENE_ID",
    "SpecialQuadrilateralProblem",
]
