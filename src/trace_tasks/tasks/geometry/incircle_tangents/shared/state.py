"""State objects for incircle-tangent diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image, ImageDraw

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class TangentTriangleCase:
    """Tangent lengths from vertices A, B, and C."""

    tangent_a: int
    tangent_b: int
    tangent_c: int


@dataclass(frozen=True)
class IncircleDiagramSpec:
    """Task-owned semantic specification for one incircle diagram."""

    answer: int | float
    answer_type: str
    answer_rounding: str
    unknown_measure: str
    formula_family: str
    tangent_a: float
    tangent_b: float
    tangent_c: float
    side_ab: float
    side_bc: float
    side_ca: float
    semiperimeter: float
    area: float
    displayed_area: float
    inradius: float
    unknown_label: str
    show_area_label: bool
    show_radius_segment: bool
    annotation_roles: Tuple[str, ...]

    def measurement_fields(self) -> Dict[str, Any]:
        """Return JSON-ready measurements for trace payloads."""

        return {
            "formula_family": str(self.formula_family),
            "unknown_measure": str(self.unknown_measure),
            "side_ab": round(float(self.side_ab), 3),
            "side_bc": round(float(self.side_bc), 3),
            "side_ca": round(float(self.side_ca), 3),
            "tangent_a": round(float(self.tangent_a), 3),
            "tangent_b": round(float(self.tangent_b), 3),
            "tangent_c": round(float(self.tangent_c), 3),
            "semiperimeter": round(float(self.semiperimeter), 3),
            "area": round(float(self.displayed_area), 3),
            "exact_area": round(float(self.area), 6),
            "inradius": round(float(self.inradius), 3),
            "answer_value": self.answer,
        }


@dataclass
class RenderContext:
    """Concrete render resources for one image."""

    rng: Any
    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    line_color: Color
    label_color: Color
    label_stroke_color: Color
    accent_color: Color
    fill_color: Color
    line_width: int
    label_stroke_width: int
    font: Any
    small_font: Any
    scene_transform: Any


@dataclass(frozen=True)
class RenderedIncircleScene:
    """Rendered image and projected scene witnesses."""

    image: Image.Image
    label_bboxes: Dict[str, BBox]
    annotation_points: Dict[str, Point]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    annotation_roles: Tuple[str, ...]


__all__ = [
    "BBox",
    "Color",
    "IncircleDiagramSpec",
    "Point",
    "RenderContext",
    "RenderedIncircleScene",
    "TangentTriangleCase",
]
