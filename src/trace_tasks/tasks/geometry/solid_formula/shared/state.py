"""State contracts for solid-formula diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from PIL import Image, ImageDraw

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class SolidFormulaProblem:
    """Task-bound measurements and formula metadata for one solid diagram."""

    solid_kind: str
    answer: float
    unknown_dimension: str
    formula_family: str
    formula: str
    radius: float | None = None
    total_height: float | None = None
    cylinder_height: float | None = None
    cone_height: float | None = None
    volume: float | None = None
    volume_pi_multiple: float | None = None
    side_a: float | None = None
    side_b: float | None = None
    prism_height: float | None = None
    pyramid_height: float | None = None
    triangle_base: float | None = None
    prism_length: float | None = None
    wall_height: float | None = None
    roof_height: float | None = None
    answer_support_probabilities: Mapping[str, float] | None = None
    construction_case_count_for_answer: int = 1


@dataclass
class RenderContext:
    """Mutable PIL drawing context with resolved style."""

    rng: Any
    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    line_color: Color
    label_color: Color
    label_stroke_color: Color
    fill_color: Color
    secondary_fill_color: Color
    accent_color: Color
    muted_color: Color
    line_width: int
    font: Any
    small_font: Any
    label_stroke_width: int
    diagram_style_meta: Mapping[str, Any]
    background_meta: Mapping[str, Any]
    font_meta: Mapping[str, Any]
    palette_meta: Mapping[str, Any]


@dataclass(frozen=True)
class RenderedSolidFormulaScene:
    """Rendered solid diagram and projected witness positions."""

    image: Image.Image
    annotation_bboxes: Mapping[str, BBox]
    annotation_roles: Tuple[str, ...]
    label_bboxes: Mapping[str, BBox]
    scene_entities: Tuple[dict[str, Any], ...]
    render_map: Mapping[str, Any]
    measurements: Mapping[str, Any]


__all__ = [
    "BBox",
    "Color",
    "Point",
    "RenderContext",
    "RenderedSolidFormulaScene",
    "SolidFormulaProblem",
]
