"""State contracts for solid-revolution diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from PIL import Image, ImageDraw

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class SolidRevolutionProblem:
    """Task-bound measurements and formula metadata for one revolution diagram."""

    solid_kind: str
    generating_shape: str
    answer: float
    formula_family: str
    formula: str
    radius: float | None = None
    diameter: float | None = None
    radial_input_kind: str | None = None
    height: float | None = None
    slant_height: float | None = None
    diagonal: float | None = None
    half_height: float | None = None
    top_radius: float | None = None
    bottom_radius: float | None = None
    total_height: float | None = None
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
    solid_fill_color: Color
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
class RenderedSolidRevolutionScene:
    """Rendered revolution diagram and projected witness positions."""

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
    "RenderedSolidRevolutionScene",
    "SolidRevolutionProblem",
]
