"""State contracts for solid cross-section diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from PIL import Image, ImageDraw

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class ConeSliceCase:
    """One cone cross-section measurement case."""

    base_radius: int
    solid_height: int
    slice_distance_from_apex: int


@dataclass(frozen=True)
class PyramidSliceCase:
    """One square-pyramid cross-section measurement case."""

    base_side: int
    solid_height: int
    slice_distance_from_apex: int


@dataclass(frozen=True)
class SolidCrossSectionProblem:
    """Task-bound measurements and formula metadata for one diagram."""

    solid_kind: str
    answer: float
    formula_family: str
    formula: str
    solid_height: float
    slice_distance_from_apex: float
    base_radius: float | None = None
    base_side: float | None = None
    slice_radius: float | None = None
    slice_side: float | None = None
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
    slice_fill_color: Color
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
class RenderedSolidCrossSectionScene:
    """Rendered solid cross-section diagram and projected witness positions."""

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
    "ConeSliceCase",
    "Point",
    "PyramidSliceCase",
    "RenderContext",
    "RenderedSolidCrossSectionScene",
    "SolidCrossSectionProblem",
]
