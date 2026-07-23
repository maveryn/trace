"""Passive state for container volume-transfer diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image, ImageDraw

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class ResolvedProblem:
    objective: str
    diagram_mode: str
    source_shape: str
    target_shape: str
    source_base_area: int
    source_height: int
    source_volume: int
    target_base_area: int
    target_height: int
    target_length: int
    target_width: int
    target_volume: int
    fill_count: int
    pour_count: int
    resulting_height: float
    answer: int | float
    formula_family: str
    formula: str
    query_probabilities: Dict[str, float]
    case_probabilities: Dict[str, float]
    answer_support_probabilities: Dict[str, float]


@dataclass
class RenderContext:
    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    line_color: Color
    secondary_color: Color
    label_color: Color
    label_stroke_color: Color
    source_fill: Color
    target_fill: Color
    liquid_fill: Color
    accent_color: Color
    muted_color: Color
    panel_fill_color: Color
    line_width: int
    label_stroke_width: int
    font: Any
    small_font: Any
    diagram_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]


@dataclass(frozen=True)
class RenderedScene:
    image: Image.Image
    annotation_bboxes: Dict[str, BBox]
    label_bboxes: Dict[str, BBox]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]


__all__ = ["BBox", "Color", "Point", "RenderContext", "RenderedScene", "ResolvedProblem"]
