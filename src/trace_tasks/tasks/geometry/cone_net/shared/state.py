"""State contracts for cone-sector net diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class ConeNetCase:
    """One sector-net measurement case."""

    slant_height: int
    theta_degrees: int


@dataclass(frozen=True)
class ConeNetDiagramSpec:
    """Task-bound labels and measurements for one cone-net diagram."""

    answer: float
    slant_height: int
    theta_degrees: int
    base_radius: float
    cone_height: float
    arc_length: float
    target_measure: str
    target_label: str
    target_label_anchor: str
    annotation_roles: Tuple[str, ...]
    formula_family: str
    reasoning_steps: int


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
    cone_fill_color: Color
    accent_color: Color
    line_width: int
    font: Any
    small_font: Any
    label_stroke_width: int


@dataclass(frozen=True)
class RenderedConeNetScene:
    """Rendered cone-net diagram and projected annotation positions."""

    image: Image.Image
    annotation_roles: Tuple[str, ...]
    annotation_keyed_points: Mapping[str, Point]
    label_bboxes: Dict[str, BBox]
    point_label_bboxes: Dict[str, BBox]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    measurements: Dict[str, Any]


__all__ = [
    "BBox",
    "Color",
    "ConeNetCase",
    "ConeNetDiagramSpec",
    "Point",
    "RenderContext",
    "RenderedConeNetScene",
]
