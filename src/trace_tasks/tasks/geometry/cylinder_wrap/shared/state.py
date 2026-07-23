"""Passive state objects for cylinder-wrap rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class SurfacePathProblem:
    """One unwrapped-cylinder path-length construction."""

    circumference: int
    height: int
    path_length: int


@dataclass(frozen=True)
class WrappedMarkProblem:
    """One strip-to-rim position matching construction."""

    option_count: int
    target_index: int
    option_labels: Tuple[str, ...]
    answer_label: str


@dataclass
class RenderContext:
    """Mutable PIL drawing context with resolved style."""

    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    line_color: Color
    secondary_color: Color
    guide_color: Color
    label_color: Color
    label_stroke_color: Color
    panel_fill: Color
    panel_alt_fill: Color
    panel_border: Color
    accent_color: Color
    secondary_accent_color: Color
    line_width: int
    font: Any
    small_font: Any
    tiny_font: Any
    font_family: str
    wrap_style_id: str
    marker_style_id: str


@dataclass(frozen=True)
class RenderedCylinderWrapScene:
    """Rendered cylinder-wrap scene plus public visual witnesses."""

    image: Image.Image
    answer: int | str
    answer_type: str
    annotation_type: str
    annotation_value: Mapping[str, Any]
    annotation_roles: Tuple[str, ...]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    witness: Dict[str, Any]


__all__ = [
    "BBox",
    "Color",
    "Point",
    "RenderContext",
    "RenderedCylinderWrapScene",
    "SurfacePathProblem",
    "WrappedMarkProblem",
]
