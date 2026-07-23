"""Typed scene state for bearing-route diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image, ImageDraw


Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]

SCENE_ID = "bearing_route"
DEGREE_SYMBOL = chr(176)


@dataclass(frozen=True)
class RouteCase:
    leg_a: int
    leg_b: int
    displacement: int
    bearing_a: int
    bearing_b: int
    turn_direction: str
    option_count: int
    target_index: int | None
    option_labels: Tuple[str, ...]
    option_values: Tuple[int, ...] = ()
    final_bearing: int | None = None


@dataclass
class RenderContext:
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
    route_style_id: str
    marker_style: str


@dataclass(frozen=True)
class RenderedBearingScene:
    image: Image.Image
    answer: int | str
    answer_type: str
    annotation_bboxes: Tuple[BBox, ...]
    annotation_roles: Tuple[str, ...]
    annotation_points: Tuple[Point, ...]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    witness: Dict[str, Any]


__all__ = [
    "BBox",
    "Color",
    "DEGREE_SYMBOL",
    "Point",
    "RenderContext",
    "RenderedBearingScene",
    "RouteCase",
    "SCENE_ID",
]
