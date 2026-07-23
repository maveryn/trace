"""State records for survey-traverse scene rendering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Tuple

from PIL import Image, ImageDraw

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]

DOMAIN = "geometry"
SCENE_ID = "survey_traverse"
SCENE_KIND = "geometry_survey_traverse"
DEGREE_SYMBOL = chr(176)


@dataclass(frozen=True)
class BearingTurnCase:
    """Closed-traverse turn construction values selected by the public bearing task."""

    answer: int
    base_bearing: int
    turn_angle: int
    turn_direction: str
    station_labels: Tuple[str, str, str]
    bearing_probabilities: Dict[str, float]
    turn_probabilities: Dict[str, float]


@dataclass(frozen=True)
class ElevationLevelingCase:
    """Leveling-note construction values selected by the public elevation task."""

    answer: int
    reference_elevation: int
    backsight: int
    foresight: int
    height_of_instrument: int
    station_labels: Tuple[str, str, str]
    case_probabilities: Dict[str, float]


@dataclass(frozen=True)
class AreaOffsetCase:
    """Offset-trapezoid construction values selected by the public area task."""

    answer: int
    station_labels: Tuple[str, str, str, str]
    chainages: Tuple[int, ...]
    offsets: Tuple[int, ...]
    case_probabilities: Dict[str, float]


@dataclass
class RenderContext:
    """Prepared image, fonts, and style values shared by scene renderers."""

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
    accent_color: Color
    secondary_accent_color: Color
    line_width: int
    label_stroke_width: int
    font: Any
    small_font: Any
    tiny_font: Any
    diagram_style_meta: Dict[str, Any]
    background_meta: Dict[str, Any]


@dataclass(frozen=True)
class RenderedAreaScene:
    """Rendered scene with role-bound bbox-map annotation witnesses."""

    image: Image.Image
    annotation_bboxes: Dict[str, BBox]
    annotation_roles: Tuple[str, ...]
    scene_entities: Tuple[Dict[str, Any], ...]
    label_bboxes: Dict[str, BBox]
    render_map: Dict[str, Any]
    witness: Dict[str, Any]


__all__ = [
    "AreaOffsetCase",
    "BBox",
    "BearingTurnCase",
    "Color",
    "DEGREE_SYMBOL",
    "DOMAIN",
    "ElevationLevelingCase",
    "Point",
    "RenderContext",
    "RenderedAreaScene",
    "SCENE_ID",
    "SCENE_KIND",
]
