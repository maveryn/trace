"""State containers for measuring-tool rendering primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Tuple

from PIL import Image, ImageDraw

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class LengthMeasurementPlan:
    """Scene-neutral length measurement parameters selected by a public task."""

    measurement_kind: str
    shape_kind: str
    target_length_cm: int
    ruler_start_cm: int
    ruler_max_cm: int
    answer_probabilities: Mapping[str, float]


@dataclass(frozen=True)
class AngleMeasurementPlan:
    """Scene-neutral angle measurement parameters selected by a public task."""

    measurement_kind: str
    shape_kind: str
    target_angle_degrees: int
    answer_probabilities: Mapping[str, float]


@dataclass
class RenderContext:
    """Rendering context shared by measuring-tool scene drawers."""

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


@dataclass(frozen=True)
class RenderedToolScene:
    """Rendered measuring-tool scene plus verifier payload fragments."""

    image: Image.Image
    answer: int
    annotation_points: Mapping[str, list[float]]
    scene_entities: tuple[dict[str, Any], ...]
    render_map: Mapping[str, Any]
    witness: Mapping[str, Any]


__all__ = [
    "AngleMeasurementPlan",
    "BBox",
    "Color",
    "LengthMeasurementPlan",
    "Point",
    "RenderContext",
    "RenderedToolScene",
]
