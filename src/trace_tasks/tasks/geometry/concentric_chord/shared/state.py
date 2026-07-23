"""State contracts for the concentric-chord geometry scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]

ANNOTATION_KEYS: tuple[str, str, str, str] = ("O", "A", "B", "T")


@dataclass(frozen=True)
class ConcentricChordCase:
    """One integer right-triangle case for a tangent chord."""

    outer_radius: int
    inner_radius: int
    half_chord: int

    @property
    def chord_length(self) -> int:
        return 2 * int(self.half_chord)


@dataclass(frozen=True)
class ConcentricChordDiagramSpec:
    """Task-bound labels and measurements for one rendered diagram."""

    answer: int
    outer_radius: int
    inner_radius: int
    half_chord: int
    chord_length: int
    outer_radius_label: str
    inner_radius_label: str
    chord_label: str
    formula_family: str
    unknown_measure: str


@dataclass
class RenderContext:
    """Mutable PIL drawing context with resolved geometry style."""

    rng: Any
    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    line_color: Color
    label_color: Color
    label_stroke_color: Color
    accent_color: Color
    line_width: int
    font: Any
    small_font: Any
    label_stroke_width: int
    scene_transform: Any


@dataclass(frozen=True)
class RenderedConcentricChordScene:
    """Rendered concentric-chord diagram and projected witness positions."""

    image: Image.Image
    annotation_roles: Tuple[str, ...]
    annotation_keyed_points: Mapping[str, Point]
    label_bboxes: Dict[str, BBox]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    measurements: Dict[str, Any]


__all__ = [
    "ANNOTATION_KEYS",
    "BBox",
    "Color",
    "ConcentricChordCase",
    "ConcentricChordDiagramSpec",
    "Point",
    "RenderContext",
    "RenderedConcentricChordScene",
]
