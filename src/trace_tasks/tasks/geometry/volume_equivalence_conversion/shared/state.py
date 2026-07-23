"""Passive state containers for volume-equivalence conversion scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image, ImageDraw

Point = tuple[float, float]
BBox = tuple[float, float, float, float]
Color = tuple[int, int, int]


@dataclass(frozen=True)
class SolidSpec:
    shape: str
    base_area: int = 0
    height: int = 0
    length: int = 0
    width: int = 0
    depth: int = 0


@dataclass(frozen=True)
class OptionSpec:
    label: str
    solid: SolidSpec
    volume: int


@dataclass(frozen=True)
class ResolvedProblem:
    source: SolidSpec
    target: SolidSpec
    answer: int | str
    answer_schema: str
    formula_family: str
    formula: str
    target_unknown_role: str
    option_specs: tuple[OptionSpec, ...] = ()
    selected_option_label: str = ""
    case_probabilities: dict[str, float] = field(default_factory=dict)
    answer_support_probabilities: dict[str, float] = field(default_factory=dict)
    option_count_probabilities: dict[str, float] = field(default_factory=dict)


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
    option_fill: Color
    accent_color: Color
    muted_color: Color
    line_width: int
    label_stroke_width: int
    font: Any
    small_font: Any
    diagram_style_meta: dict[str, Any]
    background_meta: dict[str, Any]


@dataclass(frozen=True)
class RenderedScene:
    image: Image.Image
    annotation_bboxes: dict[str, BBox]
    label_bboxes: dict[str, BBox]
    scene_entities: tuple[dict[str, Any], ...]
    render_map: dict[str, Any]
    render_meta: dict[str, Any]


__all__ = [
    "BBox",
    "Color",
    "OptionSpec",
    "Point",
    "RenderContext",
    "RenderedScene",
    "ResolvedProblem",
    "SolidSpec",
]
