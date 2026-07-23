"""Passive dataclasses for composite-shape geometry scene primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Tuple

from PIL import Image, ImageDraw

Point = Tuple[float, float]
BBox = Tuple[float, float, float, float]
Color = Tuple[int, int, int]


@dataclass
class CompositeRenderContext:
    """Resolved canvas, style, font, and drawing handles for one render."""

    rng: Any
    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    background_color: Color
    line_color: Color
    label_color: Color
    label_stroke_color: Color
    accent_color: Color
    fill_color: Color
    secondary_fill_color: Color
    line_width: int
    label_stroke_width: int
    font: Any
    small_font: Any
    scene_transform: Any | None = None


@dataclass(frozen=True)
class CompositeShapeProblem:
    """Task-bound answer, formula, and render facts supplied by a public file."""

    prompt_key: str
    shape_family: str
    metric_kind: str
    answer_value: int | float
    answer_type: str
    reasoning_kind: str
    scene_kind: str
    witness_type: str
    dimensions: Mapping[str, Any]
    formula_family: str
    reasoning_steps: int
    annotation_roles_hint: Tuple[str, ...] = ()
    prompt_slots: Mapping[str, Any] = field(default_factory=dict)
    metadata_fields: Mapping[str, Any] = field(default_factory=dict)
    execution_fields: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RenderedCompositeShape:
    """Rendered image with visual witness metadata after final layout."""

    image: Image.Image
    answer_value: int | float
    annotation_roles: Tuple[str, ...]
    scene_entities: Tuple[Dict[str, Any], ...]
    render_map: Dict[str, Any]
    witness: Dict[str, Any]
    annotation_keyed_bboxes: Mapping[str, BBox] | None = None
    annotation_keyed_points: Mapping[str, Point] | None = None
