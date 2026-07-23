"""State objects for triangle-relations analytical diagrams."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Tuple

from PIL import Image, ImageDraw

DOMAIN = "geometry"
SCENE_ID = "triangle_relations"
SCENE_KIND = "geometry_triangle_relations"
PROMPT_BUNDLE_ID = "geometry_triangle_relations_v1"
SCENE_PROMPT_KEY = "triangle_relations_scene"

Point = Tuple[float, float]
Segment = Tuple[Point, Point]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class SegmentLabel:
    """One visible side-length or target-segment label."""

    segment: tuple[str, str]
    text: str
    offset: float = 28.0
    role: str = ""
    placement: str = "auto"


@dataclass(frozen=True)
class AngleLabel:
    """One visible angle mark and optional angle readout."""

    vertex: str
    arm_a: str
    arm_b: str
    text: str
    radius: float = 48.0
    role: str = ""


@dataclass(frozen=True)
class RightAngleMark:
    """One right-angle square marker bound to three visible points."""

    vertex: str
    arm_a: str
    arm_b: str


@dataclass(frozen=True)
class TickGroup:
    """One set of equal-length tick marks or parallel marks."""

    segments: tuple[tuple[str, str], ...]
    count: int = 1
    kind: str = "equal"


@dataclass(frozen=True)
class TriangleRelationsCase:
    """Resolved numeric case and scene grammar for one task sample."""

    case_kind: str
    answer: int | float
    answer_type: str
    answer_rounding: str
    formula_family: str
    formula_text: str
    reasoning_steps: int
    vertices: Mapping[str, Point]
    edges: tuple[tuple[str, str], ...]
    polygons: tuple[tuple[str, ...], ...]
    segment_labels: tuple[SegmentLabel, ...]
    target_segment: tuple[str, str] | None = None
    target_point: str | None = None
    point_annotation_labels: tuple[str, ...] = ()
    point_mark_labels: tuple[str, ...] = ()
    angle_labels: tuple[AngleLabel, ...] = ()
    right_angles: tuple[RightAngleMark, ...] = ()
    tick_groups: tuple[TickGroup, ...] = ()
    filled_polygons: tuple[tuple[str, ...], ...] = ()
    trace_values: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TriangleRelationsProblem:
    """Task-owned objective data passed into scene rendering."""

    case: TriangleRelationsCase
    answer_support_probabilities: Mapping[str, float]
    prompt_target: str
    annotation_mode: str


@dataclass
class RenderContext:
    """Mutable PIL context plus sampled style for one render attempt."""

    rng: Any
    image: Image.Image
    draw: ImageDraw.ImageDraw
    width: int
    height: int
    line_color: Color
    label_color: Color
    label_stroke_color: Color
    accent_color: Color
    fill_color: Color
    alt_fill_color: Color
    line_width: int
    label_stroke_width: int
    readout_text_metadata: Mapping[str, Any]
    font: Any
    small_font: Any
    diagram_style_meta: Mapping[str, Any]
    background_meta: Mapping[str, Any]
    scene_transform: Any


@dataclass(frozen=True)
class RenderedTriangleRelationsScene:
    """Rendered triangle-relations image with projected witness geometry."""

    image: Image.Image
    answer: int | float
    annotation_mode: str
    annotation_segment: Segment | None
    annotation_point: Point | None
    annotation_points: Mapping[str, Point]
    annotation_roles: tuple[str, ...]
    scene_entities: tuple[dict[str, Any], ...]
    render_map: Mapping[str, Any]
    witness: Mapping[str, Any]
    reasoning_steps: int


__all__ = [
    "AngleLabel",
    "Color",
    "DOMAIN",
    "PROMPT_BUNDLE_ID",
    "Point",
    "RenderContext",
    "RenderedTriangleRelationsScene",
    "RightAngleMark",
    "SCENE_ID",
    "SCENE_KIND",
    "SCENE_PROMPT_KEY",
    "Segment",
    "SegmentLabel",
    "TickGroup",
    "TriangleRelationsCase",
    "TriangleRelationsProblem",
]
