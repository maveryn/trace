"""State records for the graph-paper geometry scene."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

SCENE_ID = "graph_paper"
LABEL_POOL = tuple("ABCDEFGHIJKL")
POINT_LABELS = tuple("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

Color = tuple[int, int, int]
Point = tuple[float, float]
BBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class GraphPaperContext:
    """Rendered coordinate-grid context shared by all graph-paper objectives."""

    image: Image.Image
    draw: ImageDraw.ImageDraw
    canvas_size: int
    graph_cells: int
    panel_box: BBox
    content_box: BBox
    origin_px: Point
    spacing_px: float
    graph_half_range: int
    ink_color: Color
    accent_color: Color
    grid_color: Color
    axis_color: Color
    label_color: Color
    label_stroke_color: Color
    shape_fill_color: Color
    object_colors: tuple[Color, ...]
    background_meta: Mapping[str, Any]
    shape_style_meta: Mapping[str, Any] = field(default_factory=dict)
    graph_layout_meta: Mapping[str, Any] = field(default_factory=dict)
    post_noise_meta: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphObject:
    """One selectable or countable visual object on the graph-paper scene."""

    label: str
    kind: str
    points_px: tuple[Point, ...]
    bbox_px: BBox
    metric_value: float
    class_name: str
    graph_points: tuple[Point, ...] = field(default_factory=tuple)
    extra: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptPlan:
    """Task-owned prompt keys and user-visible slot text."""

    bundle_id: str
    scene_key: str
    task_key: str
    prompt_key: str
    answer_hint: str
    annotation_hint: str
    json_example: str
    json_example_answer_only: str
    shape_text: str = ""
    metric_text: str = ""
    target_text: str = ""


@dataclass(frozen=True)
class GraphPaperPrepared:
    """Prompt/image/trace components before final public TaskOutput binding."""

    prompt: str
    prompt_variants: Mapping[str, str]
    prompt_variant: Mapping[str, Any]
    prompt_variant_active_key: str
    prompt_variants_for_trace: Mapping[str, Any]
    image: Image.Image
    annotation_value: Any
    projected_annotation: Mapping[str, Any]
    witness_symbolic: Mapping[str, Any]
    render_spec: Mapping[str, Any]
    render_map: Mapping[str, Any]
    scene_ir: Mapping[str, Any]
    execution_trace: Mapping[str, Any]
