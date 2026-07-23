"""State objects for regular-polygon decomposition diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

from trace_tasks.tasks.geometry.shared.scene_transform import LazySceneTransform

Point = tuple[float, float]
BBox = tuple[float, float, float, float]
Color = tuple[int, int, int]


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
    accent_color: Color
    fill_color: Color
    shaded_fill_color: Color
    panel_fill_color: Color
    panel_border_color: Color
    line_width: int
    label_stroke_width: int
    font: Any
    small_font: Any
    diagram_style_meta: dict[str, Any]
    background_meta: dict[str, Any]
    scene_transform: LazySceneTransform


@dataclass(frozen=True)
class RegularPolygonProblem:
    n_sides: int
    wedge_count: int
    start_index: int
    answer: float
    answer_type: str
    target_name: str
    relation: str
    total_area: float | None
    wedge_area: float | None
    side_length: float | None
    apothem: float | None
    perimeter: float | None
    central_angle_degrees: int
    case_index: int
    layout_seed: int
    show_angle_unknown: bool = False
    show_known_side_length: bool = False
    show_unknown_side_length: bool = False
    show_apothem: bool = False
    show_total_area_readout: bool = False
    show_perimeter_readout: bool = False
    show_wedge_area_readout: bool = False
    show_shaded_region: bool = True
    show_side_endpoint_labels: bool = True
    show_region_label: bool = False
    show_midpoint_label: bool = False


@dataclass(frozen=True)
class SceneGeometry:
    center: Point
    vertices: tuple[Point, ...]
    selected_wedge_indices: tuple[int, ...]
    selected_region_midpoint: Point
    angle_span_degrees: float


@dataclass(frozen=True)
class RenderedRegularPolygonScene:
    image: Image.Image
    geometry: SceneGeometry
    annotation_points: dict[str, Point]
    readout_bboxes: dict[str, BBox]
    construction_bboxes: dict[str, BBox]
    render_map: dict[str, Any]


__all__ = [
    "BBox",
    "Color",
    "Point",
    "RegularPolygonProblem",
    "RenderContext",
    "RenderedRegularPolygonScene",
    "SceneGeometry",
]
