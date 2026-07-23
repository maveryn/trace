"""State objects for scientific axis-frame chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image

from trace_tasks.tasks.charts.shared.cartesian.geometry import round_bbox, round_point


DOMAIN = "charts"
SCENE_ID = "scientific_axis_frame"
SCENE_NAMESPACE = "charts.scientific_axis_frame"
PROMPT_BUNDLE_ID = "charts_scientific_axis_frame_v1"

RGB = tuple[int, int, int]
BBox = list[float]


@dataclass(frozen=True)
class AxisSpec:
    axis: str
    values: tuple[int, ...]
    start: int
    step: int
    count: int
    deltas: tuple[int, ...]


@dataclass(frozen=True)
class AxisFrameBinding:
    axis: str
    answer: int
    answer_type: str
    annotation_roles: dict[str, str]
    tick_values: dict[str, int]
    trace: dict[str, Any]


@dataclass(frozen=True)
class AxisFrameDataset:
    x_axis: AxisSpec
    y_axis: AxisSpec
    binding: AxisFrameBinding
    series_points: tuple[tuple[float, float], ...]


@dataclass(frozen=True)
class AxisFrameRenderParams:
    canvas_width: int
    canvas_height: int
    margin_left_px: int
    margin_right_px: int
    margin_top_px: int
    margin_bottom_px: int
    axis_label_font_size_px: int
    tick_font_size_px: int
    axis_line_width_px: int
    grid_line_width_px: int
    line_width_px: int
    point_radius_px: int
    text_rgb: RGB
    muted_text_rgb: RGB
    text_stroke_rgb: RGB
    axis_rgb: RGB
    grid_rgb: RGB
    panel_fill_rgb: RGB
    panel_outline_rgb: RGB
    series_rgb: RGB
    marker_rgb: RGB
    font_family: str
    layout_jitter_meta: dict[str, Any]


@dataclass(frozen=True)
class RenderedAxisFrameScene:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    plot_bbox_px: BBox
    tick_label_bboxes_px: dict[str, BBox]
    tick_points_px: dict[str, list[float]]
    axis_label_bboxes_px: dict[str, BBox]
    render_meta: dict[str, Any]


@dataclass(frozen=True)
class AxisFrameRenderResult:
    image: Image.Image
    rendered_scene: RenderedAxisFrameScene
    chart_font_family: str
    render_params: AxisFrameRenderParams
    background_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]


def bbox(values: tuple[float, float, float, float] | list[float]) -> BBox:
    return round_bbox(values)


def point(x: float, y: float) -> list[float]:
    return round_point(float(x), float(y))


__all__ = [
    "AxisFrameBinding",
    "AxisFrameDataset",
    "AxisFrameRenderParams",
    "AxisFrameRenderResult",
    "AxisSpec",
    "BBox",
    "DOMAIN",
    "PROMPT_BUNDLE_ID",
    "RGB",
    "RenderedAxisFrameScene",
    "SCENE_ID",
    "SCENE_NAMESPACE",
    "bbox",
    "point",
]
