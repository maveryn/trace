"""State containers for the parallel-coordinates chart scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


RGB = tuple[int, int, int]
BBox = list[float]
Point = list[float]


@dataclass(frozen=True)
class ParallelProfile:
    """One rendered profile line and its metric values."""

    profile_id: str
    label: str
    values: tuple[int, ...]
    color_rgb: RGB


@dataclass(frozen=True)
class ParallelQueryState:
    """Bound semantic state for one parallel-coordinate objective."""

    answer: int | str
    answer_type: str
    axis_i: int
    axis_j: int
    threshold: int | None
    annotation_profile_ids: tuple[str, ...]
    crossing_pairs: tuple[tuple[str, str], ...]
    params: dict[str, Any]


@dataclass(frozen=True)
class ParallelDataset:
    """Dataset consumed by renderer and task-owned answer binding."""

    scene_variant: str
    metrics: tuple[str, ...]
    profiles: tuple[ParallelProfile, ...]
    value_min: int
    value_max: int
    query: ParallelQueryState


@dataclass(frozen=True)
class ParallelRenderParams:
    """Resolved style and layout parameters for the renderer."""

    canvas_width: int
    canvas_height: int
    plot_margin_left_px: int
    plot_margin_right_px: int
    plot_margin_top_px: int
    plot_margin_bottom_px: int
    panel_fill_rgb: RGB
    panel_border_rgb: RGB
    plot_fill_rgb: RGB
    axis_rgb: RGB
    selected_axis_rgb: RGB
    grid_rgb: RGB
    threshold_rgb: RGB
    text_rgb: RGB
    muted_text_rgb: RGB
    text_stroke_rgb: RGB
    line_width_px: int
    point_radius_px: int
    axis_line_width_px: int
    selected_axis_line_width_px: int
    grid_line_width_px: int
    label_font_size_px: int
    tick_font_size_px: int
    title_font_size_px: int
    threshold_font_size_px: int
    layout_jitter_meta: dict[str, Any]


@dataclass(frozen=True)
class RenderedParallelScene:
    """Rendered chart plus pixel projections for annotation binding."""

    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    plot_bbox_px: BBox
    axis_x_px: dict[int, float]
    point_bboxes_px: dict[str, BBox]
    segment_bboxes_px: dict[str, BBox]
    profile_bboxes_px: dict[str, BBox]
    label_bboxes_px: dict[str, BBox]
    threshold_bboxes_px: dict[int, BBox]


@dataclass(frozen=True)
class ParallelRenderResult:
    """Rendered image, projections, and visual metadata."""

    image: Image.Image
    rendered_scene: RenderedParallelScene
    render_params: ParallelRenderParams
    background_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    chart_font_family: str


__all__ = [
    "BBox",
    "ParallelDataset",
    "ParallelProfile",
    "ParallelQueryState",
    "ParallelRenderParams",
    "ParallelRenderResult",
    "Point",
    "RGB",
    "RenderedParallelScene",
]
