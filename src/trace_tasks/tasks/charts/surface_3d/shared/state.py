"""State containers for synthetic 3D chart panel primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from PIL import Image


RGB = tuple[int, int, int]
BBox = list[float]


@dataclass(frozen=True)
class Point3D:
    """One projected point or one point in a connected 3D series."""

    point_id: str
    label: str
    x_value: float
    y_value: float
    z_value: float
    color_rgb: RGB
    shape: str = "circle"


@dataclass(frozen=True)
class SurfaceCell:
    """One categorical surface marker at an x/y category position."""

    cell_id: str
    x_label: str
    y_label: str
    x_index: int
    y_index: int
    value: int


@dataclass(frozen=True)
class Panel3D:
    """One mini 3D panel with an ordered value series."""

    panel_label: str
    values: tuple[int, ...]
    color_rgb: RGB


@dataclass(frozen=True)
class Surface3DDataset:
    """Scene grammar inputs independent of public task identity."""

    scene_variant: str
    points: tuple[Point3D, ...]
    surface_cells: tuple[SurfaceCell, ...]
    panels: tuple[Panel3D, ...]
    x_axis_label: str
    y_axis_label: str
    z_axis_label: str
    x_range: tuple[float, float]
    y_range: tuple[float, float]
    z_range: tuple[float, float]
    x_labels: tuple[str, ...]
    y_labels: tuple[str, ...]
    title: str
    connect_points_by_label: bool = False
    reference_y_value: float | None = None


@dataclass(frozen=True)
class Surface3DRenderParams:
    """Resolved visual parameters for one 3D chart render."""

    canvas_width: int
    canvas_height: int
    plot_margin_left_px: int
    plot_margin_right_px: int
    plot_margin_top_px: int
    plot_margin_bottom_px: int
    panel_gap_px: int
    point_radius_px: int
    line_width_px: int
    axis_line_width_px: int
    grid_line_width_px: int
    tick_font_size_px: int
    label_font_size_px: int
    title_font_size_px: int
    panel_title_font_size_px: int
    plot_fill_rgb: RGB
    panel_fill_rgb: RGB
    panel_border_rgb: RGB
    axis_color_rgb: RGB
    grid_color_rgb: RGB
    text_color_rgb: RGB
    text_stroke_rgb: RGB
    surface_low_rgb: RGB
    surface_high_rgb: RGB
    surface_edge_rgb: RGB
    marker_outline_rgb: RGB
    layout_jitter_meta: dict[str, Any]


@dataclass(frozen=True)
class RenderedSurface3D:
    """Rendered 3D chart plus projected visual witness boxes."""

    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    plot_bbox_px: BBox
    point_bboxes_px: dict[str, BBox]
    surface_cell_bboxes_px: dict[str, BBox]
    panel_bboxes_px: dict[str, BBox]


@dataclass(frozen=True)
class Surface3DRenderArtifacts:
    """Final rendered image and associated render metadata."""

    image: Image.Image
    rendered_scene: RenderedSurface3D
    render_params: Surface3DRenderParams
    background_style: Mapping[str, Any]
    post_image_noise: Mapping[str, Any]
    chart_font_family: str


__all__ = [
    "BBox",
    "Panel3D",
    "Point3D",
    "RGB",
    "RenderedSurface3D",
    "Surface3DDataset",
    "Surface3DRenderArtifacts",
    "Surface3DRenderParams",
    "SurfaceCell",
]
