"""State records for scatter-cluster chart scenes."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from PIL import Image


DOMAIN = "charts"
SCENE_ID = "scatter_cluster"
SCENE_NAMESPACE = "charts.scatter_cluster"
PROMPT_BUNDLE_ID = "charts_scatter_cluster_v1"

SINGLE_SCATTER = "single_scatter"
AREA_ENVELOPE_SCATTER = "area_envelope_scatter"
SUPPORTED_SCENE_VARIANTS: tuple[str, ...] = (SINGLE_SCATTER, AREA_ENVELOPE_SCATTER)
OPTION_LABELS: tuple[str, ...] = ("A", "B", "C", "D", "E", "F")

RGB = tuple[int, int, int]
BBox = list[float]


@dataclass(frozen=True)
class ScatterPoint:
    point_id: str
    cluster_label: str
    x_value: float
    y_value: float


@dataclass(frozen=True)
class AreaEnvelope:
    center_x: float
    center_y: float
    radius_x: float
    radius_y: float
    angle_degrees: float
    area_value: float


@dataclass(frozen=True)
class Cluster:
    cluster_label: str
    color_rgb: RGB
    center_x: float
    center_y: float
    slope: float
    spread_x: float
    spread_y: float
    points: tuple[ScatterPoint, ...]
    area_envelope: AreaEnvelope | None = None


@dataclass(frozen=True)
class OptionMarker:
    option_label: str
    x_value: float
    y_value: float


@dataclass(frozen=True)
class ScatterClusterQuestion:
    branch_id: str
    branch_probabilities: dict[str, float]
    answer: str
    answer_type: str
    annotation_type: str
    annotation_cluster_labels: tuple[str, ...]
    annotation_option_labels: tuple[str, ...] = field(default_factory=tuple)
    params: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ScatterClusterDataset:
    scene_variant: str
    clusters: tuple[Cluster, ...]
    question: ScatterClusterQuestion
    option_markers: tuple[OptionMarker, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ScatterClusterInputs:
    cluster_count: int
    labels: tuple[str, ...]
    points_per_cluster: int
    answer_label: str


@dataclass(frozen=True)
class ScatterClusterRenderParams:
    canvas_width: int
    canvas_height: int
    plot_margin_left_px: int
    plot_margin_right_px: int
    plot_margin_top_px: int
    plot_margin_bottom_px: int
    axis_line_width_px: int
    grid_line_width_px: int
    tick_length_px: int
    point_radius_px: int
    tick_font_size_px: int
    legend_font_size_px: int
    title_font_size_px: int
    cluster_hull_padding_px: int
    axis_color_rgb: RGB
    grid_color_rgb: RGB
    text_color_rgb: RGB
    text_stroke_rgb: RGB
    plot_fill_rgb: RGB
    panel_fill_rgb: RGB
    panel_border_rgb: RGB
    layout_jitter_meta: dict[str, Any]


@dataclass(frozen=True)
class RenderedScatterCluster:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    plot_bbox_px: BBox
    point_bboxes: dict[str, BBox]
    cluster_bboxes: dict[str, BBox]
    cluster_envelope_bboxes: dict[str, BBox]
    cluster_label_bboxes: dict[str, BBox]
    legend_bboxes: dict[str, BBox]
    option_bboxes: dict[str, BBox]
    option_centers_px: dict[str, list[float]]


@dataclass(frozen=True)
class ScatterClusterRenderResult:
    image: Image.Image
    rendered_scene: RenderedScatterCluster
    render_params: ScatterClusterRenderParams
    background_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    chart_font_family: str
