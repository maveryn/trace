"""State objects for radar chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


DOMAIN = "charts"
SCENE_ID = "radar"
SCENE_NAMESPACE = "charts.radar"
PROMPT_BUNDLE_ID = "charts_radar_v1"

SMALL_MULTIPLE_SCENE_VARIANT = "small_multiple_radar"
SINGLE_PROFILE_SCENE_VARIANT = "single_radar_multi_profile"

BBox = tuple[float, float, float, float]
Point = list[float]
RGB = tuple[int, int, int]


@dataclass(frozen=True)
class RadarProfile:
    profile_label: str
    values: dict[str, int]
    color_rgb: RGB


@dataclass(frozen=True)
class RadarPanel:
    panel_label: str
    profiles: tuple[RadarProfile, ...]


@dataclass(frozen=True)
class RadarQuery:
    branch_id: str
    answer: str | int
    answer_type: str
    annotation_type: str
    metric_label: str
    panel_label: str
    profile_a_label: str
    profile_b_label: str
    threshold_value: int
    minimum_metric_count: int
    annotation_point_ids: tuple[str, ...]
    annotation_panel_labels: tuple[str, ...]
    annotation_point_id_pairs: tuple[tuple[str, str], ...]
    params: dict[str, Any]


@dataclass(frozen=True)
class RadarDataset:
    metrics: tuple[str, ...]
    panels: tuple[RadarPanel, ...]
    scene_variant: str
    branch_id: str
    branch_probabilities: dict[str, float]
    highlight_metric_label: str
    query: RadarQuery


@dataclass(frozen=True)
class RenderedRadarScene:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    point_bboxes: dict[str, list[float]]
    panel_bboxes: dict[str, list[float]]
    panel_title_bboxes: dict[str, list[float]]
    legend_bboxes: dict[str, list[float]]
    plot_bbox_px: list[float]
    render_meta: dict[str, Any]


@dataclass(frozen=True)
class RadarRenderResult:
    rendered_scene: RenderedRadarScene
    chart_font_family: str

    @property
    def image(self) -> Image.Image:
        return self.rendered_scene.image
