"""State containers for contour-density chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


RGB = Tuple[int, int, int]
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Region:
    region_id: str
    label: str
    option_label: str
    center_x: float
    center_y: float
    radius_x: float
    radius_y: float
    density: float
    density_level: int
    color_rgb: RGB


@dataclass(frozen=True)
class Reference:
    kind: str
    x_value: float
    y_value: float


@dataclass(frozen=True)
class DensityThresholdGuide:
    label: str
    level: int
    operator: str


@dataclass(frozen=True)
class QuerySelection:
    prompt_key: str
    answer: Any
    answer_type: str
    annotation_type: str
    annotation_roles: Dict[str, str]
    annotation_region_ids: Tuple[str, ...]
    trace: Dict[str, Any]


@dataclass(frozen=True)
class ContourDataset:
    scene_variant: str
    regions: Tuple[Region, ...]
    query: QuerySelection
    reference: Reference | None
    threshold_guide: DensityThresholdGuide | None = None


@dataclass(frozen=True)
class RenderParams:
    canvas_width: int
    canvas_height: int
    margin_left: int
    margin_right: int
    margin_top: int
    margin_bottom: int
    grid_line_width: int
    axis_line_width: int
    tick_font_size: int
    label_font_size: int
    title_font_size: int
    marker_radius: int
    text_rgb: RGB
    muted_rgb: RGB
    axis_rgb: RGB
    grid_rgb: RGB
    plot_fill_rgb: RGB
    reference_rgb: RGB
    text_stroke_rgb: RGB
    layout_jitter: Dict[str, Any]


@dataclass(frozen=True)
class RenderedContourScene:
    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    plot_bbox_px: List[float]
    region_bboxes: Dict[str, List[float]]
    option_bboxes: Dict[str, List[float]]
    reference_bboxes: Dict[str, List[float]]
    render_meta: Dict[str, Any]


__all__ = [
    "RGB",
    "BBox",
    "Region",
    "Reference",
    "DensityThresholdGuide",
    "QuerySelection",
    "ContourDataset",
    "RenderParams",
    "RenderedContourScene",
]
