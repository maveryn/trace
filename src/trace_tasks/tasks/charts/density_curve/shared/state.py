"""State objects for density-curve chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image

from trace_tasks.tasks.charts.shared.label_assets import ResolvedChartLabels


RGB = Tuple[int, int, int]
BBox = List[float]


@dataclass(frozen=True)
class CurvePoint:
    """One sampled point on a smooth density curve."""

    x_value: float
    y_value: float


@dataclass(frozen=True)
class DensityCurve:
    """One labeled density curve and its computed metrics."""

    label: str
    color_rgb: RGB
    line_style: str
    family: str
    component_count: int
    points: Tuple[CurvePoint, ...]
    mean_x: float
    mode_x: float
    mode_y: float
    interval_mass: float
    density_at_x: float


@dataclass(frozen=True)
class DensityCurveSceneSample:
    """A sampled density-curve scene before objective binding."""

    curves: Tuple[DensityCurve, ...]
    x_min: float
    x_max: float
    y_max: float
    curve_count: int
    interval_start: float
    interval_end: float
    reference_x: float
    label_resolution: ResolvedChartLabels
    curve_count_range: Tuple[int, int]
    curve_count_probabilities: Dict[str, float]
    density_family_probabilities: Dict[str, float]


@dataclass(frozen=True)
class DensityCurveObjectiveSpec:
    """Task-owned semantic objective parameters for density-curve tasks."""

    metric_name: str
    direction: str
    annotation_key: str
    visible_role: str
    min_gap_param: str
    min_gap_fallback: float


@dataclass(frozen=True)
class DensityCurveQuery:
    """One bound objective over a sampled density-curve scene."""

    prompt_key: str
    answer_label: str
    annotation_key: str
    visible_role: str
    interval_start: float
    interval_end: float
    reference_x: float
    trace: Dict[str, Any]


@dataclass(frozen=True)
class DensityCurveDataset:
    """A complete density-curve dataset ready for rendering."""

    curves: Tuple[DensityCurve, ...]
    query: DensityCurveQuery
    x_min: float
    x_max: float
    y_max: float
    curve_count: int
    label_resolution: ResolvedChartLabels


@dataclass(frozen=True)
class DensityCurveRendered:
    """Rendered image and projection maps for density-curve tasks."""

    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    plot_bbox_px: BBox
    legend_bbox_px: BBox
    legend_item_bboxes_px: Dict[str, BBox]
    curve_bboxes_px: Dict[str, BBox]
    mean_marker_bboxes_px: Dict[str, BBox]
    mode_marker_bboxes_px: Dict[str, BBox]
    interval_mass_bboxes_px: Dict[str, BBox]
    interval_mass_points_px: Dict[str, List[float]]
    density_at_x_points_px: Dict[str, List[float]]
    title_bbox_px: BBox
    render_meta: Dict[str, Any]
