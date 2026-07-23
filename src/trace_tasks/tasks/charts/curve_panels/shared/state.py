"""Passive state objects for curve-panel chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image

RGB = Tuple[int, int, int]
BBox = Tuple[float, float, float, float]


@dataclass(frozen=True)
class Curve:
    method_label: str
    values: Tuple[int, ...]
    color_rgb: RGB


@dataclass(frozen=True)
class Panel:
    panel_label: str
    curves: Tuple[Curve, ...]


@dataclass(frozen=True)
class Intersection:
    intersection_id: str
    panel_label: str
    method_a_label: str
    method_b_label: str
    x_value: float
    y_value: float


@dataclass(frozen=True)
class ThresholdCrossing:
    crossing_id: str
    panel_label: str
    method_label: str
    x_value: float
    y_value: float
    direction: str


@dataclass(frozen=True)
class QuerySelection:
    prompt_key: str
    scene_variant: str
    answer: str | int
    answer_type: str
    panel_label: str
    method_label: str
    method_a_label: str
    method_b_label: str
    x_value: int
    start_x_value: int
    end_x_value: int
    threshold_value: int
    threshold_direction: str
    threshold_panel_labels: Tuple[str, ...]
    annotation_panel_labels: Tuple[str, ...]
    annotation_point_ids: Tuple[str, ...]
    annotation_keyed_point_ids: Dict[str, str]
    annotation_intersection_ids: Tuple[str, ...]
    annotation_threshold_crossing_ids: Tuple[str, ...]
    trace: Dict[str, Any]


@dataclass(frozen=True)
class CurvePanelDataset:
    scene_variant: str
    x_values: Tuple[int, ...]
    y_min: int
    y_max: int
    panels: Tuple[Panel, ...]
    query: QuerySelection
    intersections: Tuple[Intersection, ...]
    threshold_crossings: Tuple[ThresholdCrossing, ...]


@dataclass(frozen=True)
class RenderedCurvePanels:
    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    plot_bbox_px: List[float]
    panel_bboxes: Dict[str, List[float]]
    panel_plot_bboxes: Dict[str, List[float]]
    point_bboxes: Dict[str, List[float]]
    intersection_bboxes: Dict[str, List[float]]
    threshold_crossing_bboxes: Dict[str, List[float]]
    legend_bboxes: Dict[str, List[float]]
    render_meta: Dict[str, Any]


__all__ = [
    "BBox",
    "Curve",
    "CurvePanelDataset",
    "Intersection",
    "Panel",
    "QuerySelection",
    "RGB",
    "RenderedCurvePanels",
    "ThresholdCrossing",
]
