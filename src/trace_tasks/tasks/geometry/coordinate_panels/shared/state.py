"""Passive state for coordinate-panel geometry scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image

GraphPoint = Tuple[int, int]
PixelPoint = Tuple[float, float]
GraphSegment = Tuple[GraphPoint, GraphPoint]
PixelSegment = Tuple[PixelPoint, PixelPoint]
Color = Tuple[int, int, int]


@dataclass(frozen=True)
class PanelDefaults:
    panel_canvas_width: int = 1024
    panel_canvas_height: int = 720
    panel_grid_min: int = -6
    panel_grid_max: int = 6
    panel_graph_abs_max: int = 4
    panel_marker_radius_px: int = 5
    panel_marker_radius_px_min: int = 4
    panel_marker_radius_px_max: int = 6


@dataclass(frozen=True)
class PanelSpec:
    label: str
    points: Tuple[GraphPoint, GraphPoint, GraphPoint, GraphPoint]
    points_px: Tuple[PixelPoint, PixelPoint, PixelPoint, PixelPoint]
    classified_kind: str
    panel_bbox: List[int]
    plot_bbox: List[int]


@dataclass(frozen=True)
class SegmentPanelSpec:
    label: str
    segments_graph: Tuple[GraphSegment, GraphSegment]
    segments_px: Tuple[PixelSegment, PixelSegment]
    relation_flags: Dict[str, bool]
    panel_bbox: List[int]
    plot_bbox: List[int]


@dataclass(frozen=True)
class TransformPanelSpec:
    label: str
    source_points_graph: Tuple[GraphPoint, ...]
    candidate_points_graph: Tuple[GraphPoint, ...]
    source_points_px: Tuple[PixelPoint, ...]
    candidate_points_px: Tuple[PixelPoint, ...]
    transform_flags: Dict[str, bool]
    panel_bbox: List[int]
    plot_bbox: List[int]


@dataclass(frozen=True)
class PanelScene:
    panels_by_label: Dict[str, PanelSpec]
    marker_meta: Dict[str, Any]
    panel_style_meta: Dict[str, Any]
    image: Image.Image
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    option_count_probabilities: Dict[str, float]


@dataclass(frozen=True)
class SegmentPanelScene:
    panels_by_label: Dict[str, SegmentPanelSpec]
    marker_meta: Dict[str, Any]
    panel_style_meta: Dict[str, Any]
    image: Image.Image
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    option_count_probabilities: Dict[str, float]


@dataclass(frozen=True)
class TransformPanelScene:
    panels_by_label: Dict[str, TransformPanelSpec]
    marker_meta: Dict[str, Any]
    panel_style_meta: Dict[str, Any]
    image: Image.Image
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]
    option_count_probabilities: Dict[str, float]


__all__ = [
    "Color",
    "GraphPoint",
    "GraphSegment",
    "PanelDefaults",
    "PanelScene",
    "PanelSpec",
    "PixelPoint",
    "PixelSegment",
    "SegmentPanelScene",
    "SegmentPanelSpec",
    "TransformPanelScene",
    "TransformPanelSpec",
]
