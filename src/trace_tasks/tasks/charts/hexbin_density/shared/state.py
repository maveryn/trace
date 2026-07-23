"""State containers for hexbin-density chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image

from trace_tasks.tasks.charts.hexbin_density.shared.defaults import RGB


@dataclass(frozen=True)
class HexBin:
    bin_id: str
    row_index: int
    column_index: int
    density_level: int
    fill_rgb: RGB


@dataclass(frozen=True)
class ThresholdQuery:
    threshold_direction: str
    threshold_phrase: str
    threshold_operator: str
    threshold_level: int
    answer: int
    annotation_bin_ids: Tuple[str, ...]
    trace: Dict[str, Any]


@dataclass(frozen=True)
class HexbinDataset:
    row_count: int
    column_count: int
    bins: Tuple[HexBin, ...]
    query: ThresholdQuery
    density_palette_scheme: str
    density_palette_rgb: Tuple[RGB, ...]
    density_palette_trace: Dict[str, Any]


@dataclass(frozen=True)
class RenderParams:
    canvas_width: int
    canvas_height: int
    margin_left: int
    margin_right: int
    margin_top: int
    margin_bottom: int
    legend_width: int
    axis_line_width: int
    grid_line_width: int
    hex_outline_width: int
    tick_font_size: int
    label_font_size: int
    title_font_size: int
    plot_fill_rgb: RGB
    axis_rgb: RGB
    grid_rgb: RGB
    text_rgb: RGB
    muted_rgb: RGB
    hex_outline_rgb: RGB
    threshold_guide_fill_rgb: RGB
    layout_jitter: Dict[str, Any]


@dataclass(frozen=True)
class RenderedHexbinScene:
    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    plot_bbox_px: List[float]
    legend_bbox_px: List[float]
    title_bbox_px: List[float]
    threshold_guide_bbox_px: List[float]
    bin_bboxes_px: Dict[str, List[float]]
    bin_centers_px: Dict[str, List[float]]
    render_meta: Dict[str, Any]


__all__ = [
    "HexBin",
    "HexbinDataset",
    "RenderedHexbinScene",
    "RenderParams",
    "ThresholdQuery",
]
