"""State containers for the matrix chart scene."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


@dataclass(frozen=True)
class MatrixRenderParams:
    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    panel_padding_px: int
    title_band_height_px: int
    legend_height_px: int
    row_label_width_px: int
    col_label_height_px: int
    cell_gap_px: int
    cell_border_width_px: int
    title_font_size_px: int
    header_font_size_px: int
    cell_font_size_px: int
    legend_font_size_px: int
    panel_fill_rgb: Tuple[int, int, int]
    panel_border_rgb: Tuple[int, int, int]
    title_rgb: Tuple[int, int, int]
    header_text_rgb: Tuple[int, int, int]
    grid_rgb: Tuple[int, int, int]
    inactive_cell_rgb: Tuple[int, int, int]
    highlight_rgb: Tuple[int, int, int]
    legend_text_rgb: Tuple[int, int, int]
    layout_offset_x_px: int
    layout_offset_y_px: int
    layout_jitter_meta: Dict[str, Any]
    font_family: str


@dataclass(frozen=True)
class RenderedMatrix:
    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    panel_bbox_px: List[float]
    title_bbox_px: List[float]
    matrix_bbox_px: List[float]
    legend_bbox_px: List[float]
    cell_bbox_map: Dict[str, List[float]]
    row_label_bbox_map: Dict[str, List[float]]
    column_label_bbox_map: Dict[str, List[float]]


@dataclass(frozen=True)
class MatrixRenderResult:
    image: Image.Image
    rendered_scene: RenderedMatrix
    render_params: MatrixRenderParams
    background_meta: Dict[str, Any]
    post_noise_meta: Dict[str, Any]


@dataclass(frozen=True)
class MatrixVisualSelection:
    scene_variant: str
    scene_variant_probabilities: Dict[str, float]
    palette_variant: str
    palette_variant_probabilities: Dict[str, float]
    header_layout: str
    header_layout_probabilities: Dict[str, float]
    grid_style: str
    grid_style_probabilities: Dict[str, float]


__all__ = ["MatrixRenderParams", "MatrixRenderResult", "MatrixVisualSelection", "RenderedMatrix"]
