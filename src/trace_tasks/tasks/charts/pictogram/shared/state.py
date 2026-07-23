"""State objects for pictogram chart scene packages."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


DOMAIN = "charts"
SCENE_ID = "pictogram"
SCENE_NAMESPACE = "charts.pictogram"
PROMPT_BUNDLE_ID = "charts_pictogram_v1"

SUPPORTED_SCENE_VARIANTS: tuple[str, ...] = (
    "waffle_grid_blocks",
    "pictogram_rows",
)
SUPPORTED_GLYPHS: tuple[str, ...] = ("circle", "square", "star", "person", "car", "leaf", "coin", "book")

RGB = tuple[int, int, int]
BBox = list[float]


@dataclass(frozen=True)
class PictogramCategory:
    category_id: str
    label: str
    mark_count: int
    total: int
    color_rgb: RGB


@dataclass(frozen=True)
class PictogramQuery:
    branch_id: str
    answer: Any
    answer_type: str
    annotation_type: str
    annotation_category_ids: tuple[str, ...]
    params: dict[str, Any]


@dataclass(frozen=True)
class PictogramDataset:
    categories: tuple[PictogramCategory, ...]
    unit_scale: int
    unit_scale_probabilities: dict[str, float]
    branch_id: str
    branch_probabilities: dict[str, float]
    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    glyph_name: str
    glyph_probabilities: dict[str, float]
    query: PictogramQuery
    title: str


@dataclass(frozen=True)
class PictogramBaseSample:
    scene_variant: str
    scene_variant_probabilities: dict[str, float]
    glyph_name: str
    glyph_probabilities: dict[str, float]
    unit_scale: int
    unit_scale_probabilities: dict[str, float]
    category_count_range: tuple[int, int]
    mark_count_range: tuple[int, int]
    mark_counts: tuple[int, ...]
    title: str


@dataclass(frozen=True)
class PictogramRenderParams:
    canvas_width: int
    canvas_height: int
    outer_margin_px: int
    title_band_height_px: int
    legend_height_px: int
    row_gap_px: int
    label_width_px: int
    mark_gap_px: int
    mark_columns_max: int
    mark_size_max_px: int
    row_corner_radius_px: int
    panel_outline_width_px: int
    title_font_size_px: int
    label_font_size_px: int
    legend_font_size_px: int
    value_font_size_px: int
    text_rgb: RGB
    muted_text_rgb: RGB
    text_stroke_rgb: RGB
    panel_fill_rgb: RGB
    panel_outline_rgb: RGB
    legend_fill_rgb: RGB
    mark_outline_rgb: RGB
    layout_jitter_meta: dict[str, Any]


@dataclass(frozen=True)
class RenderedPictogramScene:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    plot_bbox_px: BBox
    legend_bbox_px: BBox
    category_bboxes_px: dict[str, BBox]
    mark_bboxes_px: dict[str, list[BBox]]
    render_meta: dict[str, Any]
