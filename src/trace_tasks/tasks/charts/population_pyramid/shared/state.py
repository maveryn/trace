"""State objects for population-pyramid chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PIL import Image


DOMAIN = "charts"
SCENE_ID = "population_pyramid"
SCENE_NAMESPACE = "charts.population_pyramid"
PROMPT_BUNDLE_ID = "charts_population_pyramid_v1"

RGB = tuple[int, int, int]
BBox = list[float]


@dataclass(frozen=True)
class PopulationPyramidRow:
    row_id: str
    label: str
    left_value: int
    right_value: int

    @property
    def gap(self) -> int:
        return abs(int(self.left_value) - int(self.right_value))

    @property
    def total(self) -> int:
        return int(self.left_value) + int(self.right_value)


@dataclass(frozen=True)
class PopulationPyramidQuery:
    branch_id: str
    answer: int | str
    answer_type: str
    annotation_type: str
    annotation_row_ids: tuple[str, ...]
    params: dict[str, Any]


@dataclass(frozen=True)
class PopulationPyramidBase:
    age_labels: tuple[str, ...]
    left_series_label: str
    right_series_label: str
    left_color_rgb: RGB
    right_color_rgb: RGB
    title: str
    params: dict[str, Any]


@dataclass(frozen=True)
class PopulationPyramidDataset:
    left_series_label: str
    right_series_label: str
    left_color_rgb: RGB
    right_color_rgb: RGB
    rows: tuple[PopulationPyramidRow, ...]
    branch_id: str
    branch_probabilities: dict[str, float]
    query: PopulationPyramidQuery
    title: str


@dataclass(frozen=True)
class PopulationPyramidRenderParams:
    canvas_width: int
    canvas_height: int
    plot_margin_left_px: int
    plot_margin_right_px: int
    plot_margin_top_px: int
    plot_margin_bottom_px: int
    title_band_height_px: int
    legend_gap_px: int
    axis_line_width_px: int
    grid_line_width_px: int
    bar_outline_width_px: int
    bar_gap_px: int
    title_font_size_px: int
    label_font_size_px: int
    tick_font_size_px: int
    legend_font_size_px: int
    value_font_size_px: int
    axis_max: int
    tick_step: int
    panel_fill_rgb: RGB
    panel_border_rgb: RGB
    axis_rgb: RGB
    grid_rgb: RGB
    text_rgb: RGB
    muted_text_rgb: RGB
    text_stroke_rgb: RGB
    font_family: str
    layout_jitter_meta: dict[str, Any]


@dataclass(frozen=True)
class RenderedPopulationPyramidScene:
    image: Image.Image
    entities: tuple[dict[str, Any], ...]
    plot_bbox_px: BBox
    row_bar_bboxes_px: dict[str, BBox]
    left_bar_bboxes_px: dict[str, BBox]
    right_bar_bboxes_px: dict[str, BBox]
    row_label_bboxes_px: dict[str, BBox]
    render_meta: dict[str, Any]
