"""State containers for dumbbell chart scene primitives."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from PIL import Image


RGB = Tuple[int, int, int]


@dataclass(frozen=True)
class DumbbellRow:
    """One visible paired-value row in a dumbbell chart."""

    row_id: str
    label: str
    value_a: int
    value_b: int

    @property
    def gap(self) -> int:
        return abs(int(self.value_a) - int(self.value_b))

    @property
    def signed_delta_a_minus_b(self) -> int:
        return int(self.value_a) - int(self.value_b)


@dataclass(frozen=True)
class DumbbellQuery:
    """Task-owned answer and annotation binding for a rendered dumbbell chart."""

    answer: str | int
    answer_type: str
    annotation_row_ids: Tuple[str, ...]
    params: Dict[str, Any]


@dataclass(frozen=True)
class DumbbellDataset:
    """A fully bound dumbbell scene sample before rendering."""

    scene_variant: str
    series_a_name: str
    series_b_name: str
    rows: Tuple[DumbbellRow, ...]
    query: DumbbellQuery


@dataclass(frozen=True)
class DumbbellRenderParams:
    """Resolved rendering parameters for one dumbbell chart image."""

    canvas_width: int
    canvas_height: int
    plot_margin_left_px: int
    plot_margin_right_px: int
    plot_margin_top_px: int
    plot_margin_bottom_px: int
    axis_line_width_px: int
    grid_line_width_px: int
    row_line_width_px: int
    connector_width_px: int
    point_radius_px: int
    point_outline_width_px: int
    tick_length_px: int
    title_font_size_px: int
    subtitle_font_size_px: int
    label_font_size_px: int
    tick_font_size_px: int
    legend_font_size_px: int
    panel_fill_rgb: RGB
    panel_border_rgb: RGB
    plot_fill_rgb: RGB
    axis_color_rgb: RGB
    grid_color_rgb: RGB
    row_line_rgb: RGB
    connector_rgb: RGB
    text_color_rgb: RGB
    muted_text_rgb: RGB
    text_stroke_rgb: RGB
    series_a_rgb: RGB
    series_b_rgb: RGB
    font_family: str
    layout_jitter_meta: Dict[str, Any]


@dataclass(frozen=True)
class RenderedDumbbell:
    """Rendered dumbbell image plus projection maps."""

    image: Image.Image
    entities: Tuple[Dict[str, Any], ...]
    plot_bbox_px: List[float]
    row_label_bboxes_px: Dict[str, List[float]]
    point_bboxes_px: Dict[str, List[float]]
    row_pair_bboxes_px: Dict[str, List[float]]
    connector_bboxes_px: Dict[str, List[float]]
    legend_bboxes_px: Dict[str, List[float]]


__all__ = [
    "DumbbellDataset",
    "DumbbellQuery",
    "DumbbellRenderParams",
    "DumbbellRow",
    "RGB",
    "RenderedDumbbell",
]
