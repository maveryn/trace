"""Public chart scene specs and render payload types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

from PIL import Image


ChartColor = Tuple[int, int, int]

SUPPORTED_CHART_SCENE_VARIANTS: Tuple[str, ...] = (
    "area",
    "bar",
    "line",
    "scatter",
    "radar",
    "pie",
    "donut",
    "horizontal_bar",
    "dot_plot",
    "lollipop",
)

SUPPORTED_MULTISERIES_CHART_SCENE_VARIANTS: Tuple[str, ...] = (
    "grouped_bar",
    "grouped_horizontal_bar",
    "multi_line",
    "grouped_lollipop",
)

SUPPORTED_COMPOSITION_CHART_SCENE_VARIANTS: Tuple[str, ...] = (
    "stacked_bar",
    "stacked_horizontal_bar",
)

SUPPORTED_DISTRIBUTION_CHART_SCENE_VARIANTS: Tuple[str, ...] = (
    "boxplot",
    "histogram",
    "violin",
)


@dataclass(frozen=True)
class ChartMarkSpec:
    """One symbolic chart mark."""

    label: str
    value: int
    fill_rgb: ChartColor | None = None
    outline_rgb: ChartColor | None = None
    visible: bool = True


@dataclass(frozen=True)
class MultiSeriesChartMarkSpec:
    """One symbolic chart mark in a multiseries scene."""

    category_label: str
    series_label: str
    category_rank: int
    series_rank: int
    value: int
    fill_rgb: ChartColor | None = None
    outline_rgb: ChartColor | None = None


@dataclass(frozen=True)
class HistogramBinSpec:
    """One numeric histogram bin."""

    label: str
    count: int
    interval_start: int
    interval_end: int
    fill_rgb: ChartColor | None = None
    outline_rgb: ChartColor | None = None


@dataclass(frozen=True)
class BoxPlotSpec:
    """One rendered categorical boxplot."""

    label: str
    whisker_min: int
    q1: int
    median: int
    q3: int
    whisker_max: int
    fill_rgb: ChartColor | None = None
    outline_rgb: ChartColor | None = None


@dataclass(frozen=True)
class ViolinPlotSpec:
    """One rendered violin plot summary."""

    label: str
    support_min: int
    support_max: int
    mode_values: Tuple[int, ...]
    fill_rgb: ChartColor | None = None
    outline_rgb: ChartColor | None = None


@dataclass(frozen=True)
class ChartRenderParams:
    """Resolved render parameters for one labeled chart scene."""

    canvas_width: int
    canvas_height: int
    plot_margin_left_px: int
    plot_margin_right_px: int
    plot_margin_top_px: int
    plot_margin_bottom_px: int
    axis_line_width_px: int
    grid_line_width_px: int
    tick_length_px: int
    label_font_size_px: int
    tick_font_size_px: int
    label_stroke_width_px: int
    label_bold: bool
    mark_outline_width_px: int
    line_width_px: int
    point_radius_px: int
    bar_width_fraction: float
    axis_color_rgb: ChartColor
    grid_color_rgb: ChartColor
    mark_fill_rgb: ChartColor
    mark_outline_rgb: ChartColor
    text_color_rgb: ChartColor
    text_stroke_rgb: ChartColor
    plot_fill_rgb: ChartColor
    value_axis_window_enabled: bool = False
    value_axis_span_min: int = 10
    value_axis_span_max: int = 25
    value_axis_hard_max: int = 99
    value_axis_major_tick_step: int = 5
    value_axis_minor_tick_step: int = 1
    value_axis_allow_nonzero_min: bool = True
    guide_line_mode: str = "off"
    guide_line_prob: float = 0.0
    guide_line_style: str = "dashed"
    guide_line_width_px: int = 1
    guide_line_color_rgb: ChartColor = (150, 156, 166)
    layout_jitter_px: Tuple[int, int] = (0, 0)
    layout_jitter_meta: Dict[str, Any] | None = None
    violin_mode_line_style: str = "full"
    violin_fill_style: str = "solid"
    violin_width_scale: float = 1.0
    violin_smoothing_scale: float = 1.0
    violin_palette_mode: str = "single"
    violin_palette_offset: int = 0


@dataclass(frozen=True)
class RenderedChartScene:
    """Rendered chart payload plus trace-ready metadata."""

    image: Image.Image
    mark_traces: Tuple[Dict[str, Any], ...]
    entities: Tuple[Dict[str, Any], ...]
    plot_bbox_px: Tuple[int, int, int, int]
    y_axis_max: int
    y_ticks: Tuple[int, ...]
    scene_variant: str
    value_axis_min: int = 0
    value_axis_max: int = 0
    value_axis_span: int = 0
    value_axis_major_ticks: Tuple[int, ...] = ()
    value_axis_minor_ticks: Tuple[int, ...] = ()
    value_axis_window_enabled: bool = False
    guide_line_style: str = "none"
    guide_lines: Tuple[Dict[str, Any], ...] = field(default_factory=tuple)
    legend_bbox_px: Tuple[float, ...] = ()
    legend_item_bboxes_px: Dict[str, Tuple[float, ...]] = field(default_factory=dict)


__all__ = [
    "BoxPlotSpec",
    "ChartColor",
    "ChartMarkSpec",
    "ChartRenderParams",
    "HistogramBinSpec",
    "MultiSeriesChartMarkSpec",
    "RenderedChartScene",
    "SUPPORTED_CHART_SCENE_VARIANTS",
    "SUPPORTED_COMPOSITION_CHART_SCENE_VARIANTS",
    "SUPPORTED_DISTRIBUTION_CHART_SCENE_VARIANTS",
    "SUPPORTED_MULTISERIES_CHART_SCENE_VARIANTS",
    "ViolinPlotSpec",
]
