"""Shared mini coordinate-panel rendering helpers for geometry label tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Tuple

from PIL import ImageDraw

from ...shared.text_rendering import draw_text_centered, load_font
from ...shared.text_legibility import draw_text_traced

Point = Tuple[float, float]
BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class CoordinatePanelConfig:
    """Coordinate window and panel-grid shape for small-multiple graph tasks."""

    grid_min: int = -5
    grid_max: int = 5
    columns: int = 3
    rows: int = 2


@dataclass(frozen=True)
class CoordinatePanelStyle:
    """Paint-only style for mini coordinate panels."""

    panel_fill: Tuple[int, int, int] = (255, 255, 255)
    panel_outline: Tuple[int, int, int] = (196, 205, 218)
    plot_fill: Tuple[int, int, int] = (252, 253, 255)
    plot_outline: Tuple[int, int, int] = (184, 194, 208)
    grid_color: Tuple[int, int, int] = (222, 228, 236)
    axis_color: Tuple[int, int, int] = (108, 118, 134)
    tick_color: Tuple[int, int, int] = (95, 105, 122)
    text_color: Tuple[int, int, int] = (38, 46, 59)
    text_stroke_color: Tuple[int, int, int] = (255, 255, 255)

    def to_trace_dict(self) -> dict[str, list[int]]:
        """Return JSON-serializable style metadata."""

        return {
            "panel_fill": [int(value) for value in self.panel_fill],
            "panel_outline": [int(value) for value in self.panel_outline],
            "plot_fill": [int(value) for value in self.plot_fill],
            "plot_outline": [int(value) for value in self.plot_outline],
            "grid_color": [int(value) for value in self.grid_color],
            "axis_color": [int(value) for value in self.axis_color],
            "tick_color": [int(value) for value in self.tick_color],
            "text_color": [int(value) for value in self.text_color],
            "text_stroke_color": [int(value) for value in self.text_stroke_color],
        }


DEFAULT_COORDINATE_PANEL_STYLE = CoordinatePanelStyle()


def linspace(start: float, end: float, count: int) -> Iterable[float]:
    """Yield `count` evenly spaced values from start to end, inclusive."""

    if int(count) <= 1:
        yield float(start)
        return
    for index in range(int(count)):
        yield float(start) + ((float(end) - float(start)) * float(index) / float(count - 1))


def coordinate_panel_layout(
    canvas_width: int,
    canvas_height: int,
    *,
    config: CoordinatePanelConfig = CoordinatePanelConfig(),
) -> dict[str, int]:
    """Return deterministic panel geometry for a coordinate-panel grid."""

    margin_x = max(28, int(round(float(canvas_width) * 0.035)))
    margin_y = max(24, int(round(float(canvas_height) * 0.045)))
    gap_x = max(18, int(round(float(canvas_width) * 0.024)))
    gap_y = max(24, int(round(float(canvas_height) * 0.045)))
    panel_width = int((int(canvas_width) - (2 * margin_x) - ((int(config.columns) - 1) * gap_x)) // int(config.columns))
    panel_height = int((int(canvas_height) - (2 * margin_y) - ((int(config.rows) - 1) * gap_y)) // int(config.rows))
    return {
        "margin_x": int(margin_x),
        "margin_y": int(margin_y),
        "gap_x": int(gap_x),
        "gap_y": int(gap_y),
        "panel_width": int(panel_width),
        "panel_height": int(panel_height),
    }


def panel_bbox_for_index(
    layout: Mapping[str, int],
    index: int,
    *,
    config: CoordinatePanelConfig = CoordinatePanelConfig(),
) -> BBox:
    """Return one panel bbox from grid layout and flat panel index."""

    col = int(index) % int(config.columns)
    row = int(index) // int(config.columns)
    left = int(layout["margin_x"]) + (col * (int(layout["panel_width"]) + int(layout["gap_x"])))
    top = int(layout["margin_y"]) + (row * (int(layout["panel_height"]) + int(layout["gap_y"])))
    return (left, top, left + int(layout["panel_width"]), top + int(layout["panel_height"]))


def plot_bbox_for_panel(panel_bbox: BBox) -> BBox:
    """Return the inner square plot bbox for one coordinate panel."""

    left, top, right, bottom = panel_bbox
    panel_w = int(right) - int(left)
    panel_h = int(bottom) - int(top)
    plot_size = min(panel_w - 54, panel_h - 54)
    plot_left = int(left) + 36
    plot_top = int(top) + 32
    return (plot_left, plot_top, plot_left + int(plot_size), plot_top + int(plot_size))


def graph_point_to_panel_pixel(
    point: Point,
    *,
    plot_bbox: BBox,
    config: CoordinatePanelConfig = CoordinatePanelConfig(),
) -> Point:
    """Project one graph coordinate into the panel's pixel coordinate space."""

    left, top, right, bottom = [float(value) for value in plot_bbox]
    width = float(right) - float(left)
    height = float(bottom) - float(top)
    span = float(int(config.grid_max) - int(config.grid_min))
    return (
        left + ((float(point[0]) - float(int(config.grid_min))) / span * width),
        top + ((float(int(config.grid_max)) - float(point[1])) / span * height),
    )


def draw_coordinate_panel_grid(
    draw: ImageDraw.ImageDraw,
    *,
    panel_bbox: BBox,
    plot_bbox: BBox,
    label: str,
    config: CoordinatePanelConfig = CoordinatePanelConfig(),
    style: CoordinatePanelStyle | None = None,
) -> None:
    """Draw one labeled coordinate panel with axes, grid lines, and tick labels."""

    resolved_style = style if style is not None else DEFAULT_COORDINATE_PANEL_STYLE

    draw.rounded_rectangle(
        panel_bbox,
        radius=7,
        fill=tuple(int(value) for value in resolved_style.panel_fill),
        outline=tuple(int(value) for value in resolved_style.panel_outline),
        width=2,
    )
    draw.rectangle(
        plot_bbox,
        fill=tuple(int(value) for value in resolved_style.plot_fill),
        outline=tuple(int(value) for value in resolved_style.plot_outline),
        width=1,
    )

    tick_font = load_font(10, bold=False)
    label_font = load_font(24, bold=True)
    label_center = (float(panel_bbox[0] + 17), float(panel_bbox[1] + 17))
    draw_text_centered(
        draw,
        text=str(label),
        center=label_center,
        font=label_font,
        fill=tuple(int(value) for value in resolved_style.text_color),
        stroke_fill=tuple(int(value) for value in resolved_style.text_stroke_color),
        stroke_width=1,
    )

    grid_min = int(config.grid_min)
    grid_max = int(config.grid_max)
    for value in range(grid_min, grid_max + 1):
        x0, _ = graph_point_to_panel_pixel((value, grid_min), plot_bbox=plot_bbox, config=config)
        _, y0 = graph_point_to_panel_pixel((grid_min, value), plot_bbox=plot_bbox, config=config)
        line_width = 2 if value == 0 else 1
        line_fill = resolved_style.axis_color if value == 0 else resolved_style.grid_color
        draw.line([(x0, plot_bbox[1]), (x0, plot_bbox[3])], fill=line_fill, width=line_width)
        draw.line([(plot_bbox[0], y0), (plot_bbox[2], y0)], fill=line_fill, width=line_width)
        if value in {-4, -2, 0, 2, 4}:
            draw_text_traced(draw,(x0 - 5, plot_bbox[3] + 3), str(value), font=tick_font, fill=resolved_style.tick_color, role="readout", required=False)
            draw_text_traced(draw,(plot_bbox[0] - 21, y0 - 6), str(value), font=tick_font, fill=resolved_style.tick_color, role="readout", required=False)

    x_axis_end = graph_point_to_panel_pixel((grid_max, 0), plot_bbox=plot_bbox, config=config)
    y_axis_end = graph_point_to_panel_pixel((0, grid_max), plot_bbox=plot_bbox, config=config)
    draw.polygon(
        [
            (x_axis_end[0], x_axis_end[1]),
            (x_axis_end[0] - 7, x_axis_end[1] - 4),
            (x_axis_end[0] - 7, x_axis_end[1] + 4),
        ],
        fill=tuple(int(value) for value in resolved_style.axis_color),
    )
    draw.polygon(
        [
            (y_axis_end[0], y_axis_end[1]),
            (y_axis_end[0] - 4, y_axis_end[1] + 7),
            (y_axis_end[0] + 4, y_axis_end[1] + 7),
        ],
        fill=tuple(int(value) for value in resolved_style.axis_color),
    )


def draw_endpoint(
    draw: ImageDraw.ImageDraw,
    point_px: Point,
    *,
    color: Tuple[int, int, int],
    radius: int,
    outline: Tuple[int, int, int] = (255, 255, 255),
) -> None:
    """Draw one small point marker centered on a projected coordinate."""

    x, y = float(point_px[0]), float(point_px[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=color, outline=outline, width=1)
