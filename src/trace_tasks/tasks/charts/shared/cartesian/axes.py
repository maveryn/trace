"""Axis and frame drawing primitives for cartesian chart renderers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

from PIL import ImageDraw

from .geometry import project_index, project_linear, project_linear_inverted


RGB = Sequence[int]


def draw_plot_frame(
    draw: ImageDraw.ImageDraw,
    plot_bbox: Sequence[float],
    *,
    fill: RGB,
    outline: RGB,
    width: int = 1,
) -> None:
    """Draw one rectangular plot surface."""

    draw.rectangle(
        [float(value) for value in plot_bbox[:4]],
        fill=tuple(int(value) for value in fill),
        outline=tuple(int(value) for value in outline),
        width=max(1, int(width)),
    )


def draw_axis_lines(
    draw: ImageDraw.ImageDraw,
    plot_bbox: Sequence[float],
    *,
    axis_rgb: RGB,
    axis_width_px: int,
    left: bool = True,
    bottom: bool = True,
    right: bool = False,
    top: bool = False,
) -> None:
    """Draw selected border-axis lines around a plot bbox."""

    x0, y0, x1, y1 = [float(value) for value in plot_bbox[:4]]
    color = tuple(int(value) for value in axis_rgb)
    width = max(1, int(axis_width_px))
    if bool(left):
        draw.line([x0, y0, x0, y1], fill=color, width=width)
    if bool(bottom):
        draw.line([x0, y1, x1, y1], fill=color, width=width)
    if bool(right):
        draw.line([x1, y0, x1, y1], fill=color, width=width)
    if bool(top):
        draw.line([x0, y0, x1, y0], fill=color, width=width)


def draw_horizontal_value_grid_ticks(
    draw: ImageDraw.ImageDraw,
    plot_bbox: Sequence[float],
    *,
    tick_values: Iterable[int | float],
    domain_min: int | float,
    domain_max: int | float,
    grid_rgb: RGB,
    axis_rgb: RGB,
    grid_width_px: int,
    tick_width_px: int,
    tick_length_px: float = 0.0,
    tick_side: str = "left",
    grid_values: Iterable[int | float] | None = None,
) -> dict[float, float]:
    """Draw horizontal gridlines and optional side tick marks for y values."""

    x0, y0, x1, y1 = [float(value) for value in plot_bbox[:4]]
    grid_color = tuple(int(value) for value in grid_rgb)
    axis_color = tuple(int(value) for value in axis_rgb)
    grid_value_set = None if grid_values is None else {float(value) for value in grid_values}
    positions: dict[float, float] = {}
    for raw_value in tick_values:
        value = float(raw_value)
        y = project_linear_inverted(
            value,
            domain_min=float(domain_min),
            domain_max=float(domain_max),
            pixel_top=y0,
            pixel_bottom=y1,
        )
        if grid_value_set is None or value in grid_value_set:
            draw.line([x0, y, x1, y], fill=grid_color, width=max(1, int(grid_width_px)))
        if float(tick_length_px) > 0.0:
            if str(tick_side) == "right":
                draw.line([x1, y, x1 + float(tick_length_px), y], fill=axis_color, width=max(1, int(tick_width_px)))
            else:
                draw.line([x0 - float(tick_length_px), y, x0, y], fill=axis_color, width=max(1, int(tick_width_px)))
        positions[value] = float(y)
    return positions


def draw_vertical_index_grid_ticks(
    draw: ImageDraw.ImageDraw,
    plot_bbox: Sequence[float],
    *,
    count: int,
    grid_rgb: RGB,
    axis_rgb: RGB,
    grid_width_px: int,
    tick_width_px: int,
    tick_length_px: float = 0.0,
) -> dict[int, float]:
    """Draw vertical gridlines and bottom ticks for evenly spaced indices."""

    x0, y0, x1, y1 = [float(value) for value in plot_bbox[:4]]
    grid_color = tuple(int(value) for value in grid_rgb)
    axis_color = tuple(int(value) for value in axis_rgb)
    positions: dict[int, float] = {}
    for index in range(max(0, int(count))):
        x = project_index(int(index), pixel_min=x0, pixel_max=x1, count=int(count))
        draw.line([x, y0, x, y1], fill=grid_color, width=max(1, int(grid_width_px)))
        if float(tick_length_px) > 0.0:
            draw.line([x, y1, x, y1 + float(tick_length_px)], fill=axis_color, width=max(1, int(tick_width_px)))
        positions[int(index)] = float(x)
    return positions


def draw_vertical_value_grid_ticks(
    draw: ImageDraw.ImageDraw,
    plot_bbox: Sequence[float],
    *,
    tick_values: Iterable[int | float],
    domain_min: int | float,
    domain_max: int | float,
    grid_rgb: RGB,
    axis_rgb: RGB,
    grid_width_px: int,
    tick_width_px: int,
    tick_length_px: float = 0.0,
    grid_values: Iterable[int | float] | None = None,
) -> dict[float, float]:
    """Draw vertical gridlines and bottom ticks for x-axis data values."""

    x0, y0, x1, y1 = [float(value) for value in plot_bbox[:4]]
    grid_color = tuple(int(value) for value in grid_rgb)
    axis_color = tuple(int(value) for value in axis_rgb)
    grid_value_set = None if grid_values is None else {float(value) for value in grid_values}
    positions: dict[float, float] = {}
    for raw_value in tick_values:
        value = float(raw_value)
        x = project_linear(
            value,
            domain_min=float(domain_min),
            domain_max=float(domain_max),
            pixel_min=x0,
            pixel_max=x1,
        )
        if grid_value_set is None or value in grid_value_set:
            draw.line([x, y0, x, y1], fill=grid_color, width=max(1, int(grid_width_px)))
        if float(tick_length_px) > 0.0:
            draw.line([x, y1, x, y1 + float(tick_length_px)], fill=axis_color, width=max(1, int(tick_width_px)))
        positions[value] = float(x)
    return positions


def draw_horizontal_value_ticks_from_positions(
    draw: ImageDraw.ImageDraw,
    plot_bbox: Sequence[float],
    *,
    positions: Mapping[int | float, float],
    axis_rgb: RGB,
    tick_width_px: int,
    tick_length_px: float,
    tick_side: str = "left",
) -> None:
    """Draw side ticks for precomputed y positions."""

    x0, _y0, x1, _y1 = [float(value) for value in plot_bbox[:4]]
    axis_color = tuple(int(value) for value in axis_rgb)
    for y in positions.values():
        if str(tick_side) == "right":
            draw.line([x1, float(y), x1 + float(tick_length_px), float(y)], fill=axis_color, width=max(1, int(tick_width_px)))
        else:
            draw.line([x0 - float(tick_length_px), float(y), x0, float(y)], fill=axis_color, width=max(1, int(tick_width_px)))


__all__ = [
    "draw_axis_lines",
    "draw_horizontal_value_grid_ticks",
    "draw_horizontal_value_ticks_from_positions",
    "draw_plot_frame",
    "draw_vertical_index_grid_ticks",
    "draw_vertical_value_grid_ticks",
]
