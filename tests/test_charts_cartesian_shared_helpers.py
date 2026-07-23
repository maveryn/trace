"""Tests for shared cartesian-chart helper primitives."""

from __future__ import annotations

from PIL import Image, ImageDraw

from trace_tasks.tasks.charts.shared.cartesian.axes import (
    draw_axis_lines,
    draw_horizontal_value_grid_ticks,
    draw_plot_frame,
    draw_vertical_index_grid_ticks,
    draw_vertical_value_grid_ticks,
)
from trace_tasks.tasks.charts.shared.cartesian.frame import plot_bbox_from_margins
from trace_tasks.tasks.charts.shared.cartesian.geometry import (
    clip_bbox_to_container,
    project_index,
    project_linear,
    project_linear_inverted,
    project_xy,
    round_bbox,
    round_point,
    union_bboxes,
)
from trace_tasks.tasks.charts.shared.cartesian.lines import draw_dashed_line, line_segments_for_style
from trace_tasks.tasks.charts.shared.cartesian.markers import draw_marker


def test_cartesian_geometry_helpers_match_existing_chart_rounding() -> None:
    assert round_bbox([1.23456, 2, 3.9999, 4.0001]) == [1.235, 2.0, 4.0, 4.0]
    assert round_point(12.3456, 8) == [12.346, 8.0]
    assert union_bboxes([[1, 2, 4, 8], [0, 3, 9, 6]]) == [0.0, 2.0, 9.0, 8.0]
    assert union_bboxes([[], [1, 2]]) == []
    assert clip_bbox_to_container([-2, 3, 12, 20], [0, 0, 10, 10]) == [0.0, 3.0, 10.0, 10.0]


def test_cartesian_projection_helpers_cover_normal_and_single_slot_cases() -> None:
    assert project_linear(50, domain_min=0, domain_max=100, pixel_min=20, pixel_max=220) == 120.0
    assert project_linear_inverted(25, domain_min=0, domain_max=100, pixel_top=10, pixel_bottom=210) == 160.0
    assert project_index(2, pixel_min=10, pixel_max=70, count=4) == 50.0
    assert project_index(0, pixel_min=10, pixel_max=70, count=1) == 40.0
    assert project_xy(x_value=50, y_value=25, plot_bbox=[20, 10, 220, 210]) == (120.0, 160.0)
    assert project_xy(x_value=200, y_value=-5, plot_bbox=[0, 0, 100, 100], clamp=True) == (100.0, 100.0)
    assert plot_bbox_from_margins(
        canvas_width=640,
        canvas_height=420,
        margin_left_px=64,
        margin_right_px=48,
        margin_top_px=40,
        margin_bottom_px=72,
    ) == [64.0, 40.0, 592.0, 348.0]


def test_cartesian_axis_helpers_return_projected_tick_positions() -> None:
    image = Image.new("RGB", (140, 120), "white")
    draw = ImageDraw.Draw(image)
    plot_box = [20.0, 10.0, 120.0, 90.0]

    draw_plot_frame(draw, plot_box, fill=(250, 250, 250), outline=(20, 20, 20), width=1)
    draw_axis_lines(draw, plot_box, axis_rgb=(10, 20, 30), axis_width_px=2)
    y_positions = draw_horizontal_value_grid_ticks(
        draw,
        plot_box,
        tick_values=(0, 50, 100),
        domain_min=0,
        domain_max=100,
        grid_rgb=(220, 220, 220),
        axis_rgb=(10, 20, 30),
        grid_width_px=1,
        tick_width_px=1,
        tick_length_px=5,
    )
    x_positions = draw_vertical_value_grid_ticks(
        draw,
        plot_box,
        tick_values=(0, 50, 100),
        domain_min=0,
        domain_max=100,
        grid_rgb=(220, 220, 220),
        axis_rgb=(10, 20, 30),
        grid_width_px=1,
        tick_width_px=1,
        tick_length_px=5,
    )
    index_positions = draw_vertical_index_grid_ticks(
        draw,
        plot_box,
        count=3,
        grid_rgb=(220, 220, 220),
        axis_rgb=(10, 20, 30),
        grid_width_px=1,
        tick_width_px=1,
    )

    assert y_positions == {0.0: 90.0, 50.0: 50.0, 100.0: 10.0}
    assert x_positions == {0.0: 20.0, 50.0: 70.0, 100.0: 120.0}
    assert index_positions == {0: 20.0, 1: 70.0, 2: 120.0}


def test_cartesian_line_helpers_draw_without_changing_style_patterns() -> None:
    image = Image.new("RGB", (80, 30), "white")
    draw = ImageDraw.Draw(image)

    draw_dashed_line(draw, (5, 15), (75, 15), fill=(10, 20, 30), width=2, dash_px=8, gap_px=4)

    assert image.getbbox() == (0, 0, 80, 30)
    assert line_segments_for_style("dotted", width=2) == ((0.0, 9.0),)
    assert line_segments_for_style("short_dash", width=2) == ((9.0, 6.0),)


def test_cartesian_marker_helper_returns_stable_bboxes_for_common_shapes() -> None:
    image = Image.new("RGB", (80, 80), "white")
    draw = ImageDraw.Draw(image)

    assert draw_marker(
        draw,
        center=(20, 20),
        radius=6,
        shape="circle",
        fill=(20, 120, 200),
        outline=(255, 255, 255),
        width=2,
    ) == [14.0, 14.0, 26.0, 26.0]
    assert draw_marker(
        draw,
        center=(45, 20),
        radius=5,
        shape="triangle",
        fill=(20, 120, 200),
        outline=(10, 10, 10),
        width=2,
        marker_fill="open",
        polygon_outline_width=2,
    ) == [40.0, 15.0, 50.0, 25.0]
    assert draw_marker(
        draw,
        center=(20, 48),
        radius=6,
        shape="ring",
        fill=(20, 120, 200),
        outline=(20, 120, 200),
        width=4,
        ring_style="outline",
    ) == [14.0, 42.0, 26.0, 54.0]
