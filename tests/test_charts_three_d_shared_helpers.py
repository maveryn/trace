"""Tests for shared 3D chart renderer-family helpers."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.charts.shared.three_d.color import blend_rgb, lighten_rgb, shade_rgb
from trace_tasks.tasks.charts.shared.three_d.geometry import bbox_center, point_bbox, polygon_bbox, round_bbox
from trace_tasks.tasks.charts.shared.three_d.projection import (
    ProjectionBasis2D,
    axis_line_position,
    project_ranged_point_3d,
    surface_plot_basis,
    value_to_unit,
)


def test_three_d_color_helpers_match_depth_formula() -> None:
    assert shade_rgb((100, 150, 250), 0.72) == (72, 108, 180)
    assert lighten_rgb((100, 150, 250), 0.25) == (139, 176, 251)
    assert blend_rgb((10, 30, 50), (110, 130, 150), 0.25) == (35, 55, 75)
    assert blend_rgb((10, 30, 50), (110, 130, 150), -3.0) == (10, 30, 50)
    assert blend_rgb((10, 30, 50), (110, 130, 150), 4.0) == (110, 130, 150)


def test_three_d_geometry_helpers_round_screen_bboxes() -> None:
    assert round_bbox([1.23456, 2.34567, 3.45678, 4.56789]) == [1.235, 2.346, 3.457, 4.568]
    assert point_bbox(20.0, 30.0, 4.5) == [15.5, 25.5, 24.5, 34.5]
    assert bbox_center([10.0, 20.0, 30.0, 50.0]) == [20.0, 35.0]
    assert polygon_bbox([(4.0, 9.0), (2.0, 12.5), (8.25, 6.0)]) == [2.0, 6.0, 8.25, 12.5]


def test_projection_basis_projects_unit_coordinates() -> None:
    basis = ProjectionBasis2D(
        origin=(10.0, 20.0),
        x_vec=(100.0, 0.0),
        y_vec=(0.0, -50.0),
        z_vec=(25.0, -100.0),
    )
    assert basis.project_unit(0.5, 0.25, 0.1) == (62.5, -2.5)


def test_surface_projection_preserves_current_chart_basis() -> None:
    plot_bbox = [100.0, 80.0, 900.0, 680.0]
    basis = surface_plot_basis(plot_bbox)
    assert basis.origin == pytest.approx((244.0, 620.0))
    assert basis.x_vec == pytest.approx((416.0, -84.0))
    assert basis.y_vec == pytest.approx((192.0, -144.0))
    assert basis.z_vec == pytest.approx((0.0, -336.0))
    assert value_to_unit(25.0, (0.0, 100.0)) == 0.25
    assert value_to_unit(25.0, (5.0, 5.0)) == 0.0
    assert project_ranged_point_3d(
        25.0,
        50.0,
        75.0,
        plot_bbox=plot_bbox,
        x_range=(0.0, 100.0),
        y_range=(0.0, 100.0),
        z_range=(0.0, 100.0),
    ) == pytest.approx((444.0, 275.0))


def test_axis_line_position_clamps_fraction() -> None:
    assert axis_line_position((0.0, 10.0), (100.0, 110.0), fraction=0.25) == (25.0, 35.0)
    assert axis_line_position((0.0, 10.0), (100.0, 110.0), fraction=-1.0) == (0.0, 10.0)
    assert axis_line_position((0.0, 10.0), (100.0, 110.0), fraction=2.0) == (100.0, 110.0)
