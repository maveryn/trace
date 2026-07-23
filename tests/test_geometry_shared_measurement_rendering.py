from __future__ import annotations

import inspect

import pytest
from PIL import Image, ImageDraw

from trace_tasks.tasks.geometry.shared.measurement_rendering import (
    assert_bboxes_inside,
    bbox_union_from_bboxes,
    draw_dimension_line,
    draw_readout_centered,
    draw_right_angle_marker,
)
from trace_tasks.tasks.shared.text_rendering import load_font


class _RenderContext:
    def __init__(self) -> None:
        self.width = 240
        self.height = 180
        self.image = Image.new("RGB", (self.width, self.height), (248, 249, 250))
        self.draw = ImageDraw.Draw(self.image)
        self.line_color = (28, 37, 53)
        self.secondary_color = (65, 75, 90)
        self.label_color = (15, 23, 42)
        self.label_stroke_color = (255, 255, 255)
        self.label_backing_color = (255, 255, 255)
        self.line_width = 4
        self.label_stroke_width = 1
        self.font = load_font(20, bold=True)
        self.small_font = load_font(16, bold=True)


def _assert_inside(bbox: tuple[float, float, float, float], *, width: int = 240, height: int = 180) -> None:
    x0, y0, x1, y1 = bbox
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


def test_draw_readout_centered_returns_canvas_bbox() -> None:
    ctx = _RenderContext()
    bbox = draw_readout_centered(ctx, "AB=12", (90, 70), small=True)

    _assert_inside(bbox)


def test_measurement_readout_helpers_default_to_plain_text() -> None:
    assert inspect.signature(draw_readout_centered).parameters["backed"].default is False
    assert inspect.signature(draw_dimension_line).parameters["backed"].default is False


def test_draw_dimension_line_returns_label_bbox() -> None:
    ctx = _RenderContext()
    bbox = draw_dimension_line(
        ctx,
        (35, 130),
        (160, 130),
        "12",
        label_offset=(0, -24),
        tick_px=7.0,
    )

    _assert_inside(bbox)


def test_draw_right_angle_marker_returns_marker_bbox() -> None:
    ctx = _RenderContext()
    bbox = draw_right_angle_marker(
        ctx,
        (70, 70),
        arm_a=(1, 0),
        arm_b=(0, 1),
        side_px=14,
    )

    _assert_inside(bbox)
    assert bbox[0] <= 70 <= bbox[2]
    assert bbox[1] <= 70 <= bbox[3]


def test_assert_bboxes_inside_raises_for_edge_bbox() -> None:
    assert_bboxes_inside([(20, 20, 80, 80)], width=100, height=100, margin=3)

    with pytest.raises(ValueError, match="too close"):
        assert_bboxes_inside(
            [(1, 20, 80, 80)],
            width=100,
            height=100,
            margin=3,
            error_message="too close",
        )


def test_bbox_union_from_bboxes_pads_and_clamps() -> None:
    bbox = bbox_union_from_bboxes(
        ((-2.0, 20.0, 50.0, 80.0), (40.0, 70.0, 130.0, 190.0)),
        width=120,
        height=160,
        pad=5.0,
    )

    assert bbox == (0.0, 15.0, 120.0, 160.0)
