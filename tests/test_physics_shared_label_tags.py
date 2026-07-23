"""Tests for shared physics rounded label-tag helpers."""

from __future__ import annotations

from PIL import Image, ImageDraw

from trace_tasks.tasks.physics.shared.label_tags import draw_text_tag, text_tag_bbox
from trace_tasks.tasks.shared.text_rendering import load_font


def test_text_tag_bbox_expands_centered_text_by_padding() -> None:
    image = Image.new("RGB", (240, 120), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(20, bold=True)

    text_bbox = draw.textbbox((0, 0), "A=5 N", font=font, stroke_width=0)
    tag_bbox = text_tag_bbox(draw, text="A=5 N", center=(120.0, 60.0), font=font, pad_x_px=9.0, pad_y_px=6.0)

    assert tag_bbox[0] == round(120.0 - (text_bbox[2] - text_bbox[0]) * 0.5 - 9.0, 3)
    assert tag_bbox[2] == round(120.0 + (text_bbox[2] - text_bbox[0]) * 0.5 + 9.0, 3)
    assert tag_bbox[1] == round(60.0 - (text_bbox[3] - text_bbox[1]) * 0.5 - 6.0, 3)
    assert tag_bbox[3] == round(60.0 + (text_bbox[3] - text_bbox[1]) * 0.5 + 6.0, 3)


def test_draw_text_tag_returns_bbox_covering_tag_area() -> None:
    image = Image.new("RGB", (240, 120), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(20, bold=True)

    expected_tag_bbox = text_tag_bbox(draw, text="B", center=(80.0, 45.0), font=font)
    rendered_bbox = draw_text_tag(
        draw,
        text="B",
        center=(80.0, 45.0),
        font=font,
        fill_rgb=(255, 255, 255),
        outline_rgb=(40, 60, 80),
        text_rgb=(20, 30, 40),
        stroke_width_px=1,
    )

    assert rendered_bbox[0] <= expected_tag_bbox[0]
    assert rendered_bbox[1] <= expected_tag_bbox[1]
    assert rendered_bbox[2] >= expected_tag_bbox[2]
    assert rendered_bbox[3] >= expected_tag_bbox[3]
