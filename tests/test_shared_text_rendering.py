from __future__ import annotations

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.text_rendering import draw_text_centered, load_font


def test_draw_text_centered_compensates_for_font_bbox_offsets() -> None:
    image = Image.new("RGB", (120, 120), "white")
    draw = ImageDraw.Draw(image)
    font = load_font(28, bold=True)
    center = (60.0, 60.0)

    bbox = draw_text_centered(
        draw,
        text="10",
        center=center,
        font=font,
        fill=(20, 20, 20),
        stroke_width=0,
        trace=False,
    )

    assert abs(((bbox[0] + bbox[2]) * 0.5) - center[0]) <= 0.01
    assert abs(((bbox[1] + bbox[3]) * 0.5) - center[1]) <= 0.01
