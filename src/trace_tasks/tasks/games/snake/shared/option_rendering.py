"""Visual option drawing for Snake path-result tasks."""

from __future__ import annotations

from typing import Any, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.games.shared.text import draw_game_text_traced as draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font

from .state import SnakeSample


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    text: str,
    font: Any,
    fill: Tuple[int, int, int],
    stroke_fill: Tuple[int, int, int] | None = None,
    stroke_width: int = 0,
) -> None:
    """Draw centered text with a Pillow text-box fallback."""

    try:
        bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width)))
        width = float(bbox[2] - bbox[0])
        height = float(bbox[3] - bbox[1])
        x = float(center[0]) - (width / 2.0) - float(bbox[0])
        y = float(center[1]) - (height / 2.0) - float(bbox[1])
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        x = float(center[0]) - (float(width) / 2.0)
        y = float(center[1]) - (float(height) / 2.0)
    draw_text_traced(
        draw,
        (x, y),
        str(text),
        font=font,
        fill=fill,
        stroke_width=max(0, int(stroke_width)),
        stroke_fill=stroke_fill,
        role="readout",
        required=False,
    )


def draw_path_result_options(
    *,
    image: Image.Image,
    render_map: Mapping[str, Any],
    sample: SnakeSample,
    font_family: str = "",
) -> tuple[Image.Image, dict[str, list[float]]]:
    """Draw image-visible A-D result options for a path-result sample."""

    out = image.convert("RGBA")
    draw = ImageDraw.Draw(out, "RGBA")
    cell_bboxes = render_map.get("cell_bboxes_px", {})
    board_bbox = tuple(float(v) for v in render_map.get("board_bbox_px", (54, 54, 846, 846)))
    option_bboxes: dict[str, list[float]] = {}
    cell_width = (float(board_bbox[2]) - float(board_bbox[0])) / max(1.0, float(sample.state.board_size))
    point_font = load_font(max(18, int(round(cell_width * 0.34))), bold=True, font_family=str(font_family))
    card_font = load_font(max(18, int(round(cell_width * 0.24))), bold=True, font_family=str(font_family))
    label_fill = (17, 24, 39)
    option_fill = (255, 255, 255, 235)
    option_outline = (16, 24, 40, 255)

    game_over_option: Mapping[str, object] | None = None
    for option in sample.result_options:
        label = str(option.get("label", ""))
        if str(option.get("kind")) != "point":
            game_over_option = option
            continue
        cell_id = str(option.get("cell_id", ""))
        if cell_id not in cell_bboxes:
            continue
        bbox = tuple(float(v) for v in cell_bboxes[cell_id])
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        radius = max(15.0, min(float(bbox[2] - bbox[0]), float(bbox[3] - bbox[1])) * 0.33)
        marker_bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
        draw.ellipse(marker_bbox, fill=option_fill, outline=option_outline, width=max(3, int(round(radius * 0.16))))
        _draw_centered_text(
            draw,
            center=(cx, cy),
            text=label,
            font=point_font,
            fill=label_fill,
            stroke_fill=(255, 255, 255),
            stroke_width=1,
        )
        option_bboxes[label] = [round(float(value), 3) for value in marker_bbox]

    if game_over_option is not None:
        label = str(game_over_option.get("label", ""))
        canvas_w, canvas_h = out.size
        card_w = min(260.0, max(170.0, (float(board_bbox[2]) - float(board_bbox[0])) * 0.34))
        card_h = max(48.0, min(68.0, float(canvas_h) - float(board_bbox[3]) - 18.0))
        if card_h < 44.0:
            card_h = 52.0
            top = max(10.0, float(board_bbox[1]) - card_h - 16.0)
        else:
            top = float(board_bbox[3]) + 12.0
        left = min(max(14.0, (float(canvas_w) - card_w) / 2.0), float(canvas_w) - card_w - 14.0)
        card_bbox = (left, top, left + card_w, top + card_h)
        draw.rounded_rectangle(card_bbox, radius=12, fill=(250, 250, 250, 238), outline=option_outline, width=3)
        _draw_centered_text(
            draw,
            center=((card_bbox[0] + card_bbox[2]) / 2.0, (card_bbox[1] + card_bbox[3]) / 2.0),
            text=f"{label}: GAME OVER",
            font=card_font,
            fill=label_fill,
            stroke_fill=(255, 255, 255),
            stroke_width=1,
        )
        option_bboxes[label] = [round(float(value), 3) for value in card_bbox]
    return out.convert("RGB"), option_bboxes


__all__ = ["draw_path_result_options"]
