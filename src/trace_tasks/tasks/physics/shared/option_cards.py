"""Shared option-card rendering helpers for physics diagrams."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal, Sequence, Tuple

from PIL import ImageDraw

from ...shared.bbox_projection import bbox_union_many
from ...shared.drawing import draw_centered_text
from ...shared.text_rendering import resolve_text_stroke_fill

RGB = Tuple[int, int, int]
BBox = List[float]
TextAlign = Literal["left", "center"]


@dataclass(frozen=True)
class OptionCardRenderResult:
    """Rendered option-card bboxes keyed by visible option letter."""

    option_bboxes: Dict[str, BBox]
    option_letter_bboxes: Dict[str, BBox]
    option_text_bboxes: Dict[str, BBox]
    panel_bbox: BBox | None


def _bbox(values: Sequence[float]) -> BBox:
    return [round(float(value), 3) for value in values]


def _draw_multiline_centered_text(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font: object,
    fill_rgb: RGB,
    line_spacing_px: float,
    stroke_width: int,
) -> BBox:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return _bbox((center[0], center[1], center[0], center[1]))

    line_boxes = [draw.textbbox((0, 0), line, font=font, stroke_width=int(stroke_width)) for line in lines]
    line_heights = [float(box[3] - box[1]) for box in line_boxes]
    total_height = sum(line_heights) + float(line_spacing_px) * max(0, len(lines) - 1)
    y_cursor = float(center[1]) - total_height * 0.5
    rendered_bboxes: List[BBox] = []
    for line, line_height in zip(lines, line_heights):
        line_center = (float(center[0]), float(y_cursor + line_height * 0.5))
        rendered_bboxes.append(
            draw_centered_text(
                draw,
                text=str(line),
                center=line_center,
                font=font,
                fill=fill_rgb,
                stroke_fill=resolve_text_stroke_fill(fill_rgb),
                stroke_width=int(stroke_width),
            )
        )
        y_cursor += float(line_height + line_spacing_px)
    return _bbox(bbox_union_many(*rendered_bboxes))


def _text_width(draw: ImageDraw.ImageDraw, *, text: str, font: object, stroke_width: int) -> float:
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    if not lines:
        return 0.0
    return max(
        float(box[2] - box[0])
        for box in (draw.textbbox((0, 0), line, font=font, stroke_width=int(stroke_width)) for line in lines)
    )


def draw_lettered_option_cards(
    draw: ImageDraw.ImageDraw,
    *,
    options: Iterable[Tuple[str, str]],
    option_left: float,
    option_top: float,
    card_width: float,
    card_height: float,
    card_gap_x: float,
    card_gap_y: float,
    columns: int,
    option_font: object,
    letter_font: object,
    text_rgb: RGB,
    card_fill_rgb: RGB,
    card_outline_rgb: RGB,
    label_fill_rgb: RGB,
    label_outline_rgb: RGB,
    label_text_rgb: RGB,
    card_radius_px: float = 12.0,
    card_outline_width_px: int = 2,
    label_radius_px: float = 10.0,
    label_outline_width_px: int = 2,
    label_stroke_width_px: int = 1,
    text_stroke_width_px: int = 1,
    label_width_px: float = 44.0,
    label_height_px: float = 42.0,
    label_center_x_offset_px: float = 34.0,
    text_left_offset_px: float = 76.0,
    text_right_padding_px: float = 16.0,
    text_align: TextAlign = "left",
    line_spacing_px: float = 5.0,
    panel_fill_rgb: RGB | None = None,
    panel_outline_rgb: RGB | None = None,
    panel_padding_px: float = 20.0,
    panel_radius_px: float = 16.0,
    panel_outline_width_px: int = 3,
) -> OptionCardRenderResult:
    """Draw a deterministic grid of lettered option cards.

    The helper reserves a fixed letter-badge column and reports both badge and
    text bboxes so tasks can regression-test that answer options remain visual
    choices without badge/text overlap.
    """

    option_list = [(str(letter), str(text)) for letter, text in options]
    if int(columns) < 1:
        raise ValueError("columns must be at least 1")

    rows = (len(option_list) + int(columns) - 1) // int(columns)
    panel_bbox: BBox | None = None
    if panel_fill_rgb is not None or panel_outline_rgb is not None:
        panel_width = float(columns) * float(card_width) + max(0, int(columns) - 1) * float(card_gap_x)
        panel_height = float(rows) * float(card_height) + max(0, rows - 1) * float(card_gap_y)
        panel_bbox = _bbox(
            (
                float(option_left) - float(panel_padding_px),
                float(option_top) - float(panel_padding_px),
                float(option_left) + panel_width + float(panel_padding_px),
                float(option_top) + panel_height + float(panel_padding_px),
            )
        )
        draw.rounded_rectangle(
            tuple(panel_bbox),
            radius=float(panel_radius_px),
            fill=tuple(int(v) for v in panel_fill_rgb) if panel_fill_rgb is not None else None,
            outline=tuple(int(v) for v in panel_outline_rgb) if panel_outline_rgb is not None else None,
            width=int(panel_outline_width_px),
        )

    option_bboxes: Dict[str, BBox] = {}
    option_letter_bboxes: Dict[str, BBox] = {}
    option_text_bboxes: Dict[str, BBox] = {}
    for index, (letter, text) in enumerate(option_list):
        col = int(index % int(columns))
        row = int(index // int(columns))
        card_left = float(option_left) + float(col) * (float(card_width) + float(card_gap_x))
        card_top = float(option_top) + float(row) * (float(card_height) + float(card_gap_y))
        card_bbox = _bbox((card_left, card_top, card_left + float(card_width), card_top + float(card_height)))
        draw.rounded_rectangle(
            tuple(card_bbox),
            radius=float(card_radius_px),
            fill=tuple(int(v) for v in card_fill_rgb),
            outline=tuple(int(v) for v in card_outline_rgb),
            width=int(card_outline_width_px),
        )

        label_center = (card_left + float(label_center_x_offset_px), card_top + float(card_height) * 0.5)
        label_bbox = _bbox(
            (
                label_center[0] - float(label_width_px) * 0.5,
                label_center[1] - float(label_height_px) * 0.5,
                label_center[0] + float(label_width_px) * 0.5,
                label_center[1] + float(label_height_px) * 0.5,
            )
        )
        draw.rounded_rectangle(
            tuple(label_bbox),
            radius=float(label_radius_px),
            fill=tuple(int(v) for v in label_fill_rgb),
            outline=tuple(int(v) for v in label_outline_rgb),
            width=int(label_outline_width_px),
        )
        letter_bbox = draw_centered_text(
            draw,
            text=str(letter),
            center=label_center,
            font=letter_font,
            fill=label_text_rgb,
            stroke_fill=resolve_text_stroke_fill(label_text_rgb),
            stroke_width=int(label_stroke_width_px),
        )
        letter_bbox = _bbox(bbox_union_many(label_bbox, letter_bbox))

        text_area_left = card_left + float(text_left_offset_px)
        text_area_right = card_left + float(card_width) - float(text_right_padding_px)
        if text_align == "center":
            text_center_x = (text_area_left + text_area_right) * 0.5
        else:
            text_width = _text_width(draw, text=str(text), font=option_font, stroke_width=int(text_stroke_width_px))
            text_center_x = text_area_left + min(text_width, max(1.0, text_area_right - text_area_left)) * 0.5
        text_bbox = _draw_multiline_centered_text(
            draw,
            text=str(text),
            center=(float(text_center_x), card_top + float(card_height) * 0.5),
            font=option_font,
            fill_rgb=text_rgb,
            line_spacing_px=float(line_spacing_px),
            stroke_width=int(text_stroke_width_px),
        )

        option_bboxes[str(letter)] = _bbox(bbox_union_many(card_bbox, letter_bbox, text_bbox))
        option_letter_bboxes[str(letter)] = list(letter_bbox)
        option_text_bboxes[str(letter)] = list(text_bbox)

    return OptionCardRenderResult(
        option_bboxes=option_bboxes,
        option_letter_bboxes=option_letter_bboxes,
        option_text_bboxes=option_text_bboxes,
        panel_bbox=panel_bbox,
    )


__all__ = ["OptionCardRenderResult", "draw_lettered_option_cards"]
