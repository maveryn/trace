"""Visual option-strip rendering for pedigree-chart tasks."""

from __future__ import annotations

from typing import Dict, Mapping, Tuple

from PIL import ImageDraw

from ....shared.mcq import option_label_for_index
from ....shared.text_rendering import load_font
from ....shared.text_legibility import draw_text_traced

OPTION_LABELS: Tuple[str, ...] = tuple(option_label_for_index(index) for index in range(6))


def draw_pedigree_options(
    *,
    image,
    render_params,
    option_values_by_label: Mapping[str, str],
) -> Dict[str, list[int]]:
    """Draw the fixed six-option strip and return option bboxes."""

    draw = ImageDraw.Draw(image)
    panel_left = int(render_params.outer_margin_px + render_params.panel_padding_px + 18)
    panel_right = int(render_params.canvas_width - render_params.outer_margin_px - render_params.panel_padding_px - 18)
    panel_bottom = int(render_params.canvas_height - render_params.outer_margin_px - render_params.panel_padding_px - 12)
    panel_top = int(panel_bottom - 42)
    gap = 8
    option_count = len(OPTION_LABELS)
    cell_width = int((panel_right - panel_left - ((option_count - 1) * gap)) / option_count)
    option_font = load_font(
        max(15, int(round(render_params.label_font_size_px * 0.92))),
        bold=True,
        font_family=str(render_params.font_family),
    )
    option_bboxes: Dict[str, list[int]] = {}
    for index, option_label in enumerate(OPTION_LABELS):
        x0 = int(panel_left + (index * (cell_width + gap)))
        y0 = int(panel_top)
        x1 = int(x0 + cell_width)
        y1 = int(panel_bottom)
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=10,
            fill=tuple(int(value) for value in render_params.panel_fill_rgb),
            outline=tuple(int(value) for value in render_params.panel_border_rgb),
            width=2,
        )
        text = f"{option_label}: {option_values_by_label[str(option_label)]}"
        text_bbox = draw.textbbox((0, 0), text, font=option_font)
        text_width = int(text_bbox[2] - text_bbox[0])
        text_height = int(text_bbox[3] - text_bbox[1])
        draw_text_traced(
            draw,
            (int(x0 + ((cell_width - text_width) / 2)), int(y0 + ((y1 - y0 - text_height) / 2) - 1)),
            text,
            fill=tuple(int(value) for value in render_params.title_color_rgb),
            font=option_font,
            role="graph_pedigree_option_text",
            required=False,
        )
        option_bboxes[str(option_label)] = [int(x0), int(y0), int(x1), int(y1)]
    return dict(option_bboxes)


__all__ = ["OPTION_LABELS", "draw_pedigree_options"]
