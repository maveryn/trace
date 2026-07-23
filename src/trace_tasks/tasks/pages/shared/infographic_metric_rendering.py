"""Rendering helpers for infographic metric-card page tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.render_variation import apply_layout_jitter_to_margins, resolve_render_rgb
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import fit_font_to_box, load_font
from trace_tasks.tasks.pages.shared.page_semantic_assets import render_page_semantic_asset_rgba, resolve_page_semantic_asset
from trace_tasks.tasks.pages.shared.infographic_metric_common import (
    TASK_ID,
    _INFOGRAPHIC_STYLE_VARIANTS,
    _PALETTE,
    _RENDER_DEFAULTS,
    _STYLE_TITLE_COPY,
    _MetricCard,
    _RenderedInfographic,
    _partition_cards,
)

def _render_style_seed(params: Mapping[str, Any]) -> int:
    try:
        return int(params.get("_render_style_seed", 0) or 0)
    except Exception:
        return 0


def _render_rgb(params: Mapping[str, Any], key: str, fallback: Sequence[int]) -> Tuple[int, int, int]:
    return resolve_render_rgb(
        params,
        _RENDER_DEFAULTS,
        str(key),
        fallback,
        instance_seed=_render_style_seed(params),
        namespace=TASK_ID,
    )


def _text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font: Any,
    *,
    stroke_width: int = 0,
) -> List[float]:
    try:
        bbox = draw.textbbox((float(xy[0]), float(xy[1])), str(text), font=font, stroke_width=int(stroke_width))
        return [float(value) for value in bbox]
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        return [float(xy[0]), float(xy[1]), float(xy[0]) + float(width), float(xy[1]) + float(height)]


def _draw_icon(
    image: Image.Image,
    *,
    bbox: Tuple[float, float, float, float],
    kind: str,
    color: Tuple[int, int, int],
) -> None:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    icon = render_page_semantic_asset_rgba(
        str(kind),
        size_px=(max(1, int(round(x1 - x0))), max(1, int(round(y1 - y0)))),
        tint_rgb=tuple(int(channel) for channel in color),
    )
    paste_x = int(round(x0 + max(0.0, (x1 - x0 - icon.width) * 0.5)))
    paste_y = int(round(y0 + max(0.0, (y1 - y0 - icon.height) * 0.5)))
    image.paste(icon, (paste_x, paste_y), icon)


def _blend_rgb(color_a: Sequence[int], color_b: Sequence[int], weight_b: float) -> Tuple[int, int, int]:
    """Blend two RGB colors with a small deterministic style weight."""

    weight = max(0.0, min(1.0, float(weight_b)))
    return tuple(
        int(round((float(color_a[index]) * (1.0 - weight)) + (float(color_b[index]) * weight)))
        for index in range(3)
    )


def _fit_value_font(draw: ImageDraw.ImageDraw, text: str, *, max_width: float, max_height: float) -> Any:
    return fit_font_to_box(
        draw,
        text=str(text),
        max_width=float(max_width),
        max_height=float(max_height),
        bold=True,
        min_size_px=18,
        max_size_px=42,
        fill_ratio=0.95,
    )


def _draw_metric_card(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    card: _MetricCard,
    card_bbox: Sequence[float],
    infographic_style: str,
    card_fill: Tuple[int, int, int],
    page_outline: Tuple[int, int, int],
    text_rgb: Tuple[int, int, int],
    muted_rgb: Tuple[int, int, int],
    caption_font: Any,
    label_font_base: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    x0, y0, x1, y1 = [float(value) for value in card_bbox]
    card_w = float(x1 - x0)
    card_h = float(y1 - y0)
    card_fill_i = tuple(int(channel) for channel in card_fill)
    if str(infographic_style) in {"kpi_dashboard", "radial_spokes", "circular_sections"}:
        card_fill_i = _blend_rgb(card_fill, card.color_rgb, 0.06)
    elif str(infographic_style) == "staggered_mosaic":
        card_fill_i = _blend_rgb(card_fill, card.color_rgb, 0.035)

    draw.rounded_rectangle((x0, y0, x1, y1), radius=10, fill=card_fill_i, outline=page_outline, width=1)
    if str(infographic_style) == "circular_sections":
        draw.rounded_rectangle((x0, y0, x1, y0 + 7), radius=7, fill=card.color_rgb)
        icon_bbox = (x0 + 9, y0 + 24, x0 + 38, y0 + 53)
    elif str(infographic_style) == "radial_spokes":
        draw.rounded_rectangle((x0, y0, x1, y0 + 7), radius=7, fill=card.color_rgb)
        icon_bbox = (x0 + 15, y0 + 24, x0 + 54, y0 + 63)
    else:
        draw.rounded_rectangle((x0, y0, x0 + 8, y1), radius=8, fill=card.color_rgb)
        icon_bbox = (x0 + 18, y0 + 20, x0 + 60, y0 + 62)
    semantic_icon = resolve_page_semantic_asset(str(card.icon_kind))
    _draw_icon(image, bbox=icon_bbox, kind=card.icon_kind, color=card.color_rgb)

    if str(infographic_style) == "circular_sections":
        text_left = x0 + 44
    else:
        text_left = x0 + (66 if str(infographic_style) == "radial_spokes" else 72)
    label_font = fit_font_to_box(
        draw,
        text=card.label,
        max_width=max(40.0, x1 - text_left - 14),
        max_height=24,
        bold=True,
        min_size_px=12,
        max_size_px=int(label_font_base),
        fill_ratio=0.98,
    )
    label_xy = (text_left, y0 + (20 if str(infographic_style) in {"radial_spokes", "circular_sections"} else 18))
    draw_text_traced(draw,label_xy, card.label, fill=text_rgb, font=label_font, role="readout", required=False)
    label_bbox = _text_bbox(draw, label_xy, card.label, label_font)

    value_font = _fit_value_font(
        draw,
        card.display_text,
        max_width=max(42.0, x1 - text_left - 14),
        max_height=max(30.0, min(44.0, card_h - 58.0)),
    )
    value_xy = (text_left, y0 + (47 if str(infographic_style) in {"radial_spokes", "circular_sections"} else 45))
    draw_text_traced(draw,value_xy, card.display_text, fill=card.color_rgb, font=value_font, role="readout", required=False)
    value_bbox = _text_bbox(draw, value_xy, card.display_text, value_font)

    caption = str(card.caption_text)
    caption_xy = (x0 + (10 if str(infographic_style) == "circular_sections" else 18), y1 - 25)
    draw_text_traced(draw,caption_xy, caption, fill=muted_rgb, font=caption_font, role="readout", required=False)
    caption_bbox = _text_bbox(draw, caption_xy, caption, caption_font)

    trace = {
        "card_id": str(card.card_id),
        "label": str(card.label),
        "value": int(card.value),
        "display_text": str(card.display_text),
        "unit": str(card.unit),
        "section": str(card.section),
        "card_bbox_px": [float(x0), float(y0), float(x1), float(y1)],
        "label_bbox_px": list(label_bbox),
        "value_bbox_px": list(value_bbox),
        "caption_bbox_px": list(caption_bbox),
        "icon_bbox_px": [float(value) for value in icon_bbox],
        "icon_kind": str(card.icon_kind),
        "icon_label": str(semantic_icon.display_label),
        "icon_asset": semantic_icon.to_metadata(),
        "color_rgb": [int(channel) for channel in card.color_rgb],
        "caption_number": int(card.caption_number),
        "caption_text": str(card.caption_text),
    }
    entity = {
        "id": str(card.card_id),
        "type": "infographic_metric_card",
        "bbox_px": [float(x0), float(y0), float(x1), float(y1)],
        "attrs": {
            "label": str(card.label),
            "value": int(card.value),
            "display_text": str(card.display_text),
            "unit": str(card.unit),
            "section": str(card.section),
            "icon_kind": str(card.icon_kind),
            "icon_label": str(semantic_icon.display_label),
            "caption_number": int(card.caption_number),
            "caption_text": str(card.caption_text),
        },
    }
    return trace, entity


def _render_infographic(
    background: Image.Image,
    *,
    cards: Sequence[_MetricCard],
    section_titles: Sequence[str],
    section_card_counts: Sequence[int],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
) -> _RenderedInfographic:
    render_defaults = {**dict(render_defaults), "_render_style_seed": int(instance_seed)}
    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    margin_x = int(group_default(render_defaults, "page_margin_x_px", 34))
    margin_top = int(group_default(render_defaults, "page_margin_top_px", 30))
    margin_bottom = int(group_default(render_defaults, "page_margin_bottom_px", 30))
    margin_left, margin_right, margin_top, margin_bottom, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(margin_x),
        right_px=int(margin_x),
        top_px=int(margin_top),
        bottom_px=int(margin_bottom),
        params=render_defaults,
        defaults=_RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{TASK_ID}.layout",
    )
    section_gap = int(group_default(render_defaults, "section_gap_px", 14))
    card_gap = int(group_default(render_defaults, "card_gap_px", 12))
    section_pad = int(group_default(render_defaults, "section_padding_px", 14))
    header_h = int(group_default(render_defaults, "header_height_px", 62))
    section_header_h = int(group_default(render_defaults, "section_header_height_px", 34))

    title_font = load_font(int(group_default(render_defaults, "title_font_size_px", 30)), bold=True)
    subtitle_font = load_font(int(group_default(render_defaults, "subtitle_font_size_px", 16)), bold=False)
    section_font = load_font(int(group_default(render_defaults, "section_font_size_px", 20)), bold=True)
    label_font_base = int(group_default(render_defaults, "label_font_size_px", 18))
    caption_font = load_font(int(group_default(render_defaults, "caption_font_size_px", 13)), bold=False)

    page_fill = _render_rgb(render_defaults, "page_fill_rgb", (255, 255, 255))
    page_outline = _render_rgb(render_defaults, "page_outline_rgb", (216, 220, 226))
    text_rgb = _render_rgb(render_defaults, "text_rgb", (38, 41, 48))
    muted_rgb = _render_rgb(render_defaults, "muted_text_rgb", (96, 103, 114))
    section_fill = _render_rgb(render_defaults, "section_fill_rgb", (245, 247, 250))
    card_fill = _render_rgb(render_defaults, "card_fill_rgb", (255, 255, 255))
    style_rng = spawn_rng(int(instance_seed), f"{TASK_ID}.infographic_style")
    explicit_style = str(render_defaults.get("infographic_style", "")).strip()
    if explicit_style:
        if explicit_style not in set(_INFOGRAPHIC_STYLE_VARIANTS):
            raise ValueError(f"unsupported infographic_style: {explicit_style}")
        infographic_style = explicit_style
    else:
        infographic_style = str(_INFOGRAPHIC_STYLE_VARIANTS[int(style_rng.randrange(len(_INFOGRAPHIC_STYLE_VARIANTS)))])
    accent_rgb = _PALETTE[int(style_rng.randrange(len(_PALETTE)))]
    accent_alt_rgb = _PALETTE[int(style_rng.randrange(len(_PALETTE)))]
    if str(infographic_style) == "kpi_dashboard":
        page_fill = _blend_rgb((248, 250, 252), accent_rgb, 0.08)
        section_fill = _blend_rgb((255, 255, 255), accent_rgb, 0.10)
        card_fill = _blend_rgb((255, 255, 255), accent_alt_rgb, 0.04)
    elif str(infographic_style) == "staggered_mosaic":
        page_fill = _blend_rgb((250, 251, 252), accent_alt_rgb, 0.05)
        section_fill = _blend_rgb((247, 249, 251), accent_rgb, 0.08)
        card_fill = _blend_rgb((255, 255, 255), accent_alt_rgb, 0.03)
    elif str(infographic_style) == "column_fact_sheet":
        page_fill = _blend_rgb((255, 253, 248), accent_rgb, 0.06)
        section_fill = _blend_rgb((246, 248, 250), accent_alt_rgb, 0.12)
        card_fill = _blend_rgb((255, 255, 255), accent_rgb, 0.03)
    elif str(infographic_style) == "radial_spokes":
        page_fill = _blend_rgb((249, 250, 252), accent_rgb, 0.05)
        section_fill = _blend_rgb((255, 255, 255), accent_alt_rgb, 0.08)
        card_fill = _blend_rgb((255, 255, 255), accent_rgb, 0.02)
    elif str(infographic_style) == "circular_sections":
        page_fill = _blend_rgb((250, 250, 248), accent_alt_rgb, 0.06)
        section_fill = _blend_rgb((255, 255, 255), accent_rgb, 0.10)
        card_fill = _blend_rgb((255, 255, 255), accent_alt_rgb, 0.025)

    page_bbox = (margin_left, margin_top, width - margin_right, height - margin_bottom)
    draw.rounded_rectangle(page_bbox, radius=16, fill=page_fill, outline=page_outline, width=2)
    if str(infographic_style) == "kpi_dashboard":
        draw.rounded_rectangle(
            (margin_left + 14, margin_top + 12, width - margin_right - 14, margin_top + 58),
            radius=12,
            fill=_blend_rgb(accent_rgb, (255, 255, 255), 0.22),
        )
    elif str(infographic_style) == "column_fact_sheet":
        stripe_w = max(18, int((width - margin_left - margin_right) * 0.025))
        draw.rounded_rectangle(
            (margin_left + 14, margin_top + 14, margin_left + 14 + stripe_w, height - margin_bottom - 14),
            radius=10,
            fill=_blend_rgb(accent_rgb, (255, 255, 255), 0.10),
        )
    title_x = margin_left + 26
    title_y = margin_top + 16
    title_text, subtitle_text = _STYLE_TITLE_COPY.get(
        str(infographic_style),
        _STYLE_TITLE_COPY["card_wall"],
    )
    draw_text_traced(draw,(title_x, title_y), title_text, fill=text_rgb, font=title_font, role="readout", required=False)
    draw_text_traced(draw,(title_x, title_y + 36), subtitle_text, fill=muted_rgb, font=subtitle_font, role="readout", required=False)
    document_title_bbox = _text_bbox(draw, (title_x, title_y), str(title_text), title_font)
    document_subtitle_bbox = _text_bbox(draw, (title_x, title_y + 36), str(subtitle_text), subtitle_font)

    section_count = len(section_titles)
    layout_mode = "stacked"
    if str(infographic_style) == "column_fact_sheet":
        layout_mode = "masonry_columns"
    elif str(infographic_style) == "radial_spokes":
        layout_mode = "radial_spokes"
    elif str(infographic_style) == "circular_sections":
        layout_mode = "circular_sections"

    row_counts: List[int] = []
    col_counts: List[int] = []
    max_columns_per_section = 5 if int(section_count) >= 5 else 4
    for section_index, count in enumerate(section_card_counts):
        if str(layout_mode) in {"masonry_columns", "radial_spokes"}:
            columns = min(2, max(1, int(count)))
        elif str(layout_mode) == "circular_sections":
            columns = min(3, max(1, int(count)))
        else:
            columns = min(int(max_columns_per_section), max(1, int(count)))
            if str(infographic_style) == "staggered_mosaic" and int(count) >= 4 and int(section_index) % 2 == 1:
                columns = max(2, int(columns) - 1)
        rows = int(math.ceil(float(count) / float(columns)))
        col_counts.append(int(columns))
        row_counts.append(int(rows))

    content_top = margin_top + header_h + 22
    content_bottom = height - margin_bottom - 18

    card_traces: List[Dict[str, Any]] = []
    entities: List[Dict[str, Any]] = []
    section_bboxes: Dict[str, List[float]] = {}
    section_title_bboxes: Dict[str, List[float]] = {}
    card_index = 0

    if str(layout_mode) == "masonry_columns":
        lane_count = 2
        lane_gap = max(22, int(card_gap) + 10)
        content_left = margin_left + 44
        content_right = width - margin_right - 26
        lane_w = (float(content_right - content_left) - float((lane_count - 1) * lane_gap)) / float(lane_count)
        lane_tops = [float(content_top), float(content_top + 28)]
        card_h = 92
        for section_index, section_title in enumerate(section_titles):
            count = int(section_card_counts[section_index])
            columns = int(col_counts[section_index])
            rows = int(row_counts[section_index])
            lane_index = min(range(lane_count), key=lambda index: lane_tops[int(index)])
            section_x0 = float(content_left + (lane_index * (lane_w + lane_gap)))
            section_y0 = float(lane_tops[int(lane_index)])
            section_h = float(section_header_h + (2 * section_pad) + (rows * card_h) + (max(0, rows - 1) * card_gap))
            section_bbox = (section_x0, section_y0, section_x0 + lane_w, section_y0 + section_h)
            section_bboxes[str(section_title)] = [float(value) for value in section_bbox]
            section_fill_i = _blend_rgb(section_fill, _PALETTE[section_index % len(_PALETTE)], 0.08)
            draw.rounded_rectangle(section_bbox, radius=12, fill=section_fill_i, outline=page_outline, width=1)
            draw.rectangle(
                (section_bbox[0], section_bbox[1], section_bbox[0] + 8, section_bbox[3]),
                fill=_PALETTE[section_index % len(_PALETTE)],
            )
            title_xy = (section_bbox[0] + section_pad, section_bbox[1] + 8)
            draw_text_traced(draw,title_xy, str(section_title), fill=text_rgb, font=section_font, role="readout", required=False)
            section_title_bboxes[str(section_title)] = _text_bbox(draw, title_xy, str(section_title), section_font)

            grid_left = section_bbox[0] + section_pad
            grid_top = section_bbox[1] + section_header_h + section_pad
            grid_w = section_bbox[2] - section_bbox[0] - (2 * section_pad)
            card_w = (grid_w - ((columns - 1) * card_gap)) / float(columns)
            for local_index in range(count):
                card = cards[card_index]
                row = local_index // columns
                col = local_index % columns
                x0 = grid_left + (col * (card_w + card_gap))
                y0 = grid_top + (row * (card_h + card_gap))
                trace, entity = _draw_metric_card(
                    image,
                    draw,
                    card=card,
                    card_bbox=[float(x0), float(y0), float(x0 + card_w), float(y0 + card_h)],
                    infographic_style=str(infographic_style),
                    card_fill=card_fill,
                    page_outline=page_outline,
                    text_rgb=text_rgb,
                    muted_rgb=muted_rgb,
                    caption_font=caption_font,
                    label_font_base=label_font_base,
                )
                card_traces.append(dict(trace))
                entities.append(dict(entity))
                card_index += 1
            lane_tops[int(lane_index)] = float(section_y0 + section_h + section_gap + (6 if int(section_index) % 2 else 0))

        return _RenderedInfographic(
            image=image,
            entities=entities,
            card_traces=card_traces,
            page_bbox=[float(value) for value in page_bbox],
            document_title_bbox=[float(value) for value in document_title_bbox],
            document_subtitle_bbox=[float(value) for value in document_subtitle_bbox],
            section_bboxes=section_bboxes,
            section_title_bboxes=section_title_bboxes,
            layout_jitter_meta={
                **dict(layout_jitter_meta),
                "infographic_style": str(infographic_style),
                "layout_mode": str(layout_mode),
            },
        )

    if str(layout_mode) == "circular_sections":
        content_left = margin_left + 26
        content_right = width - margin_right - 26
        panel_gap_x = 28
        panel_gap_y = 18
        row_slots = 2 if int(section_count) <= 4 else 3
        panel_w = (float(content_right - content_left) - panel_gap_x) / 2.0
        panel_h = (float(content_bottom - content_top) - ((row_slots - 1) * panel_gap_y)) / float(row_slots)
        if int(section_count) == 4:
            section_slots = [(0.0, 0), (1.0, 0), (0.0, 1), (1.0, 1)]
        elif int(section_count) == 5:
            section_slots = [(0.0, 0), (1.0, 0), (0.0, 1), (1.0, 1), (0.5, 2)]
        else:
            section_slots = [(0.0, 0), (1.0, 0), (0.0, 1), (1.0, 1), (0.0, 2), (1.0, 2)]
        hub_x = (float(content_left) + float(content_right)) * 0.5
        hub_y = float(content_top) + ((float(content_bottom) - float(content_top)) * 0.5)
        draw.ellipse(
            (hub_x - 36, hub_y - 36, hub_x + 36, hub_y + 36),
            fill=_blend_rgb(accent_rgb, (255, 255, 255), 0.24),
            outline=_blend_rgb(page_outline, accent_rgb, 0.35),
            width=2,
        )
        draw.ellipse(
            (hub_x - 9, hub_y - 9, hub_x + 9, hub_y + 9),
            fill=accent_rgb,
        )

        for section_index in range(int(section_count)):
            slot_x, slot_row = section_slots[int(section_index)]
            section_x0 = float(content_left) + float(slot_x) * (float(content_right - content_left) - float(panel_w))
            section_y0 = float(content_top) + (float(slot_row) * (float(panel_h) + float(panel_gap_y)))
            panel_center = (section_x0 + (panel_w * 0.5), section_y0 + (panel_h * 0.5))
            draw.line(
                (hub_x, hub_y, panel_center[0], panel_center[1]),
                fill=_blend_rgb(page_outline, accent_alt_rgb, 0.34),
                width=2,
            )

        ring_slot_order = (
            (1, 0),
            (2, 0),
            (2, 1),
            (2, 2),
            (1, 2),
            (0, 2),
            (0, 1),
            (0, 0),
        )
        ring_indices_by_count: Dict[int, Tuple[int, ...]] = {
            1: (0,),
            2: (6, 2),
            3: (0, 2, 6),
            4: (0, 2, 4, 6),
            5: (0, 2, 3, 5, 6),
            6: (0, 1, 2, 4, 5, 6),
            7: (0, 1, 2, 3, 4, 5, 6),
            8: (0, 1, 2, 3, 4, 5, 6, 7),
        }

        for section_index, section_title in enumerate(section_titles):
            count = int(section_card_counts[section_index])
            slot_x, slot_row = section_slots[int(section_index)]
            section_x0 = float(content_left) + float(slot_x) * (float(content_right - content_left) - float(panel_w))
            section_y0 = float(content_top) + (float(slot_row) * (float(panel_h) + float(panel_gap_y)))
            section_bbox = (section_x0, section_y0, section_x0 + panel_w, section_y0 + panel_h)
            section_bboxes[str(section_title)] = [float(value) for value in section_bbox]
            section_fill_i = _blend_rgb(section_fill, _PALETTE[section_index % len(_PALETTE)], 0.07)
            draw.rounded_rectangle(section_bbox, radius=20, fill=section_fill_i, outline=page_outline, width=1)
            draw.ellipse(
                (
                    section_bbox[0] + 12,
                    section_bbox[1] + 12,
                    section_bbox[2] - 12,
                    section_bbox[3] - 12,
                ),
                outline=_blend_rgb(_PALETTE[section_index % len(_PALETTE)], page_outline, 0.45),
                width=2,
            )

            grid_pad = 10.0
            ring_gap = 7.0
            grid_left = section_bbox[0] + grid_pad
            grid_top = section_bbox[1] + grid_pad
            grid_w = section_bbox[2] - section_bbox[0] - (2.0 * grid_pad)
            grid_h = section_bbox[3] - section_bbox[1] - (2.0 * grid_pad)
            card_w = (grid_w - (2.0 * ring_gap)) / 3.0
            card_h = max(78.0, min(96.0, (grid_h - (2.0 * ring_gap)) / 3.0))
            grid_y = grid_top + max(0.0, (grid_h - ((3.0 * card_h) + (2.0 * ring_gap))) * 0.5)

            title_pill_w = min(card_w + 18.0, grid_w * 0.38)
            title_pill_h = 31.0
            center_x = grid_left + card_w + ring_gap + (card_w * 0.5)
            center_y = grid_y + card_h + ring_gap + (card_h * 0.5)
            title_bbox = (
                center_x - (title_pill_w * 0.5),
                center_y - (title_pill_h * 0.5),
                center_x + (title_pill_w * 0.5),
                center_y + (title_pill_h * 0.5),
            )
            draw.rounded_rectangle(
                title_bbox,
                radius=14,
                fill=_blend_rgb((255, 255, 255), _PALETTE[section_index % len(_PALETTE)], 0.05),
                outline=_blend_rgb(page_outline, _PALETTE[section_index % len(_PALETTE)], 0.22),
                width=1,
            )
            title_font_fit = fit_font_to_box(
                draw,
                text=str(section_title),
                max_width=max(52.0, title_pill_w - 12.0),
                max_height=20.0,
                bold=True,
                min_size_px=9,
                max_size_px=14,
                fill_ratio=0.96,
            )
            title_text_bbox = _text_bbox(draw, (0, 0), str(section_title), title_font_fit)
            title_text_w = float(title_text_bbox[2] - title_text_bbox[0])
            title_text_h = float(title_text_bbox[3] - title_text_bbox[1])
            title_xy = (center_x - (title_text_w * 0.5), center_y - (title_text_h * 0.62))
            draw_text_traced(draw,
                title_xy,
                str(section_title),
                fill=text_rgb,
                font=title_font_fit,
             role="readout", required=False,)
            section_title_bboxes[str(section_title)] = _text_bbox(draw, title_xy, str(section_title), title_font_fit)

            slot_indices = ring_indices_by_count.get(int(count), ring_indices_by_count[8])
            for local_index in range(count):
                card = cards[card_index]
                slot_col, slot_row_inner = ring_slot_order[int(slot_indices[int(local_index) % len(slot_indices)])]
                x0 = grid_left + (float(slot_col) * (card_w + ring_gap))
                y0 = grid_y + (float(slot_row_inner) * (card_h + ring_gap))
                trace, entity = _draw_metric_card(
                    image,
                    draw,
                    card=card,
                    card_bbox=[float(x0), float(y0), float(x0 + card_w), float(y0 + card_h)],
                    infographic_style=str(infographic_style),
                    card_fill=card_fill,
                    page_outline=page_outline,
                    text_rgb=text_rgb,
                    muted_rgb=muted_rgb,
                    caption_font=caption_font,
                    label_font_base=label_font_base,
                )
                card_traces.append(dict(trace))
                entities.append(dict(entity))
                card_index += 1

        return _RenderedInfographic(
            image=image,
            entities=entities,
            card_traces=card_traces,
            page_bbox=[float(value) for value in page_bbox],
            document_title_bbox=[float(value) for value in document_title_bbox],
            document_subtitle_bbox=[float(value) for value in document_subtitle_bbox],
            section_bboxes=section_bboxes,
            section_title_bboxes=section_title_bboxes,
            layout_jitter_meta={
                **dict(layout_jitter_meta),
                "infographic_style": str(infographic_style),
                "layout_mode": str(layout_mode),
            },
        )

    if str(layout_mode) == "radial_spokes":
        content_left = margin_left + 26
        content_right = width - margin_right - 26
        panel_gap_x = 28
        panel_gap_y = 18
        row_slots = 2 if int(section_count) <= 4 else 3
        panel_w = (float(content_right - content_left) - panel_gap_x) / 2.0
        panel_h = (float(content_bottom - content_top) - ((row_slots - 1) * panel_gap_y)) / float(row_slots)
        if int(section_count) == 4:
            slots = [(0.0, 0), (1.0, 0), (0.0, 1), (1.0, 1)]
        elif int(section_count) == 5:
            slots = [(0.0, 0), (1.0, 0), (0.0, 1), (1.0, 1), (0.5, 2)]
        else:
            slots = [(0.0, 0), (1.0, 0), (0.0, 1), (1.0, 1), (0.0, 2), (1.0, 2)]
        hub_x = (float(content_left) + float(content_right)) * 0.5
        hub_y = float(content_top) + ((float(content_bottom) - float(content_top)) * 0.5)
        for section_index in range(int(section_count)):
            slot_x, slot_row = slots[int(section_index)]
            section_x0 = float(content_left) + float(slot_x) * (float(content_right - content_left) - float(panel_w))
            section_y0 = float(content_top) + (float(slot_row) * (float(panel_h) + float(panel_gap_y)))
            panel_center = (section_x0 + (panel_w * 0.5), section_y0 + (panel_h * 0.5))
            draw.line(
                (hub_x, hub_y, panel_center[0], panel_center[1]),
                fill=_blend_rgb(page_outline, accent_rgb, 0.30),
                width=2,
            )
        draw.ellipse(
            (hub_x - 10, hub_y - 10, hub_x + 10, hub_y + 10),
            fill=_blend_rgb(accent_rgb, (255, 255, 255), 0.20),
            outline=page_outline,
            width=2,
        )

        for section_index, section_title in enumerate(section_titles):
            count = int(section_card_counts[section_index])
            columns = int(col_counts[section_index])
            rows = int(row_counts[section_index])
            slot_x, slot_row = slots[int(section_index)]
            section_x0 = float(content_left) + float(slot_x) * (float(content_right - content_left) - float(panel_w))
            section_y0 = float(content_top) + (float(slot_row) * (float(panel_h) + float(panel_gap_y)))
            card_h = int(max(84, min(118, math.floor((panel_h - section_header_h - (2 * section_pad) - (max(0, rows - 1) * card_gap)) / max(1, rows)))))
            section_h = float(section_header_h + (2 * section_pad) + (rows * card_h) + (max(0, rows - 1) * card_gap))
            section_bbox = (section_x0, section_y0, section_x0 + panel_w, section_y0 + section_h)
            section_bboxes[str(section_title)] = [float(value) for value in section_bbox]
            section_fill_i = _blend_rgb(section_fill, _PALETTE[section_index % len(_PALETTE)], 0.08)
            draw.rounded_rectangle(section_bbox, radius=14, fill=section_fill_i, outline=page_outline, width=1)
            title_xy = (section_bbox[0] + section_pad, section_bbox[1] + 8)
            draw_text_traced(draw,title_xy, str(section_title), fill=text_rgb, font=section_font, role="readout", required=False)
            section_title_bboxes[str(section_title)] = _text_bbox(draw, title_xy, str(section_title), section_font)

            grid_left = section_bbox[0] + section_pad
            grid_top = section_bbox[1] + section_header_h + section_pad
            grid_w = section_bbox[2] - section_bbox[0] - (2 * section_pad)
            card_w = (grid_w - ((columns - 1) * card_gap)) / float(columns)
            for local_index in range(count):
                card = cards[card_index]
                row = local_index // columns
                col = local_index % columns
                x0 = grid_left + (col * (card_w + card_gap))
                y0 = grid_top + (row * (card_h + card_gap))
                trace, entity = _draw_metric_card(
                    image,
                    draw,
                    card=card,
                    card_bbox=[float(x0), float(y0), float(x0 + card_w), float(y0 + card_h)],
                    infographic_style=str(infographic_style),
                    card_fill=card_fill,
                    page_outline=page_outline,
                    text_rgb=text_rgb,
                    muted_rgb=muted_rgb,
                    caption_font=caption_font,
                    label_font_base=label_font_base,
                )
                card_traces.append(dict(trace))
                entities.append(dict(entity))
                card_index += 1

        return _RenderedInfographic(
            image=image,
            entities=entities,
            card_traces=card_traces,
            page_bbox=[float(value) for value in page_bbox],
            document_title_bbox=[float(value) for value in document_title_bbox],
            document_subtitle_bbox=[float(value) for value in document_subtitle_bbox],
            section_bboxes=section_bboxes,
            section_title_bboxes=section_title_bboxes,
            layout_jitter_meta={
                **dict(layout_jitter_meta),
                "infographic_style": str(infographic_style),
                "layout_mode": str(layout_mode),
            },
        )

    total_rows = max(1, sum(row_counts))
    available = float(content_bottom - content_top - ((section_count - 1) * section_gap))
    fixed = float(section_count * (section_header_h + (2 * section_pad)))
    card_h = int(max(86, min(136, math.floor((available - fixed - (sum(max(0, rows - 1) for rows in row_counts) * card_gap)) / total_rows))))

    y = float(content_top)
    for section_index, section_title in enumerate(section_titles):
        count = int(section_card_counts[section_index])
        columns = int(col_counts[section_index])
        rows = int(row_counts[section_index])
        section_h = float(section_header_h + (2 * section_pad) + (rows * card_h) + (max(0, rows - 1) * card_gap))
        left_extra = 0
        right_extra = 0
        if str(infographic_style) == "staggered_mosaic":
            left_extra = (0, 38, 10, 56, 24, 44)[int(section_index) % 6]
            right_extra = (46, 8, 40, 0, 54, 18)[int(section_index) % 6]
        section_bbox = (
            margin_left + 18 + float(left_extra),
            y,
            width - margin_right - 18 - float(right_extra),
            y + section_h,
        )
        section_bboxes[str(section_title)] = [float(value) for value in section_bbox]
        section_fill_i = section_fill
        if str(infographic_style) in {"kpi_dashboard", "staggered_mosaic"}:
            section_fill_i = _blend_rgb(section_fill, _PALETTE[section_index % len(_PALETTE)], 0.06)
        draw.rounded_rectangle(section_bbox, radius=12, fill=section_fill_i, outline=page_outline, width=1)
        title_xy = (section_bbox[0] + section_pad, section_bbox[1] + 8)
        draw_text_traced(draw,title_xy, str(section_title), fill=text_rgb, font=section_font, role="readout", required=False)
        section_title_bboxes[str(section_title)] = _text_bbox(draw, title_xy, str(section_title), section_font)

        grid_left = section_bbox[0] + section_pad
        grid_top = section_bbox[1] + section_header_h + section_pad
        grid_w = section_bbox[2] - section_bbox[0] - (2 * section_pad)
        card_w = (grid_w - ((columns - 1) * card_gap)) / float(columns)
        for local_index in range(count):
            card = cards[card_index]
            row = local_index // columns
            col = local_index % columns
            x0 = grid_left + (col * (card_w + card_gap))
            y0 = grid_top + (row * (card_h + card_gap))
            trace, entity = _draw_metric_card(
                image,
                draw,
                card=card,
                card_bbox=[float(x0), float(y0), float(x0 + card_w), float(y0 + card_h)],
                infographic_style=str(infographic_style),
                card_fill=card_fill,
                page_outline=page_outline,
                text_rgb=text_rgb,
                muted_rgb=muted_rgb,
                caption_font=caption_font,
                label_font_base=label_font_base,
            )
            card_traces.append(dict(trace))
            entities.append(dict(entity))
            card_index += 1
        y += section_h + section_gap

    return _RenderedInfographic(
        image=image,
        entities=entities,
        card_traces=card_traces,
        page_bbox=[float(value) for value in page_bbox],
        document_title_bbox=[float(value) for value in document_title_bbox],
        document_subtitle_bbox=[float(value) for value in document_subtitle_bbox],
        section_bboxes=section_bboxes,
        section_title_bboxes=section_title_bboxes,
        layout_jitter_meta={
            **dict(layout_jitter_meta),
            "infographic_style": str(infographic_style),
            "layout_mode": str(layout_mode),
        },
    )
