"""Rendering primitives for treemap charts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from PIL import Image, ImageDraw, ImageFont

from trace_tasks.tasks.charts.shared.information_style import ChartInformationStyle
from trace_tasks.tasks.charts.shared.dense_text import (
    DENSE_TEXT_DARK_RGB,
    DENSE_TEXT_MUTED_RGB,
    dense_stroke_width,
    dense_text_style_meta,
    lighten_for_dense_text,
)
from trace_tasks.tasks.shared.bbox_projection import bbox_union_raw as _bbox_union
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import render_default, render_int, render_rgb
from .sampling import darken, lighten
from .state import BBox, RGB, RenderedTreemap, TreemapDataset, TreemapRenderParams


def _bbox(values: Sequence[float]) -> BBox:
    return [round(float(value), 3) for value in values]


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    return load_font(max(8, int(size)), bold=bool(bold))


def _truncate(text: str, max_chars: int) -> str:
    clean = str(text)
    if len(clean) <= int(max_chars):
        return clean
    return clean[: max(1, int(max_chars) - 1)] + "."


def resolve_treemap_render_params(params: Mapping[str, object], *, instance_seed: int) -> TreemapRenderParams:
    return TreemapRenderParams(
        canvas_width=int(render_default(params, "canvas_width", 1280)),
        canvas_height=int(render_default(params, "canvas_height", 940)),
        plot_margin_left_px=render_int(params, "plot_margin_left_px", 62, instance_seed=int(instance_seed)),
        plot_margin_right_px=render_int(params, "plot_margin_right_px", 78, instance_seed=int(instance_seed)),
        plot_margin_top_px=render_int(params, "plot_margin_top_px", 78, instance_seed=int(instance_seed)),
        plot_margin_bottom_px=render_int(params, "plot_margin_bottom_px", 56, instance_seed=int(instance_seed)),
        title_font_size_px=render_int(params, "title_font_size_px", 26, instance_seed=int(instance_seed)),
        parent_font_size_px=render_int(params, "parent_font_size_px", 18, instance_seed=int(instance_seed)),
        leaf_font_size_px=render_int(params, "leaf_font_size_px", 15, instance_seed=int(instance_seed)),
        value_font_size_px=render_int(params, "value_font_size_px", 16, instance_seed=int(instance_seed)),
        note_font_size_px=render_int(params, "note_font_size_px", 14, instance_seed=int(instance_seed)),
        panel_fill_rgb=render_rgb(params, "panel_fill_rgb", (252, 253, 250), instance_seed=int(instance_seed)),
        panel_border_rgb=render_rgb(params, "panel_border_rgb", (44, 50, 60), instance_seed=int(instance_seed)),
        text_color_rgb=render_rgb(params, "text_color_rgb", (26, 31, 38), instance_seed=int(instance_seed)),
        muted_text_rgb=render_rgb(params, "muted_text_rgb", (78, 86, 98), instance_seed=int(instance_seed)),
        separator_rgb=render_rgb(params, "separator_rgb", (246, 248, 250), instance_seed=int(instance_seed)),
        text_stroke_rgb=render_rgb(params, "text_stroke_rgb", (255, 255, 255), instance_seed=int(instance_seed)),
        label_stroke_width_px=render_int(params, "label_stroke_width_px", 1, instance_seed=int(instance_seed)),
    )


def apply_treemap_information_style(
    render_params: TreemapRenderParams,
    style: ChartInformationStyle,
) -> TreemapRenderParams:
    """Apply non-semantic information style roles to treemap chrome."""

    return replace(
        render_params,
        panel_fill_rgb=tuple(int(value) for value in style.panel_fill_rgb),
        panel_border_rgb=tuple(int(value) for value in style.panel_border_rgb),
        text_color_rgb=tuple(int(value) for value in style.text_rgb),
        muted_text_rgb=tuple(int(value) for value in style.muted_text_rgb),
        separator_rgb=tuple(int(value) for value in style.guide_rgb),
        text_stroke_rgb=tuple(int(value) for value in style.text_stroke_rgb),
    )


def _slice_rects(
    items: Sequence[tuple[str, int]],
    rect: tuple[float, float, float, float],
    *,
    horizontal: bool,
) -> dict[str, tuple[float, float, float, float]]:
    total = max(1.0, float(sum(max(0, int(value)) for _, value in items)))
    x0, y0, x1, y1 = (float(value) for value in rect)
    cursor = x0 if horizontal else y0
    rects: dict[str, tuple[float, float, float, float]] = {}
    for index, (item_id, value) in enumerate(items):
        share = float(max(0, int(value))) / total
        if horizontal:
            next_cursor = x1 if index == len(items) - 1 else cursor + (x1 - x0) * share
            rects[str(item_id)] = (cursor, y0, next_cursor, y1)
        else:
            next_cursor = y1 if index == len(items) - 1 else cursor + (y1 - y0) * share
            rects[str(item_id)] = (x0, cursor, x1, next_cursor)
        cursor = next_cursor
    return rects


def _text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    *,
    font: ImageFont.ImageFont,
    anchor: str,
    stroke_width: int = 0,
) -> BBox:
    return _bbox(draw.textbbox(xy, str(text), font=font, anchor=anchor, stroke_width=max(0, int(stroke_width))))


def _relative_luminance(color: RGB) -> float:
    def _linear(channel: int) -> float:
        value = float(channel) / 255.0
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    red, green, blue = (_linear(int(channel)) for channel in color)
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def _text_on_color(fill_rgb: RGB) -> tuple[RGB, RGB]:
    """Return the dense semantic treemap text style for one already-lightened fill."""

    del fill_rgb
    return DENSE_TEXT_DARK_RGB, DENSE_TEXT_MUTED_RGB


def _draw_label_value(
    draw: ImageDraw.ImageDraw,
    rect: tuple[float, float, float, float],
    *,
    label: str,
    value: int,
    text_color: RGB,
    muted_text: RGB,
    stroke: RGB,
    label_font: ImageFont.ImageFont,
    value_font: ImageFont.ImageFont,
    stroke_width: int,
) -> BBox:
    """Draw the readable leaf label/value pair and return the value witness box.

    Small treemap leaves may not have enough space for both strings; the value
    remains the required annotation target, so this helper progressively falls
    back to compact placements while preserving a projected value bbox.
    """

    x0, y0, x1, y1 = rect
    width = max(1.0, float(x1 - x0))
    height = max(1.0, float(y1 - y0))
    label_text = _truncate(str(label), max_chars=max(4, int(width // 9)))
    value_text = str(int(value))
    if height >= 54 and width >= 86:
        label_xy = (x0 + 8, y0 + 11)
        draw_text_traced(
            draw,
            label_xy,
            label_text,
            font=label_font,
            fill=muted_text,
            anchor="la",
            stroke_fill=stroke,
            stroke_width=max(0, int(stroke_width)),
            role="readout",
            required=False,
        )
        value_xy = (x0 + 8, y0 + 33)
        draw_text_traced(
            draw,
            value_xy,
            value_text,
            font=value_font,
            fill=text_color,
            anchor="la",
            stroke_fill=stroke,
            stroke_width=max(0, int(stroke_width)),
            role="readout",
            required=False,
        )
        return _text_bbox(draw, value_xy, value_text, font=value_font, anchor="la", stroke_width=stroke_width)
    if height >= 28 and width >= 118:
        label_xy = (x0 + 7, (y0 + y1) / 2.0)
        value_xy = (x1 - 7, (y0 + y1) / 2.0)
        draw_text_traced(
            draw,
            label_xy,
            label_text,
            font=label_font,
            fill=muted_text,
            anchor="lm",
            stroke_fill=stroke,
            stroke_width=max(0, int(stroke_width)),
            role="readout",
            required=False,
        )
        draw_text_traced(
            draw,
            value_xy,
            value_text,
            font=value_font,
            fill=text_color,
            anchor="rm",
            stroke_fill=stroke,
            stroke_width=max(0, int(stroke_width)),
            role="readout",
            required=False,
        )
        return _text_bbox(draw, value_xy, value_text, font=value_font, anchor="rm", stroke_width=stroke_width)
    value_xy = ((x0 + x1) / 2.0, (y0 + y1) / 2.0)
    draw_text_traced(
        draw,
        value_xy,
        value_text,
        font=value_font,
        fill=text_color,
        anchor="mm",
        stroke_fill=stroke,
        stroke_width=max(0, int(stroke_width)),
        role="readout",
        required=False,
    )
    return _text_bbox(draw, value_xy, value_text, font=value_font, anchor="mm", stroke_width=stroke_width)


def render_treemap_scene(
    background: Image.Image,
    *,
    dataset: TreemapDataset,
    params: Mapping[str, object],
    instance_seed: int,
    render_params: TreemapRenderParams | None = None,
) -> RenderedTreemap:
    """Render one treemap dataset and record all projected leaf-cell/value geometry."""

    if render_params is None:
        render_params = resolve_treemap_render_params(params, instance_seed=int(instance_seed))
    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    title_font = _font(render_params.title_font_size_px, bold=False)
    parent_font = _font(render_params.parent_font_size_px, bold=False)
    leaf_font = _font(render_params.leaf_font_size_px)
    value_font = _font(render_params.value_font_size_px, bold=False)
    note_font = _font(render_params.note_font_size_px)
    chart_bbox = (
        float(render_params.plot_margin_left_px),
        float(render_params.plot_margin_top_px),
        float(render_params.canvas_width - render_params.plot_margin_right_px),
        float(render_params.canvas_height - render_params.plot_margin_bottom_px),
    )
    draw.rounded_rectangle(
        [chart_bbox[0] - 16, chart_bbox[1] - 54, chart_bbox[2] + 16, chart_bbox[3] + 16],
        radius=6,
        fill=render_params.panel_fill_rgb,
        outline=lighten(render_params.panel_border_rgb, 0.35),
        width=2,
    )
    draw_text_traced(
        draw,
        (render_params.canvas_width / 2.0, 32),
        str(dataset.title),
        font=title_font,
        fill=render_params.text_color_rgb,
        anchor="mm",
        stroke_fill=render_params.text_stroke_rgb,
        stroke_width=0,
        role="readout",
        required=False,
    )
    draw_text_traced(
        draw,
        (chart_bbox[0], chart_bbox[1] - 22),
        f"Parent: {dataset.parent_axis}    Child: {dataset.leaf_axis}",
        font=note_font,
        fill=render_params.muted_text_rgb,
        anchor="la",
        role="readout",
        required=False,
    )
    leaves_by_id = {str(leaf.leaf_id): leaf for leaf in dataset.leaves}
    parent_rects = _slice_rects(
        [(str(parent.parent_id), int(parent.value)) for parent in dataset.parents],
        chart_bbox,
        horizontal=True,
    )
    entities: list[dict[str, object]] = []
    parent_traces: list[dict[str, object]] = []
    leaf_traces: list[dict[str, object]] = []
    annotation_bbox_by_leaf_id: dict[str, BBox] = {}
    for parent in dataset.parents:
        px0, py0, px1, py1 = parent_rects[str(parent.parent_id)]
        parent_rect = (px0 + 3, py0 + 3, px1 - 3, py1 - 3)
        draw.rectangle(parent_rect, fill=lighten(parent.color_rgb, 0.18), outline=render_params.panel_border_rgb, width=2)
        header_h = 30
        header_rect = (parent_rect[0], parent_rect[1], parent_rect[2], min(parent_rect[3], parent_rect[1] + header_h))
        header_fill = lighten_for_dense_text(parent.color_rgb, 0.42)
        header_text_rgb, header_stroke_rgb = _text_on_color(header_fill)
        draw.rectangle(header_rect, fill=header_fill, outline=render_params.panel_border_rgb, width=1)
        parent_text = _truncate(str(parent.label), max_chars=max(5, int((parent_rect[2] - parent_rect[0]) // 9)))
        draw_text_traced(
            draw,
            (header_rect[0] + 7, (header_rect[1] + header_rect[3]) / 2.0),
            parent_text,
            font=parent_font,
            fill=header_text_rgb,
            anchor="lm",
            stroke_fill=header_stroke_rgb,
            stroke_width=dense_stroke_width(),
            role="readout",
            required=False,
        )
        leaf_area = (parent_rect[0], header_rect[3], parent_rect[2], parent_rect[3])
        leaf_rects = _slice_rects(
            [(str(leaf_id), int(leaves_by_id[str(leaf_id)].value)) for leaf_id in parent.leaf_ids],
            leaf_area,
            horizontal=False,
        )
        parent_bbox = _bbox(parent_rect)
        entities.append(
            {
                "entity_id": str(parent.parent_id),
                "entity_type": "treemap_parent",
                "label": str(parent.label),
                "bbox_px": list(parent_bbox),
                "value": int(parent.value),
            }
        )
        parent_traces.append(
            {
                "parent_id": str(parent.parent_id),
                "label": str(parent.label),
                "value": int(parent.value),
                "leaf_ids": [str(leaf_id) for leaf_id in parent.leaf_ids],
                "bbox_px": list(parent_bbox),
            }
        )
        for leaf_id in parent.leaf_ids:
            leaf = leaves_by_id[str(leaf_id)]
            lx0, ly0, lx1, ly1 = leaf_rects[str(leaf_id)]
            leaf_rect = (lx0 + 1.5, ly0 + 1.5, lx1 - 1.5, ly1 - 1.5)
            leaf_fill = lighten_for_dense_text(leaf.color_rgb, 0.38)
            draw.rectangle(leaf_rect, fill=leaf_fill, outline=render_params.separator_rgb, width=2)
            leaf_text_rgb, leaf_stroke_rgb = _text_on_color(leaf_fill)
            value_bbox = _draw_label_value(
                draw,
                leaf_rect,
                label=str(leaf.label),
                value=int(leaf.value),
                text_color=leaf_text_rgb,
                muted_text=leaf_text_rgb,
                stroke=leaf_stroke_rgb,
                label_font=leaf_font,
                value_font=value_font,
                stroke_width=dense_stroke_width(),
            )
            leaf_bbox = _bbox(leaf_rect)
            annotation_bbox_by_leaf_id[str(leaf.leaf_id)] = list(leaf_bbox)
            entities.append(
                {
                    "entity_id": str(leaf.leaf_id),
                    "entity_type": "treemap_leaf",
                    "label": str(leaf.label),
                    "parent_id": str(leaf.parent_id),
                    "parent_label": str(leaf.parent_label),
                    "bbox_px": list(leaf_bbox),
                    "value_bbox_px": list(value_bbox),
                    "value": int(leaf.value),
                }
            )
            leaf_traces.append(
                {
                    "leaf_id": str(leaf.leaf_id),
                    "parent_id": str(leaf.parent_id),
                    "parent_label": str(leaf.parent_label),
                    "label": str(leaf.label),
                    "value": int(leaf.value),
                    "bbox_px": list(leaf_bbox),
                    "value_bbox_px": list(value_bbox),
                }
            )
    return RenderedTreemap(
        image=image,
        entities=tuple(dict(entity) for entity in entities),
        leaf_traces=tuple(dict(trace) for trace in leaf_traces),
        parent_traces=tuple(dict(trace) for trace in parent_traces),
        annotation_bbox_by_leaf_id=dict(annotation_bbox_by_leaf_id),
        chart_bbox_px=_bbox(chart_bbox),
        render_meta={
            "not_to_scale": False,
            "value_source": "printed_leaf_values",
            "layout": "slice_and_dice_treemap",
            "parent_count": int(len(dataset.parents)),
            "leaf_count_per_parent": int(len(dataset.parents[0].leaf_ids)) if dataset.parents else 0,
            "dense_text_style": dense_text_style_meta(role="treemap_leaf_values"),
        },
    )


def treemap_render_style_spec(render_params: TreemapRenderParams) -> dict[str, object]:
    return {
        "panel_fill_rgb": list(render_params.panel_fill_rgb),
        "panel_border_rgb": list(render_params.panel_border_rgb),
        "separator_rgb": list(render_params.separator_rgb),
        "label_stroke_width_px": int(render_params.label_stroke_width_px),
    }


def treemap_annotation_region(rendered: RenderedTreemap, leaf_ids: Sequence[str]) -> BBox:
    boxes = [
        list(rendered.annotation_bbox_by_leaf_id[str(leaf_id)])
        for leaf_id in leaf_ids
        if str(leaf_id) in rendered.annotation_bbox_by_leaf_id
    ]
    if not boxes:
        raise ValueError("no treemap annotation boxes to union")
    return [round(float(value), 3) for value in _bbox_union(boxes)]


__all__ = [
    "apply_treemap_information_style",
    "render_treemap_scene",
    "resolve_treemap_render_params",
    "treemap_annotation_region",
    "treemap_render_style_spec",
]
