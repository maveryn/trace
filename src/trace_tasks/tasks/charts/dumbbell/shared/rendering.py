"""Rendering primitives for dumbbell chart scenes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PIL import Image, ImageDraw

from trace_tasks.tasks.charts.shared.cartesian.frame import plot_bbox_from_margins
from trace_tasks.tasks.charts.shared.cartesian.geometry import project_linear, round_bbox
from trace_tasks.tasks.shared.bbox_projection import bbox_union_raw
from trace_tasks.tasks.shared.font_assets import font_asset_version
from trace_tasks.tasks.shared.render_variation import apply_layout_jitter_to_margins
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font
from trace_tasks.tasks.charts.dumbbell.shared.defaults import (
    SCENE_NAMESPACE,
    render_int,
    render_rgb,
    sample_chart_font_family,
)
from trace_tasks.tasks.charts.dumbbell.shared.state import DumbbellDataset, DumbbellRenderParams, RGB, RenderedDumbbell


def _bbox(values: Sequence[float]) -> list[float]:
    return round_bbox(values)


def resolve_dumbbell_render_params(params: Mapping[str, Any], *, instance_seed: int) -> DumbbellRenderParams:
    """Resolve all rendering parameters for one dumbbell chart."""

    render_params = {**dict(params), "_render_style_seed": int(instance_seed)}
    margin_left = render_int(render_params, "plot_margin_left_px", 224)
    margin_right = render_int(render_params, "plot_margin_right_px", 88)
    margin_top = render_int(render_params, "plot_margin_top_px", 138)
    margin_bottom = render_int(render_params, "plot_margin_bottom_px", 112)
    margin_left, margin_right, margin_top, margin_bottom, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(margin_left),
        right_px=int(margin_right),
        top_px=int(margin_top),
        bottom_px=int(margin_bottom),
        params=render_params,
        defaults={},
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.layout",
    )
    return DumbbellRenderParams(
        canvas_width=render_int(render_params, "canvas_width", 1280),
        canvas_height=render_int(render_params, "canvas_height", 900),
        plot_margin_left_px=int(margin_left),
        plot_margin_right_px=int(margin_right),
        plot_margin_top_px=int(margin_top),
        plot_margin_bottom_px=int(margin_bottom),
        axis_line_width_px=render_int(render_params, "axis_line_width_px", 2),
        grid_line_width_px=render_int(render_params, "grid_line_width_px", 1),
        row_line_width_px=render_int(render_params, "row_line_width_px", 1),
        connector_width_px=render_int(render_params, "connector_width_px", 4),
        point_radius_px=render_int(render_params, "point_radius_px", 8),
        point_outline_width_px=render_int(render_params, "point_outline_width_px", 2),
        tick_length_px=render_int(render_params, "tick_length_px", 8),
        title_font_size_px=render_int(render_params, "title_font_size_px", 30),
        subtitle_font_size_px=render_int(render_params, "subtitle_font_size_px", 18),
        label_font_size_px=render_int(render_params, "label_font_size_px", 20),
        tick_font_size_px=render_int(render_params, "tick_font_size_px", 17),
        legend_font_size_px=render_int(render_params, "legend_font_size_px", 20),
        panel_fill_rgb=render_rgb(render_params, "panel_fill_rgb", (255, 255, 255)),
        panel_border_rgb=render_rgb(render_params, "panel_border_rgb", (196, 203, 214)),
        plot_fill_rgb=render_rgb(render_params, "plot_fill_rgb", (255, 255, 255)),
        axis_color_rgb=render_rgb(render_params, "axis_color_rgb", (62, 68, 78)),
        grid_color_rgb=render_rgb(render_params, "grid_color_rgb", (224, 228, 235)),
        row_line_rgb=render_rgb(render_params, "row_line_rgb", (236, 239, 244)),
        connector_rgb=render_rgb(render_params, "connector_rgb", (154, 162, 174)),
        text_color_rgb=render_rgb(render_params, "text_color_rgb", (34, 40, 50)),
        muted_text_rgb=render_rgb(render_params, "muted_text_rgb", (83, 94, 110)),
        text_stroke_rgb=render_rgb(render_params, "text_stroke_rgb", (255, 255, 255)),
        series_a_rgb=render_rgb(render_params, "series_a_rgb", (38, 101, 176)),
        series_b_rgb=render_rgb(render_params, "series_b_rgb", (213, 92, 72)),
        font_family=sample_chart_font_family(render_params, instance_seed=int(instance_seed)),
        layout_jitter_meta=dict(layout_jitter_meta),
    )


def _text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: Any,
    *,
    stroke_width: int = 0,
) -> list[float]:
    box = draw.textbbox(tuple(float(value) for value in xy), str(text), font=font, stroke_width=max(0, int(stroke_width)))
    return _bbox([box[0], box[1], box[2], box[3]])


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: Any,
    *,
    fill: RGB,
    stroke_fill: RGB,
    stroke_width: int = 0,
) -> list[float]:
    box = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width)))
    width = float(box[2] - box[0])
    height = float(box[3] - box[1])
    x = float(xy[0]) - (width / 2.0)
    y = float(xy[1]) - (height / 2.0)
    draw_text_traced(
        draw,
        (x, y),
        str(text),
        font=font,
        fill=fill,
        stroke_fill=stroke_fill,
        stroke_width=max(0, int(stroke_width)),
        role="readout",
        required=False,
    )
    return _bbox([x, y, x + width, y + height])


def render_dumbbell_chart(
    background: Image.Image,
    *,
    dataset: DumbbellDataset,
    render_params: DumbbellRenderParams,
    instance_seed: int,
) -> RenderedDumbbell:
    """Render one fully bound dumbbell chart scene."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    left = float(render_params.plot_margin_left_px)
    right = float(width - int(render_params.plot_margin_right_px))
    top = float(render_params.plot_margin_top_px)
    bottom = float(height - int(render_params.plot_margin_bottom_px))
    plot_bbox = plot_bbox_from_margins(
        canvas_width=float(width),
        canvas_height=float(height),
        margin_left_px=float(render_params.plot_margin_left_px),
        margin_right_px=float(render_params.plot_margin_right_px),
        margin_top_px=float(render_params.plot_margin_top_px),
        margin_bottom_px=float(render_params.plot_margin_bottom_px),
    )
    panel_margin = 34
    panel_bbox = [panel_margin, 36, width - panel_margin, height - 42]
    draw.rounded_rectangle(panel_bbox, radius=8, fill=render_params.panel_fill_rgb, outline=render_params.panel_border_rgb, width=1)
    draw.rectangle([left, top, right, bottom], fill=render_params.plot_fill_rgb)

    label_font = load_font(render_params.label_font_size_px, bold=True, font_family=render_params.font_family)
    tick_font = load_font(render_params.tick_font_size_px, bold=False, font_family=render_params.font_family)
    legend_font = load_font(render_params.legend_font_size_px, bold=True, font_family=render_params.font_family)

    def x_px(value: float) -> float:
        return project_linear(float(value), domain_min=0.0, domain_max=100.0, pixel_min=left, pixel_max=right)

    tick_values = [0, 20, 40, 60, 80, 100]
    for value in tick_values:
        x = x_px(float(value))
        draw.line([x, top, x, bottom], fill=render_params.grid_color_rgb, width=render_params.grid_line_width_px)
        draw.line([x, bottom, x, bottom + render_params.tick_length_px], fill=render_params.axis_color_rgb, width=render_params.axis_line_width_px)
        _draw_centered_text(
            draw,
            (x, bottom + 28),
            str(value),
            tick_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=1,
        )
    draw.line([left, bottom, right, bottom], fill=render_params.axis_color_rgb, width=render_params.axis_line_width_px)

    row_count = len(dataset.rows)
    row_gap = (bottom - top) / max(1, row_count - 1)
    row_label_bboxes: dict[str, list[float]] = {}
    point_bboxes: dict[str, list[float]] = {}
    row_pair_bboxes: dict[str, list[float]] = {}
    connector_bboxes: dict[str, list[float]] = {}
    entities: list[dict[str, Any]] = []
    radius = float(render_params.point_radius_px)

    for index, row in enumerate(dataset.rows):
        y = top + (float(index) * row_gap)
        draw.line([left, y, right, y], fill=render_params.row_line_rgb, width=render_params.row_line_width_px)
        label_xy = (panel_margin + 28, y - 10)
        label_bbox = _text_bbox(draw, label_xy, row.label, label_font, stroke_width=1)
        draw_text_traced(
            draw,
            label_xy,
            row.label,
            font=label_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=1,
            role="readout",
            required=False,
        )
        row_label_bboxes[row.row_id] = list(label_bbox)

        xa = x_px(float(row.value_a))
        xb = x_px(float(row.value_b))
        x0, x1 = sorted([xa, xb])
        connector_box = _bbox([x0, y - (render_params.connector_width_px / 2.0), x1, y + (render_params.connector_width_px / 2.0)])
        draw.line([xa, y, xb, y], fill=render_params.connector_rgb, width=render_params.connector_width_px)
        connector_bboxes[row.row_id] = connector_box

        point_a_bbox = _bbox([xa - radius, y - radius, xa + radius, y + radius])
        point_b_bbox = _bbox([xb - radius, y - radius, xb + radius, y + radius])
        draw.ellipse(point_a_bbox, fill=render_params.series_a_rgb, outline=(255, 255, 255), width=render_params.point_outline_width_px)
        draw.ellipse(point_b_bbox, fill=render_params.series_b_rgb, outline=(255, 255, 255), width=render_params.point_outline_width_px)
        point_bboxes[f"{row.row_id}:series_a"] = list(point_a_bbox)
        point_bboxes[f"{row.row_id}:series_b"] = list(point_b_bbox)
        row_pair_bboxes[row.row_id] = bbox_union_raw([connector_box, point_a_bbox, point_b_bbox], padding=8)
        entities.append(
            {
                "entity_id": row.row_id,
                "entity_type": "dumbbell_row_pair",
                "label": row.label,
                "value_a": int(row.value_a),
                "value_b": int(row.value_b),
                "gap": int(row.gap),
                "bbox_px": list(row_pair_bboxes[row.row_id]),
                "point_a_bbox_px": list(point_a_bbox),
                "point_b_bbox_px": list(point_b_bbox),
            }
        )

    legend_x = right - 250
    legend_y = 58
    legend_bboxes: dict[str, list[float]] = {}
    for idx, (name, color) in enumerate(
        [(dataset.series_a_name, render_params.series_a_rgb), (dataset.series_b_name, render_params.series_b_rgb)]
    ):
        y = legend_y + (idx * 30)
        dot_bbox = [legend_x, y + 5, legend_x + 16, y + 21]
        draw.ellipse(dot_bbox, fill=color, outline=(255, 255, 255), width=2)
        text_xy = (legend_x + 26, y)
        text_bbox = _text_bbox(draw, text_xy, name, legend_font)
        draw_text_traced(draw, text_xy, name, font=legend_font, fill=render_params.text_color_rgb, role="readout", required=False)
        legend_bboxes[f"series_{idx}"] = bbox_union_raw([dot_bbox, text_bbox], padding=2)

    return RenderedDumbbell(
        image=image,
        entities=tuple(entities),
        plot_bbox_px=list(plot_bbox),
        row_label_bboxes_px=dict(row_label_bboxes),
        point_bboxes_px=dict(point_bboxes),
        row_pair_bboxes_px=dict(row_pair_bboxes),
        connector_bboxes_px=dict(connector_bboxes),
        legend_bboxes_px=dict(legend_bboxes),
    )


def render_metadata(render_params: DumbbellRenderParams, rendered: RenderedDumbbell, dataset: DumbbellDataset) -> dict[str, Any]:
    """Return render metadata for the trace payload."""

    return {
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "scene_variant": str(dataset.scene_variant),
        "row_count": int(len(dataset.rows)),
        "plot_bbox_px": list(rendered.plot_bbox_px),
        "legend_bboxes_px": dict(rendered.legend_bboxes_px),
        "layout_jitter": dict(render_params.layout_jitter_meta or {}),
        "font_assets": {
            "asset_version": str(font_asset_version()),
            "chart_font_family": str(render_params.font_family),
        },
        "chart_font_family": str(render_params.font_family),
        "font_asset_version": str(font_asset_version()),
    }


__all__ = [
    "render_dumbbell_chart",
    "render_metadata",
    "resolve_dumbbell_render_params",
]
