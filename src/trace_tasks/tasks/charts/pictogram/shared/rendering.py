"""Rendering primitives for pictogram and waffle charts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from PIL import Image, ImageDraw

from trace_tasks.tasks.charts.shared.dense_text import dense_fit_bold, dense_stroke_width, dense_text_style_meta
from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from .....core.visual.noise import apply_post_image_noise
from ....shared.drawing import draw_rounded_rect
from ....shared.font_assets import font_asset_version
from ....shared.render_variation import apply_layout_jitter_to_margins
from ....shared.text_legibility import draw_text_traced
from ....shared.text_rendering import draw_text_centered, fit_font_to_box, load_font, temporary_default_font_family

from .defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    RENDER_DEFAULTS,
    render_int,
    render_rgb,
    sample_chart_font_family,
)
from .state import PictogramDataset, PictogramRenderParams, RenderedPictogramScene, RGB, BBox


@dataclass(frozen=True)
class PictogramRenderResult:
    image: Image.Image
    rendered_scene: RenderedPictogramScene
    background_meta: dict[str, Any]
    post_noise_meta: dict[str, Any]
    chart_font_family: str


def bbox(values: Sequence[float]) -> BBox:
    return [round(float(value), 3) for value in values]


def resolve_render_params(params: Mapping[str, object], *, instance_seed: int) -> PictogramRenderParams:
    """Resolve canvas, row, text, and palette parameters before drawing rows."""

    outer = render_int(params, "outer_margin_px", 44)
    title_band = render_int(params, "title_band_height_px", 70)
    legend_height = render_int(params, "legend_height_px", 58)
    bottom = render_int(params, "bottom_margin_px", 48)
    left, right, top, bottom, jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(outer),
        right_px=int(outer),
        top_px=int(outer),
        bottom_px=int(bottom),
        params=params,
        defaults=RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace="charts.pictogram.layout",
    )
    return PictogramRenderParams(
        canvas_width=render_int(params, "canvas_width", 920),
        canvas_height=render_int(params, "canvas_height", 900),
        outer_margin_px=int(left),
        title_band_height_px=int(title_band),
        legend_height_px=int(legend_height),
        row_gap_px=render_int(params, "row_gap_px", 8),
        label_width_px=render_int(params, "label_width_px", 178),
        mark_gap_px=render_int(params, "mark_gap_px", 6),
        mark_columns_max=render_int(params, "mark_columns_max", 10),
        mark_size_max_px=render_int(params, "mark_size_max_px", 34),
        row_corner_radius_px=render_int(params, "row_corner_radius_px", 8),
        panel_outline_width_px=render_int(params, "panel_outline_width_px", 2),
        title_font_size_px=render_int(params, "title_font_size_px", 28),
        label_font_size_px=render_int(params, "label_font_size_px", 22),
        legend_font_size_px=render_int(params, "legend_font_size_px", 20),
        value_font_size_px=render_int(params, "value_font_size_px", 15),
        text_rgb=render_rgb(params, "text_rgb", (36, 42, 52), instance_seed=int(instance_seed)),
        muted_text_rgb=render_rgb(params, "muted_text_rgb", (86, 95, 108), instance_seed=int(instance_seed)),
        text_stroke_rgb=render_rgb(params, "text_stroke_rgb", (255, 255, 255), instance_seed=int(instance_seed)),
        panel_fill_rgb=render_rgb(params, "panel_fill_rgb", (255, 255, 255), instance_seed=int(instance_seed)),
        panel_outline_rgb=render_rgb(params, "panel_outline_rgb", (194, 202, 214), instance_seed=int(instance_seed)),
        legend_fill_rgb=render_rgb(params, "legend_fill_rgb", (248, 250, 252), instance_seed=int(instance_seed)),
        mark_outline_rgb=render_rgb(params, "mark_outline_rgb", (50, 58, 68), instance_seed=int(instance_seed)),
        layout_jitter_meta=dict(jitter_meta),
    )


def star_points(cx: float, cy: float, radius: float) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    inner = float(radius) * 0.46
    for index in range(10):
        angle = -math.pi / 2.0 + (float(index) * math.pi / 5.0)
        r = float(radius) if index % 2 == 0 else float(inner)
        points.append((float(cx + (math.cos(angle) * r)), float(cy + (math.sin(angle) * r))))
    return points


def draw_glyph(
    draw: ImageDraw.ImageDraw,
    *,
    glyph: str,
    bbox_xyxy: Sequence[float],
    fill: RGB,
    outline: RGB,
    width: int,
) -> None:
    """Draw one supported pictogram glyph inside the supplied mark box."""

    x0, y0, x1, y1 = [float(value) for value in bbox_xyxy]
    w = float(x1 - x0)
    h = float(y1 - y0)
    cx = float((x0 + x1) / 2.0)
    cy = float((y0 + y1) / 2.0)
    fill_rgb = tuple(int(value) for value in fill)
    outline_rgb = tuple(int(value) for value in outline)
    line_w = max(1, int(width))

    if str(glyph) == "circle" or str(glyph) == "coin":
        draw.ellipse((x0, y0, x1, y1), fill=fill_rgb, outline=outline_rgb, width=line_w)
        if str(glyph) == "coin":
            draw.arc(
                (x0 + w * 0.22, y0 + h * 0.18, x1 - w * 0.22, y1 - h * 0.18),
                70,
                290,
                fill=outline_rgb,
                width=1,
            )
    elif str(glyph) == "square":
        draw.rounded_rectangle(
            (x0, y0, x1, y1),
            radius=max(2, int(min(w, h) * 0.12)),
            fill=fill_rgb,
            outline=outline_rgb,
            width=line_w,
        )
    elif str(glyph) == "star":
        draw.polygon(star_points(cx, cy, min(w, h) * 0.48), fill=fill_rgb, outline=outline_rgb)
    elif str(glyph) == "person":
        draw.ellipse(
            (cx - w * 0.18, y0 + h * 0.03, cx + w * 0.18, y0 + h * 0.37),
            fill=fill_rgb,
            outline=outline_rgb,
            width=line_w,
        )
        draw.rounded_rectangle(
            (cx - w * 0.25, y0 + h * 0.40, cx + w * 0.25, y1 - h * 0.05),
            radius=max(2, int(w * 0.10)),
            fill=fill_rgb,
            outline=outline_rgb,
            width=line_w,
        )
    elif str(glyph) == "car":
        draw.rounded_rectangle(
            (x0 + w * 0.08, y0 + h * 0.34, x1 - w * 0.08, y1 - h * 0.22),
            radius=max(2, int(w * 0.08)),
            fill=fill_rgb,
            outline=outline_rgb,
            width=line_w,
        )
        draw.polygon(
            [
                (x0 + w * 0.28, y0 + h * 0.34),
                (x0 + w * 0.43, y0 + h * 0.16),
                (x0 + w * 0.68, y0 + h * 0.16),
                (x0 + w * 0.82, y0 + h * 0.34),
            ],
            fill=fill_rgb,
            outline=outline_rgb,
        )
        draw.ellipse((x0 + w * 0.20, y1 - h * 0.25, x0 + w * 0.36, y1 - h * 0.08), fill=outline_rgb)
        draw.ellipse((x1 - w * 0.36, y1 - h * 0.25, x1 - w * 0.20, y1 - h * 0.08), fill=outline_rgb)
    elif str(glyph) == "leaf":
        draw.ellipse(
            (x0 + w * 0.08, y0 + h * 0.08, x1 - w * 0.08, y1 - h * 0.08),
            fill=fill_rgb,
            outline=outline_rgb,
            width=line_w,
        )
        draw.line((x0 + w * 0.28, y1 - h * 0.22, x1 - w * 0.22, y0 + h * 0.24), fill=outline_rgb, width=max(1, line_w))
    elif str(glyph) == "book":
        draw.rounded_rectangle(
            (x0 + w * 0.10, y0 + h * 0.06, x1 - w * 0.10, y1 - h * 0.06),
            radius=max(2, int(w * 0.06)),
            fill=fill_rgb,
            outline=outline_rgb,
            width=line_w,
        )
        draw.line((cx, y0 + h * 0.10, cx, y1 - h * 0.10), fill=outline_rgb, width=max(1, line_w))
    else:
        draw.ellipse((x0, y0, x1, y1), fill=fill_rgb, outline=outline_rgb, width=line_w)


def render_chart(
    *,
    background: Image.Image,
    dataset: PictogramDataset,
    params: Mapping[str, object],
    instance_seed: int,
    render_params: PictogramRenderParams | None = None,
) -> RenderedPictogramScene:
    """Render all category rows and record row/mark projections by category id."""

    render_params = render_params or resolve_render_params(params, instance_seed=int(instance_seed))
    image = background.copy()
    draw = ImageDraw.Draw(image)
    width = int(render_params.canvas_width)
    height = int(render_params.canvas_height)
    margin = int(render_params.outer_margin_px)
    right = width - margin

    title_font = load_font(int(render_params.title_font_size_px), bold=False)
    legend_font = load_font(int(render_params.legend_font_size_px), bold=False)
    draw_text_centered(
        draw,
        text=str(dataset.title),
        center=(float(width) / 2.0, float(margin + render_params.title_band_height_px * 0.42)),
        font=title_font,
        fill=render_params.text_rgb,
        stroke_fill=render_params.text_stroke_rgb,
        stroke_width=dense_stroke_width(),
    )

    legend_x0 = float(margin)
    legend_y0 = float(margin + render_params.title_band_height_px)
    legend_x1 = float(right)
    legend_y1 = float(legend_y0 + render_params.legend_height_px)
    draw_rounded_rect(
        draw,
        (legend_x0, legend_y0, legend_x1, legend_y1),
        radius=10,
        fill=render_params.legend_fill_rgb,
        outline=render_params.panel_outline_rgb,
        width=1,
    )
    sample_size = min(30.0, float(render_params.legend_height_px) * 0.48)
    sample_box = (
        legend_x0 + 24.0,
        legend_y0 + (legend_y1 - legend_y0 - sample_size) / 2.0,
        legend_x0 + 24.0 + sample_size,
        legend_y0 + (legend_y1 - legend_y0 + sample_size) / 2.0,
    )
    sample_color = dataset.categories[0].color_rgb if dataset.categories else (37, 99, 235)
    glyph_for_legend = "square" if str(dataset.scene_variant) == "waffle_grid_blocks" else str(dataset.glyph_name)
    draw_glyph(
        draw,
        glyph=glyph_for_legend,
        bbox_xyxy=sample_box,
        fill=sample_color,
        outline=render_params.mark_outline_rgb,
        width=2,
    )
    draw_text_traced(
        draw,
        (sample_box[2] + 16.0, legend_y0 + (legend_y1 - legend_y0) * 0.32),
        f"1 mark = {dataset.unit_scale} units",
        font=legend_font,
        fill=render_params.text_rgb,
        role="readout",
        required=False,
    )

    top = float(legend_y1 + 20.0)
    bottom = float(height - render_params.outer_margin_px)
    category_count = len(dataset.categories)
    available_h = max(1.0, bottom - top - (float(category_count - 1) * float(render_params.row_gap_px)))
    row_h = float(available_h / float(max(1, category_count)))
    plot_x0 = float(margin)
    plot_x1 = float(right)
    mark_area_x0 = float(plot_x0 + render_params.label_width_px)
    mark_area_x1 = float(plot_x1 - 18.0)
    mark_area_w = max(1.0, mark_area_x1 - mark_area_x0)
    max_marks = max(category.mark_count for category in dataset.categories)
    max_cols = max(1, int(render_params.mark_columns_max))
    planned_rows = max(1, int(math.ceil(float(max_marks) / float(max_cols))))
    mark_gap = float(render_params.mark_gap_px)
    size_from_height = (float(row_h) - 14.0 - (float(planned_rows - 1) * mark_gap)) / float(planned_rows)
    size_from_width = (float(mark_area_w) - (float(max_cols - 1) * mark_gap)) / float(max_cols)
    mark_size = max(10.0, min(float(render_params.mark_size_max_px), float(size_from_height), float(size_from_width)))
    mark_cols = max(1, min(max_cols, int((mark_area_w + mark_gap) // (mark_size + mark_gap))))

    category_bboxes: dict[str, BBox] = {}
    mark_bboxes: dict[str, list[BBox]] = {}
    entities: list[dict[str, Any]] = []
    for row_index, category in enumerate(dataset.categories):
        row_y0 = float(top + (float(row_index) * (row_h + float(render_params.row_gap_px))))
        row_y1 = float(row_y0 + row_h)
        row_fill = render_params.panel_fill_rgb if row_index % 2 == 0 else render_params.legend_fill_rgb
        row_box = bbox([plot_x0, row_y0, plot_x1, row_y1])
        draw_rounded_rect(
            draw,
            (plot_x0, row_y0, plot_x1, row_y1),
            radius=int(render_params.row_corner_radius_px),
            fill=row_fill,
            outline=render_params.panel_outline_rgb,
            width=1,
        )
        label_font = fit_font_to_box(
            draw,
            text=str(category.label),
            max_width=float(render_params.label_width_px) - 20.0,
            max_height=float(row_h) - 10.0,
            bold=dense_fit_bold(),
            min_size_px=12,
            max_size_px=int(render_params.label_font_size_px),
            fill_ratio=0.92,
        )
        draw_text_centered(
            draw,
            text=str(category.label),
            center=(float(plot_x0 + render_params.label_width_px * 0.48), float((row_y0 + row_y1) / 2.0)),
            font=label_font,
            fill=render_params.text_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=dense_stroke_width(),
        )
        category_bboxes[category.category_id] = list(row_box)
        marks_for_category: list[BBox] = []
        actual_rows = max(1, int(math.ceil(float(category.mark_count) / float(mark_cols))))
        total_mark_h = (float(actual_rows) * mark_size) + (float(actual_rows - 1) * mark_gap)
        start_y = float(row_y0 + max(5.0, (row_h - total_mark_h) / 2.0))
        for mark_index in range(int(category.mark_count)):
            r = int(mark_index // mark_cols)
            c = int(mark_index % mark_cols)
            x0 = float(mark_area_x0 + float(c) * (mark_size + mark_gap))
            y0 = float(start_y + float(r) * (mark_size + mark_gap))
            mark_box = bbox([x0, y0, x0 + mark_size, y0 + mark_size])
            glyph = "square" if str(dataset.scene_variant) == "waffle_grid_blocks" else str(dataset.glyph_name)
            draw_glyph(
                draw,
                glyph=str(glyph),
                bbox_xyxy=mark_box,
                fill=category.color_rgb,
                outline=render_params.mark_outline_rgb,
                width=max(1, int(render_params.panel_outline_width_px)),
            )
            marks_for_category.append(mark_box)
        mark_bboxes[category.category_id] = marks_for_category
        entities.append(
            {
                "entity_id": str(category.category_id),
                "entity_type": "pictogram_category_row",
                "bbox_xyxy": list(row_box),
                "attrs": {
                    "label": str(category.label),
                    "mark_count": int(category.mark_count),
                    "unit_scale": int(dataset.unit_scale),
                    "total": int(category.total),
                    "color_rgb": [int(channel) for channel in category.color_rgb],
                },
            }
        )
    render_meta = {
        "scene_variant": str(dataset.scene_variant),
        "glyph_name": str(dataset.glyph_name),
        "mark_size_px": round(float(mark_size), 3),
        "mark_columns": int(mark_cols),
        "row_height_px": round(float(row_h), 3),
        "layout_jitter": dict(render_params.layout_jitter_meta),
        "dense_text_style": dense_text_style_meta(role="pictogram_category_labels"),
    }
    return RenderedPictogramScene(
        image=image,
        entities=tuple(entities),
        plot_bbox_px=bbox([plot_x0, top, plot_x1, bottom]),
        legend_bbox_px=bbox([legend_x0, legend_y0, legend_x1, legend_y1]),
        category_bboxes_px=category_bboxes,
        mark_bboxes_px=mark_bboxes,
        render_meta=render_meta,
    )


def render_pictogram_dataset(
    *,
    dataset: PictogramDataset,
    params: Mapping[str, object],
    instance_seed: int,
) -> PictogramRenderResult:
    """Render task-neutral pictogram geometry with shared chart styling metadata."""

    resolved_params = resolve_render_params(params, instance_seed=int(instance_seed))
    protected_colors = [tuple(int(channel) for channel in category.color_rgb) for category in dataset.categories]
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="pictogram",
        render_params=resolved_params,
        protected_colors=protected_colors,
    )
    chart_font_family = sample_chart_font_family(int(instance_seed), params)
    with temporary_default_font_family(str(chart_font_family)):
        rendered_scene = render_chart(
            background=background,
            dataset=dataset,
            params=params,
            instance_seed=int(instance_seed),
            render_params=render_params,
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return PictogramRenderResult(
        image=image,
        rendered_scene=RenderedPictogramScene(
            image=image,
            entities=tuple(rendered_scene.entities),
            plot_bbox_px=list(rendered_scene.plot_bbox_px),
            legend_bbox_px=list(rendered_scene.legend_bbox_px),
            category_bboxes_px=dict(rendered_scene.category_bboxes_px),
            mark_bboxes_px=dict(rendered_scene.mark_bboxes_px),
            render_meta={
                **dict(rendered_scene.render_meta),
                "background_style": {**dict(background_meta), "information_scene_style": dict(information_style_meta)},
                "information_scene_style": dict(information_style_meta),
                "post_image_noise": dict(post_noise_meta),
            },
        ),
        background_meta={**dict(background_meta), "information_scene_style": dict(information_style_meta)},
        post_noise_meta=dict(post_noise_meta),
        chart_font_family=str(chart_font_family),
    )


def font_assets_payload(*, chart_font_family: str) -> dict[str, str]:
    return {
        "font_asset_version": font_asset_version(),
        "chart_font_family": str(chart_font_family),
    }
