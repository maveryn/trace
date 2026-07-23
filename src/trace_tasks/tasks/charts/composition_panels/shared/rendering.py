"""Rendering primitives for composition-panel charts."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.tasks.charts.shared.information_style import (
    make_chart_information_background,
    resolve_chart_information_style,
)
from trace_tasks.tasks.charts.shared.dense_text import (
    dense_fit_bold,
    dense_stroke_width,
    dense_text_fill_for_surface,
    dense_text_style_meta,
)
from trace_tasks.tasks.charts.shared.composition.values import count_from_percent_share
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.shared.panel.grid_layout import layout_panel_grid
from trace_tasks.tasks.charts.composition_panels.shared.defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    render_int,
    render_rgb,
)
from trace_tasks.tasks.charts.composition_panels.shared.sampling import counts_for_panel
from trace_tasks.tasks.charts.composition_panels.shared.state import (
    SEGMENT_COLORS,
    SCENE_NAMESPACE,
    RenderedCompositionPanels,
    CompositionPanelsDataset,
)
from trace_tasks.tasks.shared.bbox_projection import bbox_union
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.render_variation import apply_layout_jitter_to_margins
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font

from .defaults import RENDER_DEFAULTS


def _text_bbox_at_origin(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: Any,
    *,
    stroke_width: int = 0,
) -> tuple[float, float, float, float]:
    try:
        bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width)))
        return float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        pad = float(max(0, int(stroke_width)))
        return float(-pad), float(-pad), float(width + pad), float(height + pad)


def bbox_center_point(bbox: Sequence[float]) -> list[float]:
    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
    ]


def _draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: Any,
    fill: tuple[int, int, int],
    *,
    stroke_width: int = 0,
    stroke_fill: tuple[int, int, int] = (255, 255, 255),
) -> list[float]:
    raw = _text_bbox_at_origin(draw, str(text), font, stroke_width=max(0, int(stroke_width)))
    width = float(raw[2] - raw[0])
    height = float(raw[3] - raw[1])
    left = float(xy[0]) - (width / 2.0) - float(raw[0])
    top = float(xy[1]) - (height / 2.0) - float(raw[1])
    draw_text_traced(
        draw,
        (float(left), float(top)),
        str(text),
        font=font,
        fill=fill,
        stroke_width=max(0, int(stroke_width)),
        stroke_fill=stroke_fill,
        role="readout",
        required=False,
    )
    return [
        float(left + raw[0]),
        float(top + raw[1]),
        float(left + raw[2]),
        float(top + raw[3]),
    ]


def _segment_bbox(center: tuple[float, float], radius: float, start_angle: float, end_angle: float) -> list[float]:
    mid = math.radians((float(start_angle) + float(end_angle)) / 2.0)
    cx = float(center[0]) + (math.cos(mid) * float(radius) * 0.58)
    cy = float(center[1]) + (math.sin(mid) * float(radius) * 0.58)
    half = max(18.0, float(radius) * 0.22)
    return [float(cx - half), float(cy - half), float(cx + half), float(cy + half)]


def _draw_chart(
    *,
    base_image: Image.Image,
    dataset: CompositionPanelsDataset,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> RenderedCompositionPanels:
    """Draw one composition-panel chart scene from a task-neutral dataset.

    Key invariant: rendering projects every visible segment percentage and total
    readout without deciding which of those witnesses are task annotations.
    """

    image = base_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    text_color = render_rgb(params, "text_color_rgb", [36, 40, 48], instance_seed=int(instance_seed))
    grid_color = render_rgb(params, "grid_color_rgb", [214, 219, 228], instance_seed=int(instance_seed))
    panel_fill = render_rgb(params, "plot_fill_rgb", [255, 255, 255], instance_seed=int(instance_seed))
    mark_outline_width_px = render_int(params, "mark_outline_width_px", 2, instance_seed=int(instance_seed))
    title_font = load_font(int(params.get("title_font_size_px", group_default(RENDER_DEFAULTS, "title_font_size_px", 24))), bold=False)
    subtitle_font = load_font(int(params.get("subtitle_font_size_px", group_default(RENDER_DEFAULTS, "subtitle_font_size_px", 17))), bold=False)
    slice_font = load_font(int(params.get("slice_font_size_px", group_default(RENDER_DEFAULTS, "slice_font_size_px", 17))), bold=dense_fit_bold())
    small_slice_font = load_font(int(params.get("small_slice_font_size_px", group_default(RENDER_DEFAULTS, "small_slice_font_size_px", 14))), bold=dense_fit_bold())
    legend_font = load_font(int(params.get("legend_font_size_px", group_default(RENDER_DEFAULTS, "legend_font_size_px", 18))), bold=False)

    margin_left = int(params.get("plot_margin_left_px", group_default(RENDER_DEFAULTS, "plot_margin_left_px", 54)))
    margin_right = int(params.get("plot_margin_right_px", group_default(RENDER_DEFAULTS, "plot_margin_right_px", 54)))
    margin_top = int(params.get("plot_margin_top_px", group_default(RENDER_DEFAULTS, "plot_margin_top_px", 38)))
    margin_bottom = int(params.get("plot_margin_bottom_px", group_default(RENDER_DEFAULTS, "plot_margin_bottom_px", 110)))
    margin_left, margin_right, margin_top, margin_bottom, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(margin_left),
        right_px=int(margin_right),
        top_px=int(margin_top),
        bottom_px=int(margin_bottom),
        params=params,
        defaults=RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.layout",
    )
    plot_bbox = (margin_left, margin_top, width - margin_right, height - margin_bottom)
    gap_x = 26
    gap_y = 28
    panel_boxes = layout_panel_grid(
        plot_bbox,
        panel_count=len(dataset.panels),
        gap_x=float(gap_x),
        gap_y=float(gap_y),
    )
    entities: list[dict[str, Any]] = []
    panel_traces: list[dict[str, Any]] = []
    annotation_bbox_by_key: dict[tuple[str, str], list[float]] = {}
    total_bbox_by_panel: dict[str, list[float]] = {}
    segment_color_by_label = {
        str(label): SEGMENT_COLORS[index % len(SEGMENT_COLORS)]
        for index, label in enumerate(dataset.segment_labels)
    }

    for panel, panel_bbox in zip(dataset.panels, panel_boxes):
        x0, y0, x1, y1 = (float(value) for value in panel_bbox)
        panel_width = float(x1 - x0)
        panel_height = float(y1 - y0)
        draw.rounded_rectangle((x0, y0, x1, y1), radius=8, fill=panel_fill, outline=grid_color, width=max(1, int(mark_outline_width_px)))
        _draw_centered(draw, ((x0 + x1) / 2.0, y0 + 23), str(panel.label), title_font, text_color)
        total_bbox_by_panel[str(panel.label)] = _draw_centered(
            draw,
            ((x0 + x1) / 2.0, y0 + 51),
            f"Total {int(panel.total)}",
            subtitle_font,
            text_color,
        )
        radius = max(42.0, min(panel_width * 0.29, (panel_height - 78.0) * 0.43))
        center = ((x0 + x1) / 2.0, y0 + 63.0 + radius)
        pie_box = (center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius)
        start_angle = -90.0
        slice_traces: list[dict[str, Any]] = []
        for segment in dataset.segment_labels:
            share = int(panel.shares_by_segment[str(segment)])
            end_angle = float(start_angle + (float(share) * 3.6))
            color = segment_color_by_label[str(segment)]
            draw.pieslice(
                pie_box,
                start=float(start_angle),
                end=float(end_angle),
                fill=color,
                outline=(255, 255, 255),
                width=max(1, int(mark_outline_width_px)),
            )
            if str(scene_variant) == "composition_donut_panels":
                inner = radius * 0.45
                draw.ellipse(
                    (center[0] - inner, center[1] - inner, center[0] + inner, center[1] + inner),
                    fill=panel_fill,
                    outline=panel_fill,
                )
            wedge_bbox = _segment_bbox(center, radius, start_angle, end_angle)
            mid = math.radians((start_angle + end_angle) / 2.0)
            if int(share) <= 7:
                label_radius = radius * (0.84 if str(scene_variant) == "composition_pie_panels" else 0.82)
                active_slice_font = small_slice_font
            else:
                label_radius = radius * (0.70 if str(scene_variant) == "composition_pie_panels" else 0.76)
                active_slice_font = slice_font
            lx = float(center[0] + (math.cos(mid) * label_radius))
            ly = float(center[1] + (math.sin(mid) * label_radius))
            text_bbox = _draw_centered(
                draw,
                (lx, ly),
                str(share),
                active_slice_font,
                dense_text_fill_for_surface(color),
                stroke_width=dense_stroke_width(),
                stroke_fill=panel_fill,
            )
            annotation_bbox_by_key[(str(panel.label), str(segment))] = list(text_bbox)
            count_value = count_from_percent_share(int(panel.total), int(share))
            slice_trace = {
                "panel_label": str(panel.label),
                "segment_label": str(segment),
                "share_percent": int(share),
                "total": int(panel.total),
                "count": int(count_value),
                "slice_bbox_px": list(text_bbox),
                "slice_wedge_bbox_px": list(wedge_bbox),
                "slice_center_px": [float(lx), float(ly)],
                "fill_rgb": list(color),
            }
            entities.append(
                {
                    "entity_id": f"{panel.label}:{segment}",
                    "kind": "composition_segment",
                    "attrs": dict(slice_trace),
                }
            )
            slice_traces.append(dict(slice_trace))
            start_angle = float(end_angle)
        panel_traces.append(
            {
                "panel_label": str(panel.label),
                "total": int(panel.total),
                "panel_bbox_px": [float(x0), float(y0), float(x1), float(y1)],
                "slices": slice_traces,
            }
        )

    legend_y = height - 58
    legend_items = list(dataset.segment_labels)
    item_width = max(118.0, (width - margin_left - margin_right) / float(max(1, len(legend_items))))
    legend_x0 = (width - (item_width * len(legend_items))) / 2.0
    legend_boxes: list[Sequence[float]] = []
    legend_item_bboxes: dict[str, list[float]] = {}
    for index, segment in enumerate(legend_items):
        x = float(legend_x0 + (index * item_width))
        color = segment_color_by_label[str(segment)]
        swatch_bbox = (float(x), float(legend_y - 11), float(x + 26), float(legend_y + 15))
        legend_text = str(segment)
        text_xy = (float(x + 34), float(legend_y - 12))
        text_bbox = draw.textbbox(text_xy, legend_text, font=legend_font)
        row_bbox = bbox_union((swatch_bbox, text_bbox), padding=4.0)
        legend_boxes.append(row_bbox)
        legend_item_bboxes[str(segment)] = list(row_bbox)
        draw.rectangle(swatch_bbox, fill=color, outline=(80, 84, 92), width=1)
        draw_text_traced(draw, text_xy, legend_text, font=legend_font, fill=text_color, role="readout", required=False)
    legend_bbox = bbox_union(legend_boxes, padding=6.0) if legend_boxes else []

    return RenderedCompositionPanels(
        image=image,
        entities=tuple(entities),
        panel_traces=tuple(panel_traces),
        plot_bbox_px=tuple(int(value) for value in plot_bbox),
        annotation_bbox_by_key=dict(annotation_bbox_by_key),
        total_bbox_by_panel=dict(total_bbox_by_panel),
        legend_bbox_px=list(legend_bbox),
        legend_item_bboxes_px=dict(legend_item_bboxes),
        layout_jitter_meta=dict(layout_jitter_meta),
        render_meta={},
    )


def render_composition_panels(
    dataset: CompositionPanelsDataset,
    *,
    scene_variant: str,
    params: Mapping[str, Any],
    instance_seed: int,
) -> RenderedCompositionPanels:
    """Build the background, draw the scene, and attach render metadata."""

    canvas_width = int(params.get("canvas_width", group_default(RENDER_DEFAULTS, "canvas_width", 1384)))
    canvas_height = int(params.get("canvas_height", group_default(RENDER_DEFAULTS, "canvas_height", 888)))
    information_style, information_style_meta = resolve_chart_information_style(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="composition_panels",
        protected_colors=SEGMENT_COLORS,
    )
    styled_params = {
        **dict(params),
        "text_color_rgb": tuple(int(value) for value in information_style.text_rgb),
        "grid_color_rgb": tuple(int(value) for value in information_style.guide_rgb),
        "plot_fill_rgb": tuple(int(value) for value in information_style.panel_fill_rgb),
        "text_stroke_rgb": tuple(int(value) for value in information_style.text_stroke_rgb),
    }
    background, background_meta = make_chart_information_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=information_style,
        instance_seed=int(instance_seed),
        namespace="charts.composition_panels.information_scene_background",
    )
    rendered = _draw_chart(
        base_image=background,
        dataset=dataset,
        scene_variant=str(scene_variant),
        params=styled_params,
        instance_seed=int(instance_seed),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedCompositionPanels(
        image=image,
        entities=tuple(rendered.entities),
        panel_traces=tuple(rendered.panel_traces),
        plot_bbox_px=tuple(rendered.plot_bbox_px),
        annotation_bbox_by_key=dict(rendered.annotation_bbox_by_key),
        total_bbox_by_panel=dict(rendered.total_bbox_by_panel),
        legend_bbox_px=list(rendered.legend_bbox_px),
        legend_item_bboxes_px=dict(rendered.legend_item_bboxes_px),
        layout_jitter_meta=dict(rendered.layout_jitter_meta),
        render_meta={
            "background_style": {**dict(background_meta), "information_scene_style": dict(information_style_meta)},
            "information_scene_style": dict(information_style_meta),
            "post_image_noise": dict(post_noise_meta),
            "composition_panels_layout": "grid",
            "dense_text_style": dense_text_style_meta(role="composition_segment_values"),
        },
    )


def panel_trace_payload(dataset: CompositionPanelsDataset) -> list[dict[str, Any]]:
    return [
        {
            "panel_label": str(panel.label),
            "total": int(panel.total),
            "shares_by_segment": {str(key): int(value) for key, value in panel.shares_by_segment.items()},
            "counts_by_segment": counts_for_panel(panel),
        }
        for panel in dataset.panels
    ]
