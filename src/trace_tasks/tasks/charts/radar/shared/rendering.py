"""Rendering helpers for radar chart profile tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw, ImageFont

from trace_tasks.tasks.charts.shared.dense_text import dense_fit_bold, dense_stroke_width
from trace_tasks.tasks.charts.shared.information_style import (
    make_chart_information_background,
    resolve_chart_information_style,
)
from .....core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.shared.panel.grid_layout import layout_panel_grid_list
from ....shared.render_variation import apply_layout_jitter_to_margins
from ....shared.text_legibility import draw_text_traced
from ....shared.text_rendering import fit_font_to_box, load_font, temporary_default_font_family
from .defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    RENDER_DEFAULTS,
    bbox as round_bbox,
    resolve_gen_int,
    resolve_int,
    resolve_rgb,
    sample_chart_font_family,
)
from .state import (
    BBox,
    RGB,
    RadarDataset,
    RadarPanel,
    RadarProfile,
    RadarRenderResult,
    RenderedRadarScene,
    SCENE_NAMESPACE,
    SINGLE_PROFILE_SCENE_VARIANT,
    SMALL_MULTIPLE_SCENE_VARIANT,
)

def _text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    *,
    stroke_width: int = 0,
) -> List[float]:
    try:
        box = draw.textbbox((float(xy[0]), float(xy[1])), str(text), font=font, stroke_width=max(0, int(stroke_width)))
        return round_bbox(box)
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        return round_bbox([float(xy[0]), float(xy[1]), float(xy[0]) + float(width), float(xy[1]) + float(height)])

def _center_text(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: RGB,
    stroke_fill: RGB = (255, 255, 255),
    stroke_width: int = 0,
) -> List[float]:
    try:
        box = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width)))
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        x = float(center[0]) - (0.5 * width) - float(box[0])
        y = float(center[1]) - (0.5 * height) - float(box[1])
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        x = float(center[0]) - (0.5 * float(width))
        y = float(center[1]) - (0.5 * float(height))
    draw_text_traced(
        draw,
        (float(x), float(y)),
        str(text),
        font=font,
        fill=fill,
        stroke_fill=stroke_fill,
        stroke_width=max(0, int(stroke_width)),
     role="readout", required=False,)
    return _text_bbox(draw, (float(x), float(y)), str(text), font, stroke_width=max(0, int(stroke_width)))

def _right_aligned_centered_text(
    draw: ImageDraw.ImageDraw,
    *,
    right_x: float,
    center_y: float,
    text: str,
    font: ImageFont.ImageFont,
    fill: RGB,
    stroke_fill: RGB = (255, 255, 255),
    stroke_width: int = 0,
) -> List[float]:
    try:
        box = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width)))
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        x = float(right_x) - float(width) - float(box[0])
        y = float(center_y) - (0.5 * float(height)) - float(box[1])
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        x = float(right_x) - float(width)
        y = float(center_y) - (0.5 * float(height))
    draw_text_traced(
        draw,
        (float(x), float(y)),
        str(text),
        font=font,
        fill=fill,
        stroke_fill=stroke_fill,
        stroke_width=max(0, int(stroke_width)),
        role="readout",
        required=False,
    )
    return _text_bbox(draw, (float(x), float(y)), str(text), font, stroke_width=max(0, int(stroke_width)))

def _panel_layout(plot_bbox: BBox, panel_count: int, gap: float) -> List[BBox]:
    return layout_panel_grid_list(
        plot_bbox,
        panel_count=int(panel_count),
        gap_x=float(gap),
        gap_y=float(gap),
    )

def _radar_points(
    *,
    center: Tuple[float, float],
    radius: float,
    metrics: Sequence[str],
    values: Mapping[str, int],
    max_value: int,
) -> Dict[str, Tuple[float, float]]:
    points: Dict[str, Tuple[float, float]] = {}
    for index, metric in enumerate(metrics):
        angle = (-math.pi / 2.0) + ((2.0 * math.pi * float(index)) / float(len(metrics)))
        radial = float(radius) * (float(values[str(metric)]) / float(max(1, int(max_value))))
        points[str(metric)] = (
            float(center[0]) + (float(radial) * math.cos(float(angle))),
            float(center[1]) + (float(radial) * math.sin(float(angle))),
        )
    return points

def _ring_points(*, center: Tuple[float, float], radius: float, metrics: Sequence[str]) -> List[Tuple[float, float]]:
    points: List[Tuple[float, float]] = []
    for index in range(len(metrics)):
        angle = (-math.pi / 2.0) + ((2.0 * math.pi * float(index)) / float(len(metrics)))
        points.append(
            (
                float(center[0]) + (float(radius) * math.cos(float(angle))),
                float(center[1]) + (float(radius) * math.sin(float(angle))),
            )
        )
    return points

def _draw_radar_panel(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: BBox,
    metrics: Sequence[str],
    panel: RadarPanel,
    params: Mapping[str, Any],
    max_value: int,
    show_title: bool,
    single_panel: bool,
    highlight_metric: str = "",
) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]], List[float]]:
    """Draw one radar panel and return reusable mark projections."""

    x1, y1, x2, y2 = (float(value) for value in bbox)
    text_rgb = resolve_rgb(params, "text_rgb", (37, 45, 58))
    muted_rgb = resolve_rgb(params, "muted_text_rgb", (88, 99, 116))
    text_stroke_rgb = resolve_rgb(params, "text_stroke_rgb", (255, 255, 255))
    grid_rgb = resolve_rgb(params, "grid_rgb", (207, 214, 225))
    spoke_rgb = resolve_rgb(params, "spoke_rgb", (186, 196, 210))
    query_spoke_rgb = resolve_rgb(params, "query_spoke_rgb", (70, 91, 118))
    query_metric_label_rgb = resolve_rgb(params, "query_metric_label_rgb", (26, 54, 93))
    panel_title_font = load_font(resolve_int(params, "panel_title_font_size_px", 20), bold=False)
    metric_font = load_font(resolve_int(params, "metric_font_size_px", 15 if not bool(single_panel) else 18), bold=dense_fit_bold())
    tick_font = load_font(resolve_int(params, "tick_font_size_px", 13 if not bool(single_panel) else 15), bold=False)
    point_radius = float(resolve_int(params, "point_radius_px", 6 if not bool(single_panel) else 8))
    title_bbox = [float(x1), float(y1), float(x1), float(y1)]
    title_height = 0.0
    if bool(show_title) and str(panel.panel_label):
        title_text = f"Panel {str(panel.panel_label)}"
        title_xy = (float(x1) + 14.0, float(y1) + 10.0)
        draw_text_traced(draw, title_xy, title_text, font=panel_title_font, fill=text_rgb, role="readout", required=False)
        title_bbox = _text_bbox(draw, title_xy, title_text, panel_title_font)
        title_height = 34.0

    label_pad = 58.0 if bool(single_panel) else 40.0
    content_top = float(y1) + float(title_height) + 18.0
    content_bottom = float(y2) - 18.0
    center = (0.5 * (float(x1) + float(x2)), 0.5 * (float(content_top) + float(content_bottom)) + 6.0)
    radius = 0.5 * min(float(x2 - x1), float(content_bottom - content_top)) - float(label_pad)
    radius = max(54.0 if not bool(single_panel) else 160.0, float(radius))

    ring_count = max(2, resolve_int(params, "ring_count", 5))
    tick_label_gap = float(resolve_int(params, "tick_label_spoke_gap_px", 10 if not bool(single_panel) else 12))
    for ring_index in range(1, int(ring_count) + 1):
        ring_value = int(round((float(max_value) * float(ring_index)) / float(ring_count)))
        ring_radius = float(radius) * (float(ring_value) / float(max(1, int(max_value))))
        points = _ring_points(center=center, radius=ring_radius, metrics=metrics)
        draw.line([*points, points[0]], fill=grid_rgb, width=resolve_int(params, "grid_line_width_px", 1))
        _right_aligned_centered_text(
            draw,
            right_x=float(center[0]) - float(tick_label_gap),
            center_y=float(center[1]) - float(ring_radius),
            text=str(ring_value),
            font=tick_font,
            fill=muted_rgb,
            stroke_fill=text_stroke_rgb,
            stroke_width=dense_stroke_width(),
        )

    for index, metric in enumerate(metrics):
        is_highlighted_metric = bool(highlight_metric) and str(metric) == str(highlight_metric)
        angle = (-math.pi / 2.0) + ((2.0 * math.pi * float(index)) / float(len(metrics)))
        spoke_end = (
            float(center[0]) + (float(radius) * math.cos(float(angle))),
            float(center[1]) + (float(radius) * math.sin(float(angle))),
        )
        draw.line(
            [center, spoke_end],
            fill=query_spoke_rgb if bool(is_highlighted_metric) else spoke_rgb,
            width=(
                resolve_int(params, "query_spoke_width_px", 3)
                if bool(is_highlighted_metric)
                else resolve_int(params, "grid_line_width_px", 1)
            ),
        )
        label_radius = float(radius) + (34.0 if bool(single_panel) else 24.0)
        raw_label_center = (
            float(center[0]) + (float(label_radius) * math.cos(float(angle))),
            float(center[1]) + (float(label_radius) * math.sin(float(angle))),
        )
        label_font = metric_font
        if not bool(single_panel):
            label_font = fit_font_to_box(
                draw,
                text=str(metric),
                max_width=72,
                max_height=22,
                bold=dense_fit_bold(),
                min_size_px=10,
                max_size_px=resolve_int(params, "metric_font_size_px", 15),
                fill_ratio=0.95,
            )
        _center_text(
            draw,
            center=raw_label_center,
            text=str(metric),
            font=label_font,
            fill=query_metric_label_rgb if bool(is_highlighted_metric) else text_rgb,
            stroke_fill=text_stroke_rgb,
            stroke_width=dense_stroke_width(),
        )

    point_bboxes: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = []
    for profile in panel.profiles:
        points_by_metric = _radar_points(
            center=center,
            radius=float(radius),
            metrics=metrics,
            values=profile.values,
            max_value=int(max_value),
        )
        polygon = [points_by_metric[str(metric)] for metric in metrics]
        if len(polygon) >= 2:
            draw.line(
                [*polygon, polygon[0]],
                fill=tuple(int(value) for value in profile.color_rgb),
                width=resolve_int(params, "profile_line_width_px", 4),
            )
        for metric in metrics:
            cx, cy = points_by_metric[str(metric)]
            marker_bbox = [
                float(cx) - float(point_radius),
                float(cy) - float(point_radius),
                float(cx) + float(point_radius),
                float(cy) + float(point_radius),
            ]
            draw.ellipse(
                marker_bbox,
                fill=tuple(int(value) for value in profile.color_rgb),
                outline=(255, 255, 255),
                width=resolve_int(params, "point_outline_width_px", 2),
            )
            point_id = f"{str(panel.panel_label)}|{str(profile.profile_label)}|{str(metric)}"
            point_bboxes[str(point_id)] = round_bbox(marker_bbox)
            entities.append(
                {
                    "entity_id": str(point_id),
                    "entity_type": "chart_radar_vertex",
                    "bbox_px": round_bbox(marker_bbox),
                    "attrs": {
                        "panel_label": str(panel.panel_label),
                        "profile_label": str(profile.profile_label),
                        "metric_label": str(metric),
                        "value": int(profile.values[str(metric)]),
                        "center_px": [round(float(cx), 3), round(float(cy), 3)],
                    },
                }
            )
    return entities, point_bboxes, list(title_bbox)

def _draw_legend(
    draw: ImageDraw.ImageDraw,
    *,
    profiles: Sequence[RadarProfile],
    params: Mapping[str, Any],
    origin: Tuple[float, float],
) -> Dict[str, List[float]]:
    font = load_font(resolve_int(params, "legend_font_size_px", 18), bold=False)
    text_rgb = resolve_rgb(params, "text_rgb", (37, 45, 58))
    legend_bboxes: Dict[str, List[float]] = {}
    x = float(origin[0])
    y = float(origin[1])
    for profile in profiles:
        swatch = [x, y + 5.0, x + 24.0, y + 19.0]
        draw.rounded_rectangle(swatch, radius=4, fill=profile.color_rgb, outline=(255, 255, 255), width=1)
        label_xy = (x + 32.0, y)
        draw_text_traced(draw, label_xy, str(profile.profile_label), font=font, fill=text_rgb, role="readout", required=False)
        text_box = _text_bbox(draw, label_xy, str(profile.profile_label), font)
        legend_bboxes[str(profile.profile_label)] = round_bbox([swatch[0], swatch[1], text_box[2], text_box[3]])
        x = float(text_box[2]) + 42.0
    return legend_bboxes

def _render_dataset(dataset: RadarDataset, *, params: Mapping[str, Any], instance_seed: int) -> RenderedRadarScene:
    """Render a complete radar chart scene without owning public task behavior."""

    params = {**dict(params), "_render_style_seed": int(instance_seed)}
    canvas_width = resolve_int(params, "canvas_width", 1424)
    canvas_height = resolve_int(params, "canvas_height", 888)
    protected_colors = [
        tuple(int(channel) for channel in profile.color_rgb)
        for panel in dataset.panels
        for profile in panel.profiles
    ]
    information_style, information_style_meta = resolve_chart_information_style(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="radar",
        protected_colors=protected_colors,
    )
    params = {
        **params,
        "text_rgb": tuple(int(value) for value in information_style.text_rgb),
        "muted_text_rgb": tuple(int(value) for value in information_style.muted_text_rgb),
        "grid_rgb": tuple(int(value) for value in information_style.grid_rgb),
        "spoke_rgb": tuple(int(value) for value in information_style.guide_rgb),
        "panel_fill_rgb": tuple(int(value) for value in information_style.panel_fill_rgb),
        "panel_border_rgb": tuple(int(value) for value in information_style.panel_border_rgb),
        "text_stroke_rgb": tuple(int(value) for value in information_style.text_stroke_rgb),
        "query_spoke_rgb": tuple(int(value) for value in information_style.highlight_rgb),
        "query_metric_label_rgb": tuple(int(value) for value in information_style.highlight_rgb),
    }
    background, background_meta = make_chart_information_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=information_style,
        instance_seed=int(instance_seed),
        namespace="charts.radar.information_scene_background",
    )
    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    outer = resolve_int(params, "outer_margin_px", 42)
    margin_left, margin_right, margin_top, margin_bottom, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(outer),
        right_px=int(outer),
        top_px=int(outer),
        bottom_px=int(outer),
        params=params,
        defaults=RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.layout",
    )
    title_band = resolve_int(params, "title_band_height_px", 86)
    panel_gap = resolve_int(params, "panel_gap_px", 26)
    panel_padding = resolve_int(params, "panel_padding_px", 18)
    panel_fill = resolve_rgb(params, "panel_fill_rgb", (255, 255, 255))
    panel_border = resolve_rgb(params, "panel_border_rgb", (190, 199, 212))
    text_rgb = resolve_rgb(params, "text_rgb", (37, 45, 58))
    muted_rgb = resolve_rgb(params, "muted_text_rgb", (88, 99, 116))
    title_font = load_font(resolve_int(params, "title_font_size_px", 30), bold=False)
    subtitle_font = load_font(resolve_int(params, "subtitle_font_size_px", 18), bold=False)

    draw_text_traced(
        draw,
        (float(margin_left), 22.0 + float(layout_jitter_meta.get("dy_px", 0))),
        "Radar Profile Charts",
        font=title_font,
        fill=text_rgb,
     role="readout", required=False,)
    subtitle = "Compare radial profile vertices against metric spokes and ring values."
    draw_text_traced(
        draw,
        (float(margin_left), 58.0 + float(layout_jitter_meta.get("dy_px", 0))),
        subtitle,
        font=subtitle_font,
        fill=muted_rgb,
     role="readout", required=False,)

    entities: List[Dict[str, Any]] = []
    point_bboxes: Dict[str, List[float]] = {}
    panel_title_bboxes: Dict[str, List[float]] = {}
    panel_bboxes: Dict[str, List[float]] = {}
    legend_bboxes: Dict[str, List[float]] = {}
    plot_bbox = [
        float(margin_left),
        float(title_band) + float(layout_jitter_meta.get("dy_px", 0)),
        float(canvas_width - margin_right),
        float(canvas_height - margin_bottom),
    ]
    max_value = resolve_gen_int(params, "value_max", 10)

    if str(dataset.scene_variant) == SINGLE_PROFILE_SCENE_VARIANT:
        legend_bboxes = _draw_legend(
            draw,
            profiles=dataset.panels[0].profiles,
            params=params,
            origin=(float(canvas_width) - 380.0, 34.0),
        )

    panel_boxes = _panel_layout(tuple(plot_bbox), len(dataset.panels), gap=float(panel_gap))
    for panel, panel_bbox in zip(dataset.panels, panel_boxes):
        px1, py1, px2, py2 = (float(value) for value in panel_bbox)
        draw.rounded_rectangle(
            [px1, py1, px2, py2],
            radius=resolve_int(params, "panel_corner_radius_px", 8),
            fill=panel_fill,
            outline=panel_border,
            width=resolve_int(params, "panel_border_width_px", 2),
        )
        content_bbox = (
            px1 + float(panel_padding),
            py1 + float(panel_padding),
            px2 - float(panel_padding),
            py2 - float(panel_padding),
        )
        rendered_entities, rendered_points, title_bbox = _draw_radar_panel(
            draw,
            bbox=content_bbox,
            metrics=dataset.metrics,
            panel=panel,
            params=params,
            max_value=int(max_value),
            show_title=str(dataset.scene_variant) == SMALL_MULTIPLE_SCENE_VARIANT,
            single_panel=str(dataset.scene_variant) == SINGLE_PROFILE_SCENE_VARIANT,
            highlight_metric=str(dataset.highlight_metric_label),
        )
        panel_bboxes[str(panel.panel_label)] = round_bbox(panel_bbox)
        if str(panel.panel_label):
            panel_title_bboxes[str(panel.panel_label)] = list(title_bbox)
        entities.extend(rendered_entities)
        point_bboxes.update(rendered_points)
        entities.append(
            {
                "entity_id": f"radar_panel_{str(panel.panel_label) or 'single'}",
                "entity_type": "chart_radar_panel",
                "bbox_px": round_bbox(panel_bbox),
                "attrs": {
                    "panel_label": str(panel.panel_label),
                    "profile_count": int(len(panel.profiles)),
                    "metric_count": int(len(dataset.metrics)),
                },
            }
        )

    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedRadarScene(
        image=image,
        entities=tuple(entities),
        point_bboxes=dict(point_bboxes),
        panel_bboxes=dict(panel_bboxes),
        panel_title_bboxes=dict(panel_title_bboxes),
        legend_bboxes=dict(legend_bboxes),
        plot_bbox_px=round_bbox(plot_bbox),
        render_meta={
            "background_style": {**dict(background_meta), "information_scene_style": dict(information_style_meta)},
            "information_scene_style": dict(information_style_meta),
            "post_image_noise": dict(post_noise_meta),
            "layout_jitter": dict(layout_jitter_meta),
            "panel_bboxes_px": dict(panel_bboxes),
            "metric_labels": list(dataset.metrics),
            "highlight_metric_label": str(dataset.highlight_metric_label),
        },
    )


def render_radar_dataset(
    *,
    dataset: RadarDataset,
    params: Mapping[str, Any],
    instance_seed: int,
) -> RadarRenderResult:
    chart_font_family = sample_chart_font_family(int(instance_seed), params)
    with temporary_default_font_family(str(chart_font_family)):
        rendered_scene = _render_dataset(dataset, params=params, instance_seed=int(instance_seed))
    return RadarRenderResult(rendered_scene=rendered_scene, chart_font_family=str(chart_font_family))


__all__ = ["render_radar_dataset"]
