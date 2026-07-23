"""Rendering helpers for scatter-cluster chart scenes."""

from __future__ import annotations

import math
from typing import Any, Sequence

from PIL import Image, ImageDraw

from trace_tasks.tasks.charts.shared.cartesian.axes import (
    draw_horizontal_value_grid_ticks,
    draw_plot_frame,
    draw_vertical_value_grid_ticks,
)
from trace_tasks.tasks.charts.shared.cartesian.frame import plot_bbox_from_margins
from trace_tasks.tasks.charts.shared.cartesian.geometry import project_xy, round_bbox
from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.bbox_projection import bbox_union_raw as _bbox_union
from trace_tasks.tasks.shared.text_legibility import draw_centered_traced_text, draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font, temporary_default_font_family

from .defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    resolve_render_params,
    sample_chart_font_family,
)
from .state import (
    AreaEnvelope,
    RGB,
    RenderedScatterCluster,
    ScatterClusterDataset,
    ScatterClusterRenderParams,
    ScatterClusterRenderResult,
    ScatterPoint,
)


def _bbox(values: Sequence[float]) -> list[float]:
    return round_bbox(values)


def _draw_text_box(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: tuple[float, float],
    *,
    font: Any,
    fill: RGB,
    stroke_fill: RGB,
    stroke_width: int = 0,
) -> list[float]:
    try:
        draw_text_traced(
            draw,
            (float(xy[0]), float(xy[1])),
            str(text),
            font=font,
            fill=fill,
            stroke_fill=stroke_fill,
            stroke_width=int(stroke_width),
            role="readout",
            required=False,
        )
        box = draw.textbbox((float(xy[0]), float(xy[1])), str(text), font=font, stroke_width=int(stroke_width))
        return _bbox(box)
    except Exception:
        draw_text_traced(
            draw,
            (float(xy[0]), float(xy[1])),
            str(text),
            font=font,
            fill=fill,
            role="readout",
            required=False,
        )
        width, height = draw.textsize(str(text), font=font)
        return _bbox([float(xy[0]), float(xy[1]), float(xy[0]) + float(width), float(xy[1]) + float(height)])


def _plot_value_xy(x_value: float, y_value: float, *, plot_bbox: Sequence[float]) -> tuple[float, float]:
    return project_xy(
        x_value=float(x_value),
        y_value=float(y_value),
        plot_bbox=plot_bbox,
        x_min=0.0,
        x_max=100.0,
        y_min=0.0,
        y_max=100.0,
    )


def _plot_xy(point: ScatterPoint, *, plot_bbox: Sequence[float]) -> tuple[float, float]:
    return _plot_value_xy(float(point.x_value), float(point.y_value), plot_bbox=plot_bbox)


def _area_envelope_polygon_px(
    envelope: AreaEnvelope,
    *,
    plot_bbox: Sequence[float],
    segments: int = 56,
) -> list[tuple[float, float]]:
    angle = math.radians(float(envelope.angle_degrees))
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    polygon: list[tuple[float, float]] = []
    for index in range(int(segments)):
        theta = 2.0 * math.pi * (float(index) / max(1.0, float(segments)))
        local_x = float(envelope.radius_x) * math.cos(theta)
        local_y = float(envelope.radius_y) * math.sin(theta)
        x_value = float(envelope.center_x) + local_x * cos_a - local_y * sin_a
        y_value = float(envelope.center_y) + local_x * sin_a + local_y * cos_a
        polygon.append(_plot_value_xy(float(x_value), float(y_value), plot_bbox=plot_bbox))
    return polygon


def _bbox_from_points(points: Sequence[tuple[float, float]], *, padding: float = 0.0) -> list[float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return _bbox([min(xs) - float(padding), min(ys) - float(padding), max(xs) + float(padding), max(ys) + float(padding)])


def render_scatter_scene(
    image: Image.Image,
    *,
    dataset: ScatterClusterDataset,
    render_params: ScatterClusterRenderParams,
) -> RenderedScatterCluster:
    """Draw the scatter plot and record pixel boxes for clusters, points, options, and legend rows."""

    draw = ImageDraw.Draw(image)
    width, height = image.size
    plot_bbox = plot_bbox_from_margins(
        canvas_width=float(width),
        canvas_height=float(height),
        margin_left_px=float(render_params.plot_margin_left_px),
        margin_right_px=float(render_params.plot_margin_right_px),
        margin_top_px=float(render_params.plot_margin_top_px),
        margin_bottom_px=float(render_params.plot_margin_bottom_px),
    )
    title_font = load_font(int(render_params.title_font_size_px), bold=True)
    tick_font = load_font(int(render_params.tick_font_size_px), bold=False)
    axis_label_font = load_font(18, bold=True)
    legend_font = load_font(int(render_params.legend_font_size_px), bold=True)
    panel_bbox = [
        float(plot_bbox[0] - 56.0),
        float(plot_bbox[1] - 52.0),
        float(width - 36.0),
        float(plot_bbox[3] + 72.0),
    ]
    draw.rounded_rectangle(
        panel_bbox,
        radius=6,
        fill=render_params.panel_fill_rgb,
        outline=render_params.panel_border_rgb,
        width=2,
    )
    draw_plot_frame(
        draw,
        plot_bbox,
        fill=render_params.plot_fill_rgb,
        outline=render_params.axis_color_rgb,
        width=int(render_params.axis_line_width_px),
    )
    tick_values = range(0, 101, 20)
    x_tick_positions = draw_vertical_value_grid_ticks(
        draw,
        plot_bbox,
        tick_values=tick_values,
        domain_min=0,
        domain_max=100,
        grid_rgb=render_params.grid_color_rgb,
        axis_rgb=render_params.axis_color_rgb,
        grid_width_px=int(render_params.grid_line_width_px),
        tick_width_px=1,
        tick_length_px=float(render_params.tick_length_px),
    )
    y_tick_positions = draw_horizontal_value_grid_ticks(
        draw,
        plot_bbox,
        tick_values=tick_values,
        domain_min=0,
        domain_max=100,
        grid_rgb=render_params.grid_color_rgb,
        axis_rgb=render_params.axis_color_rgb,
        grid_width_px=int(render_params.grid_line_width_px),
        tick_width_px=1,
        tick_length_px=float(render_params.tick_length_px),
    )
    for tick in tick_values:
        x = float(x_tick_positions[float(tick)])
        y = float(y_tick_positions[float(tick)])
        draw_text_traced(draw, (x, plot_bbox[3] + 12), str(tick), font=tick_font, fill=render_params.text_color_rgb, anchor="mt", role="readout", required=False)
        draw_text_traced(draw, (plot_bbox[0] - 13, y), str(tick), font=tick_font, fill=render_params.text_color_rgb, anchor="rm", role="readout", required=False)

    title_bbox = _draw_text_box(
        draw,
        "Scatter Plot",
        (plot_bbox[0], panel_bbox[1] + 16),
        font=title_font,
        fill=render_params.text_color_rgb,
        stroke_fill=render_params.text_stroke_rgb,
    )
    x_label_box = _draw_text_box(
        draw,
        "X score",
        ((plot_bbox[0] + plot_bbox[2]) / 2.0 - 28, plot_bbox[3] + 54),
        font=axis_label_font,
        fill=render_params.text_color_rgb,
        stroke_fill=render_params.text_stroke_rgb,
    )
    y_label_box = _draw_text_box(
        draw,
        "Y score",
        (plot_bbox[0] - 84, plot_bbox[1] - 28),
        font=axis_label_font,
        fill=render_params.text_color_rgb,
        stroke_fill=render_params.text_stroke_rgb,
    )

    point_bboxes: dict[str, list[float]] = {}
    cluster_bboxes: dict[str, list[float]] = {}
    cluster_envelope_bboxes: dict[str, list[float]] = {}
    cluster_label_bboxes: dict[str, list[float]] = {}
    legend_bboxes: dict[str, list[float]] = {}
    option_bboxes: dict[str, list[float]] = {}
    option_centers_px: dict[str, list[float]] = {}
    entities: list[dict[str, Any]] = [
        {"entity_id": "scatter_panel", "entity_type": "chart_panel", "bbox_xyxy": _bbox(panel_bbox), "attrs": {}},
        {"entity_id": "scatter_plot", "entity_type": "scatter_plot", "bbox_xyxy": _bbox(plot_bbox), "attrs": {}},
        {"entity_id": "chart_title", "entity_type": "chart_title", "bbox_xyxy": title_bbox, "attrs": {"title": "Scatter Plot"}},
        {"entity_id": "x_axis_label", "entity_type": "axis_label", "bbox_xyxy": x_label_box, "attrs": {"axis": "x"}},
        {"entity_id": "y_axis_label", "entity_type": "axis_label", "bbox_xyxy": y_label_box, "attrs": {"axis": "y"}},
    ]

    envelope_polygons: dict[str, list[tuple[float, float]]] = {}
    for cluster in dataset.clusters:
        if cluster.area_envelope is None:
            continue
        polygon = _area_envelope_polygon_px(cluster.area_envelope, plot_bbox=plot_bbox)
        envelope_polygons[str(cluster.cluster_label)] = list(polygon)
        cluster_envelope_bboxes[str(cluster.cluster_label)] = _bbox_from_points(polygon, padding=3.0)
    if envelope_polygons:
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        for cluster in dataset.clusters:
            polygon = envelope_polygons.get(str(cluster.cluster_label))
            if not polygon:
                continue
            overlay_draw.polygon(polygon, fill=(*tuple(cluster.color_rgb), 54))
            overlay_draw.line(list(polygon) + [polygon[0]], fill=(*tuple(cluster.color_rgb), 210), width=3)
        image.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB"))
        draw = ImageDraw.Draw(image)

    radius = float(render_params.point_radius_px)
    for cluster in dataset.clusters:
        plotted = [_plot_xy(point, plot_bbox=plot_bbox) for point in cluster.points]
        point_boxes_for_cluster: list[list[float]] = []
        for point, (px, py) in zip(cluster.points, plotted):
            bbox = [px - radius, py - radius, px + radius, py + radius]
            point_bboxes[str(point.point_id)] = _bbox(bbox)
            point_boxes_for_cluster.append(_bbox(bbox))
        hull = _bbox_union(point_boxes_for_cluster)
        hull_pad = float(render_params.cluster_hull_padding_px)
        hull = _bbox([hull[0] - hull_pad, hull[1] - hull_pad, hull[2] + hull_pad, hull[3] + hull_pad])
        if str(cluster.cluster_label) in cluster_envelope_bboxes:
            hull = list(cluster_envelope_bboxes[str(cluster.cluster_label)])
        cluster_bboxes[str(cluster.cluster_label)] = list(hull)
        for point, (px, py) in zip(cluster.points, plotted):
            point_box = point_bboxes[str(point.point_id)]
            draw.ellipse(point_box, fill=cluster.color_rgb, outline=(255, 255, 255), width=2)
            entities.append(
                {
                    "entity_id": str(point.point_id),
                    "entity_type": "scatter_point",
                    "bbox_xyxy": list(point_box),
                    "attrs": {
                        "cluster_label": str(cluster.cluster_label),
                        "x_value": round(float(point.x_value), 3),
                        "y_value": round(float(point.y_value), 3),
                    },
                }
            )
        entities.append(
            {
                "entity_id": f"cluster_{cluster.cluster_label}",
                "entity_type": "scatter_cluster",
                "bbox_xyxy": list(hull),
                "attrs": {
                    "cluster_label": str(cluster.cluster_label),
                    "center_x": round(float(cluster.center_x), 3),
                    "center_y": round(float(cluster.center_y), 3),
                    "slope": round(float(cluster.slope), 4),
                    "spread_x": round(float(cluster.spread_x), 3),
                    "spread_y": round(float(cluster.spread_y), 3),
                    "point_ids": [str(point.point_id) for point in cluster.points],
                    "area_envelope": (
                        {
                            "center": [
                                round(float(cluster.area_envelope.center_x), 3),
                                round(float(cluster.area_envelope.center_y), 3),
                            ],
                            "radius_x": round(float(cluster.area_envelope.radius_x), 3),
                            "radius_y": round(float(cluster.area_envelope.radius_y), 3),
                            "angle_degrees": round(float(cluster.area_envelope.angle_degrees), 3),
                            "area_value": round(float(cluster.area_envelope.area_value), 4),
                        }
                        if cluster.area_envelope is not None
                        else None
                    ),
                },
            }
        )

    if dataset.option_markers:
        marker_radius = max(13.0, float(render_params.point_radius_px) + 7.0)
        option_font = load_font(max(18, int(render_params.legend_font_size_px)), bold=True)
        for option in dataset.option_markers:
            px, py = _plot_value_xy(float(option.x_value), float(option.y_value), plot_bbox=plot_bbox)
            bbox = _bbox([px - marker_radius, py - marker_radius, px + marker_radius, py + marker_radius])
            option_bboxes[str(option.option_label)] = list(bbox)
            option_centers_px[str(option.option_label)] = [round(float(px), 3), round(float(py), 3)]
            draw.ellipse(bbox, fill=render_params.panel_fill_rgb, outline=render_params.axis_color_rgb, width=3)
            draw_centered_traced_text(
                draw,
                center=(float(px), float(py)),
                text=str(option.option_label),
                font=option_font,
                fill_rgb=render_params.text_color_rgb,
                stroke_width=1,
                stroke_rgb=render_params.text_stroke_rgb,
                role="readout",
                required=False,
            )
            entities.append(
                {
                    "entity_id": f"option_marker_{option.option_label}",
                    "entity_type": "option_marker",
                    "bbox_xyxy": list(bbox),
                    "attrs": {
                        "option_label": str(option.option_label),
                        "x_value": round(float(option.x_value), 3),
                        "y_value": round(float(option.y_value), 3),
                    },
                }
            )

    legend_left = plot_bbox[2] + 36.0
    legend_top = plot_bbox[1] + 34.0
    legend_row_height = 54.0
    legend_swatch_size = 32.0
    legend_row_right = float(width - 54.0)
    _draw_text_box(
        draw,
        "Groups",
        (legend_left, legend_top - 32.0),
        font=legend_font,
        fill=render_params.text_color_rgb,
        stroke_fill=render_params.text_stroke_rgb,
    )
    for index, cluster in enumerate(dataset.clusters):
        y = legend_top + float(index) * legend_row_height
        row_box = [legend_left - 10.0, y - 7.0, legend_row_right, y + legend_swatch_size + 7.0]
        draw.rounded_rectangle(row_box, radius=6, fill=render_params.panel_fill_rgb, outline=render_params.panel_border_rgb, width=1)
        swatch = [legend_left, y, legend_left + legend_swatch_size, y + legend_swatch_size]
        draw.rounded_rectangle(swatch, radius=7, fill=cluster.color_rgb, outline=(255, 255, 255), width=2)
        text_box = _draw_text_box(
            draw,
            str(cluster.cluster_label),
            (legend_left + legend_swatch_size + 14.0, y + 1.0),
            font=legend_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
        )
        cluster_label_bboxes[str(cluster.cluster_label)] = list(text_box)
        legend_bboxes[str(cluster.cluster_label)] = _bbox_union([row_box, swatch, text_box])
        entities.append(
            {
                "entity_id": f"legend_{cluster.cluster_label}",
                "entity_type": "legend_entry",
                "bbox_xyxy": list(legend_bboxes[str(cluster.cluster_label)]),
                "attrs": {"cluster_label": str(cluster.cluster_label)},
            }
        )

    return RenderedScatterCluster(
        image=image,
        entities=tuple(dict(item) for item in entities),
        plot_bbox_px=_bbox(plot_bbox),
        point_bboxes=dict(point_bboxes),
        cluster_bboxes=dict(cluster_bboxes),
        cluster_envelope_bboxes=dict(cluster_envelope_bboxes),
        cluster_label_bboxes=dict(cluster_label_bboxes),
        legend_bboxes=dict(legend_bboxes),
        option_bboxes=dict(option_bboxes),
        option_centers_px=dict(option_centers_px),
    )


def render_scatter_cluster_dataset(
    *,
    dataset: ScatterClusterDataset,
    params: dict[str, Any],
    instance_seed: int,
) -> ScatterClusterRenderResult:
    render_style_params = {**dict(params), "_render_style_seed": int(instance_seed)}
    resolved_params = resolve_render_params(render_style_params)
    protected_colors = [tuple(int(channel) for channel in cluster.color_rgb) for cluster in dataset.clusters]
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="scatter_cluster",
        render_params=resolved_params,
        protected_colors=protected_colors,
    )
    chart_font_family = sample_chart_font_family(int(instance_seed), params)
    with temporary_default_font_family(str(chart_font_family)):
        rendered_scene = render_scatter_scene(background, dataset=dataset, render_params=render_params)
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return ScatterClusterRenderResult(
        image=image,
        rendered_scene=rendered_scene,
        render_params=render_params,
        background_meta={**dict(background_meta), "information_scene_style": dict(information_style_meta)},
        post_noise_meta=dict(post_noise_meta),
        chart_font_family=str(chart_font_family),
    )
