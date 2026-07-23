"""Rendering primitives for scatter-readout chart scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.cartesian.axes import (
    draw_horizontal_value_grid_ticks,
    draw_plot_frame,
    draw_vertical_index_grid_ticks,
)
from trace_tasks.tasks.charts.shared.cartesian.frame import plot_bbox_from_margins
from trace_tasks.tasks.charts.shared.cartesian.geometry import project_index, project_linear_inverted, round_bbox
from trace_tasks.tasks.charts.shared.cartesian.markers import draw_marker as draw_cartesian_marker
from trace_tasks.tasks.charts.shared.dense_text import dense_fit_bold, dense_stroke_width
from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.shared.visual_defaults import (
    chart_font_asset_metadata,
    sample_chart_font_family as sample_shared_chart_font_family,
)
from trace_tasks.tasks.shared.bbox_projection import bbox_union_raw as bbox_union
from trace_tasks.tasks.shared.render_variation import apply_layout_jitter_to_margins
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font, temporary_default_font_family

from .defaults import (
    NOISE_DEFAULTS,
    RENDER_DEFAULTS,
    render_style_seed,
    resolve_float,
    resolve_int,
    resolve_rgb,
)
from .state import (
    SCENE_NAMESPACE,
    Point,
    RGB,
    RenderParams,
    RenderedScene,
    ScatterReadoutRenderResult,
    SceneDataset,
)


def bbox(values: Sequence[float]) -> list[float]:
    return round_bbox(values)


def resolve_render_params(params: Mapping[str, Any]) -> RenderParams:
    """Resolve deterministic style parameters while keeping the visual grammar fixed."""

    margin_left = resolve_int(params, "readout_plot_margin_left_px", 112)
    margin_right = resolve_int(params, "readout_plot_margin_right_px", 370)
    margin_top = resolve_int(params, "readout_plot_margin_top_px", 78)
    margin_bottom = resolve_int(params, "readout_plot_margin_bottom_px", 126)
    margin_left, margin_right, margin_top, margin_bottom, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(margin_left),
        right_px=int(margin_right),
        top_px=int(margin_top),
        bottom_px=int(margin_bottom),
        params=params,
        defaults=RENDER_DEFAULTS,
        instance_seed=render_style_seed(params),
        namespace=f"{SCENE_NAMESPACE}.layout",
    )
    title_probability = resolve_float(params, "readout_title_probability", 0.35)
    title_rng = spawn_rng(render_style_seed(params), f"{SCENE_NAMESPACE}.title")
    return RenderParams(
        canvas_width=resolve_int(params, "readout_canvas_width", 1440),
        canvas_height=resolve_int(params, "readout_canvas_height", 820),
        plot_margin_left_px=int(margin_left),
        plot_margin_right_px=int(margin_right),
        plot_margin_top_px=int(margin_top),
        plot_margin_bottom_px=int(margin_bottom),
        axis_line_width_px=resolve_int(params, "axis_line_width_px", 2),
        grid_line_width_px=resolve_int(params, "grid_line_width_px", 1),
        tick_length_px=resolve_int(params, "tick_length_px", 8),
        point_radius_px=resolve_int(params, "readout_point_radius_px", 8),
        tick_font_size_px=resolve_int(params, "readout_tick_font_size_px", 17),
        label_font_size_px=resolve_int(params, "readout_axis_label_font_size_px", 19),
        value_font_size_px=resolve_int(params, "readout_value_font_size_px", 16),
        legend_font_size_px=resolve_int(params, "readout_legend_font_size_px", 21),
        title_font_size_px=resolve_int(params, "readout_title_font_size_px", 26),
        legend_gap_px=resolve_int(params, "readout_legend_gap_px", 80),
        show_title=bool(title_rng.random() < max(0.0, min(1.0, float(title_probability)))),
        axis_color_rgb=resolve_rgb(params, "axis_color_rgb", (65, 70, 78)),
        grid_color_rgb=resolve_rgb(params, "grid_color_rgb", (224, 228, 235)),
        text_color_rgb=resolve_rgb(params, "text_color_rgb", (35, 38, 45)),
        text_stroke_rgb=resolve_rgb(params, "text_stroke_rgb", (255, 255, 255)),
        plot_fill_rgb=resolve_rgb(params, "plot_fill_rgb", (255, 255, 255)),
        panel_fill_rgb=resolve_rgb(params, "panel_fill_rgb", (252, 253, 255)),
        panel_border_rgb=resolve_rgb(params, "panel_border_rgb", (203, 209, 219)),
        layout_jitter_meta=dict(layout_jitter_meta),
    )


def sample_chart_font_family(instance_seed: int, params: Mapping[str, Any]) -> str:
    return sample_shared_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
    )


def draw_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: Tuple[float, float],
    *,
    font: Any,
    fill: RGB,
    stroke_fill: RGB,
    stroke_width: int = 0,
    anchor: str | None = None,
) -> list[float]:
    kwargs: Dict[str, Any] = {
        "font": font,
        "fill": fill,
        "stroke_fill": stroke_fill,
        "stroke_width": int(stroke_width),
    }
    if anchor is not None:
        kwargs["anchor"] = str(anchor)
    try:
        draw_text_traced(draw, (float(xy[0]), float(xy[1])), str(text), **kwargs, role="readout", required=False)
        return bbox(
            draw.textbbox(
                (float(xy[0]), float(xy[1])),
                str(text),
                font=font,
                stroke_width=int(stroke_width),
                anchor=anchor,
            )
        )
    except Exception:
        kwargs.pop("anchor", None)
        draw_text_traced(draw, (float(xy[0]), float(xy[1])), str(text), **kwargs, role="readout", required=False)
        width, height = draw.textsize(str(text), font=font)
        return bbox([float(xy[0]), float(xy[1]), float(xy[0]) + float(width), float(xy[1]) + float(height)])


def point_xy(point: Point, *, x_count: int, plot_bbox: Sequence[float]) -> tuple[float, float]:
    left, top, right, bottom = [float(value) for value in plot_bbox]
    x = project_index(int(point.x_index), pixel_min=float(left), pixel_max=float(right), count=int(x_count))
    y = project_linear_inverted(
        float(point.y_value),
        domain_min=0.0,
        domain_max=100.0,
        pixel_top=float(top),
        pixel_bottom=float(bottom),
    )
    return float(x), float(y)


def draw_marker(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    radius: float,
    shape: str,
    fill: RGB,
) -> list[float]:
    outline = fill if str(shape) == "ring" else (255, 255, 255)
    return draw_cartesian_marker(
        draw,
        center=(float(center[0]), float(center[1])),
        radius=float(radius),
        shape=str(shape),
        fill=fill,
        outline=outline,
        width=4 if str(shape) == "ring" else 2,
        triangle_style="down",
        ring_style="outline",
    )


def render_scatter_readout_scene(
    image: Image.Image,
    *,
    dataset: SceneDataset,
    render_params: RenderParams,
) -> RenderedScene:
    """Draw the full scatter-readout visual grammar and record projected witnesses."""

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
    title_font = load_font(int(render_params.title_font_size_px), bold=False)
    tick_font = load_font(int(render_params.tick_font_size_px), bold=False)
    axis_font = load_font(int(render_params.label_font_size_px), bold=False)
    value_font = load_font(int(render_params.value_font_size_px), bold=dense_fit_bold())
    legend_font = load_font(int(render_params.legend_font_size_px), bold=False)

    panel_bbox = [
        float(plot_bbox[0] - 60.0),
        float(plot_bbox[1] - 54.0),
        float(width - 36.0),
        float(plot_bbox[3] + 78.0),
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

    x_label_bboxes: dict[str, list[float]] = {}
    y_tick_positions = draw_horizontal_value_grid_ticks(
        draw,
        plot_bbox,
        tick_values=range(0, 101, 20),
        domain_min=0,
        domain_max=100,
        grid_rgb=render_params.grid_color_rgb,
        axis_rgb=render_params.axis_color_rgb,
        grid_width_px=int(render_params.grid_line_width_px),
        tick_width_px=1,
        tick_length_px=float(render_params.tick_length_px),
    )
    for tick, y in y_tick_positions.items():
        draw_text(
            draw,
            str(int(tick)),
            (plot_bbox[0] - 13.0, y),
            font=tick_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            anchor="rm",
        )

    x_count = len(dataset.x_labels)
    x_tick_positions = draw_vertical_index_grid_ticks(
        draw,
        plot_bbox,
        count=int(x_count),
        grid_rgb=render_params.grid_color_rgb,
        axis_rgb=render_params.axis_color_rgb,
        grid_width_px=int(render_params.grid_line_width_px),
        tick_width_px=1,
        tick_length_px=float(render_params.tick_length_px),
    )
    for index, label in enumerate(dataset.x_labels):
        x = float(x_tick_positions[int(index)])
        x_label_bboxes[str(label)] = draw_text(
            draw,
            str(label),
            (x, plot_bbox[3] + 14.0),
            font=tick_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            anchor="mt",
        )

    title_text = "Series Scatter Plot" if render_params.show_title else ""
    title_box: list[float] = []
    if title_text:
        title_box = draw_text(
            draw,
            title_text,
            (plot_bbox[0], panel_bbox[1] + 16.0),
            font=title_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
        )
    x_axis_box = draw_text(
        draw,
        dataset.x_axis_title,
        ((plot_bbox[0] + plot_bbox[2]) / 2.0, plot_bbox[3] + 58.0),
        font=axis_font,
        fill=render_params.text_color_rgb,
        stroke_fill=render_params.text_stroke_rgb,
        anchor="mt",
    )
    y_axis_box = draw_text(
        draw,
        dataset.y_axis_title,
        (plot_bbox[0] - 80.0, plot_bbox[1] - 30.0),
        font=axis_font,
        fill=render_params.text_color_rgb,
        stroke_fill=render_params.text_stroke_rgb,
    )

    entities: list[dict[str, Any]] = [
        {"entity_id": "scatter_panel", "entity_type": "chart_panel", "bbox_xyxy": bbox(panel_bbox), "attrs": {}},
        {"entity_id": "scatter_plot", "entity_type": "scatter_plot", "bbox_xyxy": bbox(plot_bbox), "attrs": {}},
        {"entity_id": "x_axis_label", "entity_type": "axis_label", "bbox_xyxy": x_axis_box, "attrs": {"axis": "x", "label": dataset.x_axis_title}},
        {"entity_id": "y_axis_label", "entity_type": "axis_label", "bbox_xyxy": y_axis_box, "attrs": {"axis": "y", "label": dataset.y_axis_title}},
    ]
    if title_text:
        entities.append(
            {"entity_id": "chart_title", "entity_type": "chart_title", "bbox_xyxy": title_box, "attrs": {"title": title_text}}
        )

    point_bboxes: dict[str, list[float]] = {}
    value_label_bboxes: dict[str, list[float]] = {}
    point_annotation_bboxes: dict[str, list[float]] = {}
    label_offsets = [(-22.0, -18.0), (22.0, -18.0), (-22.0, 18.0), (22.0, 18.0), (0.0, -30.0)]
    for series_index, series_item in enumerate(dataset.series):
        for point in series_item.points:
            px, py = point_xy(point, x_count=x_count, plot_bbox=plot_bbox)
            point_box = draw_marker(
                draw,
                center=(px, py),
                radius=float(render_params.point_radius_px),
                shape=str(series_item.marker_shape),
                fill=series_item.color_rgb,
            )
            point_bboxes[str(point.point_id)] = list(point_box)
            offset = label_offsets[int(series_index) % len(label_offsets)]
            value_box = draw_text(
                draw,
                str(point.y_value),
                (px + float(offset[0]), py + float(offset[1])),
                font=value_font,
                fill=render_params.text_color_rgb,
                stroke_fill=render_params.text_stroke_rgb,
                stroke_width=dense_stroke_width(),
                anchor="mm",
            )
            value_label_bboxes[str(point.point_id)] = list(value_box)
            point_annotation_bboxes[str(point.point_id)] = bbox_union([point_box, value_box])
            entities.append(
                {
                    "entity_id": str(point.point_id),
                    "entity_type": "scatter_series_point",
                    "bbox_xyxy": list(point_box),
                    "attrs": {
                        "series_label": str(series_item.label),
                        "x_label": str(point.x_label),
                        "x_index": int(point.x_index),
                        "y_value": int(point.y_value),
                        "value_label_bbox_xyxy": list(value_box),
                    },
                }
            )

    legend_bboxes: dict[str, list[float]] = {}
    legend_left = plot_bbox[2] + float(render_params.legend_gap_px)
    legend_top = plot_bbox[1] + 30.0
    legend_row_height = 52.0
    draw_text(
        draw,
        "Series",
        (legend_left, legend_top - 31.0),
        font=legend_font,
        fill=render_params.text_color_rgb,
        stroke_fill=render_params.text_stroke_rgb,
    )
    for index, series_item in enumerate(dataset.series):
        y = legend_top + float(index) * legend_row_height
        row_box = [legend_left - 10.0, y - 8.0, float(width - 54.0), y + 36.0]
        draw.rounded_rectangle(row_box, radius=6, fill=render_params.panel_fill_rgb, outline=render_params.panel_border_rgb, width=1)
        marker_box = draw_marker(
            draw,
            center=(legend_left + 15.0, y + 14.0),
            radius=10.0,
            shape=str(series_item.marker_shape),
            fill=series_item.color_rgb,
        )
        text_box = draw_text(
            draw,
            str(series_item.label),
            (legend_left + 44.0, y + 2.0),
            font=legend_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
        )
        legend_bboxes[str(series_item.label)] = bbox_union([row_box, marker_box, text_box])
        entities.append(
            {
                "entity_id": f"legend_{series_item.label}",
                "entity_type": "legend_entry",
                "bbox_xyxy": list(legend_bboxes[str(series_item.label)]),
                "attrs": {"series_label": str(series_item.label)},
            }
        )

    return RenderedScene(
        image=image,
        entities=tuple(dict(item) for item in entities),
        plot_bbox_px=bbox(plot_bbox),
        point_bboxes=dict(point_bboxes),
        value_label_bboxes=dict(value_label_bboxes),
        point_annotation_bboxes=dict(point_annotation_bboxes),
        x_label_bboxes=dict(x_label_bboxes),
        legend_bboxes=dict(legend_bboxes),
        title_bbox_px=list(title_box),
        title_text=str(title_text),
    )


def render_scatter_readout_dataset(
    *,
    dataset: SceneDataset,
    params: Mapping[str, Any],
    instance_seed: int,
) -> ScatterReadoutRenderResult:
    render_style_params = {**dict(params), "_render_style_seed": int(instance_seed)}
    resolved_params = resolve_render_params(render_style_params)
    protected_colors = [tuple(int(channel) for channel in series_item.color_rgb) for series_item in dataset.series]
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="scatter_readout",
        render_params=resolved_params,
        protected_colors=protected_colors,
    )
    chart_font_family = sample_chart_font_family(int(instance_seed), params)
    with temporary_default_font_family(str(chart_font_family)):
        rendered_scene = render_scatter_readout_scene(
            background,
            dataset=dataset,
            render_params=render_params,
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=NOISE_DEFAULTS,
    )
    return ScatterReadoutRenderResult(
        image=image,
        rendered_scene=rendered_scene,
        render_params=render_params,
        background_meta={**dict(background_meta), "information_scene_style": dict(information_style_meta)},
        post_noise_meta=dict(post_noise_meta),
        chart_font_family=str(chart_font_family),
    )


def font_assets_payload(*, chart_font_family: str) -> dict[str, Any]:
    return chart_font_asset_metadata(str(chart_font_family))


__all__ = [
    "bbox",
    "draw_marker",
    "draw_text",
    "font_assets_payload",
    "point_xy",
    "render_scatter_readout_dataset",
    "render_scatter_readout_scene",
    "resolve_render_params",
    "sample_chart_font_family",
]
