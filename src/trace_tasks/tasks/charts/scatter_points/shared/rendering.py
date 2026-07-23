"""Rendering helpers for scatter-point chart scenes."""

from __future__ import annotations
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.cartesian.axes import (
    draw_axis_lines,
    draw_horizontal_value_grid_ticks,
    draw_plot_frame,
    draw_vertical_value_grid_ticks,
)
from trace_tasks.tasks.charts.shared.cartesian.frame import plot_bbox_from_margins
from trace_tasks.tasks.charts.shared.cartesian.geometry import project_xy, round_bbox, union_bboxes as cartesian_union_bboxes
from trace_tasks.tasks.charts.shared.cartesian.lines import draw_dashed_line as draw_cartesian_dashed_line
from trace_tasks.tasks.charts.shared.cartesian.markers import draw_marker as draw_cartesian_marker
from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.render_variation import apply_layout_jitter_to_margins
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font, temporary_default_font_family
from trace_tasks.tasks.charts.shared.visual_defaults import chart_font_asset_metadata, sample_chart_font_family

from .defaults import (
    NOISE_DEFAULTS,
    RENDER_DEFAULTS,
    group_default,
    render_style_seed,
    resolve_int,
    resolve_rgb,
)
from .sampling import as_rgb
from .state import BBox, Dataset, RGB, RenderParams, RenderedScene, SCENE_NAMESPACE, ScatterPointsRenderResult


def bbox(values: Sequence[float]) -> list[float]:
    return round_bbox(values)


def bbox_union(boxes: Sequence[Sequence[float]]) -> list[float]:
    return cartesian_union_bboxes(boxes)


def resolve_render_params(params: Mapping[str, Any]) -> RenderParams:
    """Resolve only reusable scene rendering knobs, leaving objective semantics to task files."""

    margin_left = resolve_int(params, "scatter_points_plot_margin_left_px", 112)
    margin_right = resolve_int(params, "scatter_points_plot_margin_right_px", 292)
    margin_top = resolve_int(params, "scatter_points_plot_margin_top_px", 80)
    margin_bottom = resolve_int(params, "scatter_points_plot_margin_bottom_px", 118)
    margin_left, margin_right, margin_top, margin_bottom, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(margin_left),
        right_px=int(margin_right),
        top_px=int(margin_top),
        bottom_px=int(margin_bottom),
        params=params,
        defaults=RENDER_DEFAULTS,
        instance_seed=render_style_seed(params),
        namespace=SCENE_NAMESPACE,
    )
    return RenderParams(
        canvas_width=resolve_int(params, "scatter_points_canvas_width", 1280),
        canvas_height=resolve_int(params, "scatter_points_canvas_height", 820),
        plot_margin_left_px=int(margin_left),
        plot_margin_right_px=int(margin_right),
        plot_margin_top_px=int(margin_top),
        plot_margin_bottom_px=int(margin_bottom),
        axis_line_width_px=resolve_int(params, "axis_line_width_px", 2),
        grid_line_width_px=resolve_int(params, "grid_line_width_px", 1),
        tick_length_px=resolve_int(params, "tick_length_px", 8),
        point_radius_px=resolve_int(params, "scatter_points_point_radius_px", 6),
        tick_font_size_px=resolve_int(params, "scatter_points_tick_font_size_px", 16),
        label_font_size_px=resolve_int(params, "scatter_points_label_font_size_px", 19),
        legend_font_size_px=resolve_int(params, "scatter_points_legend_font_size_px", 20),
        title_font_size_px=resolve_int(params, "scatter_points_title_font_size_px", 26),
        legend_gap_px=resolve_int(params, "scatter_points_legend_gap_px", 78),
        axis_color_rgb=resolve_rgb(params, "axis_color_rgb", (34, 43, 58)),
        grid_color_rgb=resolve_rgb(params, "grid_color_rgb", (209, 216, 226)),
        text_color_rgb=resolve_rgb(params, "text_color_rgb", (20, 28, 40)),
        muted_text_rgb=resolve_rgb(params, "muted_text_rgb", (82, 93, 110)),
        text_stroke_rgb=resolve_rgb(params, "text_stroke_rgb", (255, 255, 255)),
        plot_fill_rgb=resolve_rgb(params, "plot_fill_rgb", (250, 251, 253)),
        panel_fill_rgb=resolve_rgb(params, "panel_fill_rgb", (246, 248, 252)),
        panel_border_rgb=resolve_rgb(params, "panel_border_rgb", (160, 170, 184)),
        threshold_line_rgb=resolve_rgb(params, "scatter_points_threshold_line_rgb", (31, 91, 170)),
        threshold_label_rgb=resolve_rgb(params, "scatter_points_threshold_label_rgb", (20, 60, 126)),
        layout_jitter_meta=dict(layout_jitter_meta),
    )


def data_to_pixel(x_value: float, y_value: float, plot_bbox: BBox) -> tuple[float, float]:
    return project_xy(x_value=float(x_value), y_value=float(y_value), plot_bbox=plot_bbox)


def text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: Any,
    *,
    anchor: str | None = None,
    stroke_width: int = 0,
) -> list[float]:
    box = draw.textbbox(tuple(float(value) for value in xy), str(text), font=font, anchor=anchor, stroke_width=max(0, int(stroke_width)))
    return bbox(box)


def draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: RGB,
    width: int,
    dash_px: int = 10,
    gap_px: int = 7,
) -> None:
    draw_cartesian_dashed_line(
        draw,
        (float(start[0]), float(start[1])),
        (float(end[0]), float(end[1])),
        fill=fill,
        width=int(width),
        dash_px=int(dash_px),
        gap_px=int(gap_px),
    )


def draw_marker(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    radius: float,
    shape: str,
    fill: RGB,
    outline: RGB,
) -> list[float]:
    return draw_cartesian_marker(
        draw,
        center=(float(center[0]), float(center[1])),
        radius=float(radius),
        shape=str(shape),
        fill=fill,
        outline=outline,
        width=2,
        triangle_style="up_wide",
        ring_style="inner_fill",
    )


def draw_rotated_y_label(
    image: Image.Image,
    *,
    text: str,
    font: Any,
    fill: RGB,
    position: tuple[float, float],
) -> list[float]:
    text_box = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), str(text), font=font)
    width = max(1, int(text_box[2] - text_box[0] + 10))
    height = max(1, int(text_box[3] - text_box[1] + 10))
    overlay = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.text((5 - text_box[0], 5 - text_box[1]), str(text), font=font, fill=tuple(fill) + (255,))
    rotated = overlay.rotate(90, expand=True)
    x = int(round(float(position[0]) - rotated.size[0] / 2.0))
    y = int(round(float(position[1]) - rotated.size[1] / 2.0))
    image.paste(rotated, (x, y), rotated)
    return bbox((x, y, x + rotated.size[0], y + rotated.size[1]))


def title_options(params: Mapping[str, Any]) -> tuple[str, ...]:
    raw = params.get("scatter_points_title_options", RENDER_DEFAULTS.get("scatter_points_title_options", ()))
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        titles = tuple(str(value) for value in raw if str(value).strip())
        if titles:
            return titles
    return ("Point Distribution", "Scatter Measurements", "Sample Point Map", "Observed Points")


def render_scatter_points_scene(
    image: Image.Image,
    *,
    dataset: Dataset,
    render_params: RenderParams,
    instance_seed: int,
    params: Mapping[str, Any],
) -> RenderedScene:
    """Draw the scatter scene and preserve all projected point witnesses."""

    draw = ImageDraw.Draw(image)
    width = int(render_params.canvas_width)
    height = int(render_params.canvas_height)
    panel_margin = 34
    panel_bbox = (panel_margin, panel_margin, width - panel_margin, height - panel_margin)
    plot_bbox = plot_bbox_from_margins(
        canvas_width=float(width),
        canvas_height=float(height),
        margin_left_px=float(render_params.plot_margin_left_px),
        margin_right_px=float(render_params.plot_margin_right_px),
        margin_top_px=float(render_params.plot_margin_top_px),
        margin_bottom_px=float(render_params.plot_margin_bottom_px),
    )
    draw.rounded_rectangle(panel_bbox, radius=18, fill=render_params.panel_fill_rgb, outline=render_params.panel_border_rgb, width=2)
    draw_plot_frame(draw, plot_bbox, fill=render_params.plot_fill_rgb, outline=render_params.panel_border_rgb, width=1)

    title_font = load_font(render_params.title_font_size_px, bold=False)
    label_font = load_font(render_params.label_font_size_px, bold=False)
    tick_font = load_font(render_params.tick_font_size_px, bold=False)
    legend_font = load_font(render_params.legend_font_size_px, bold=False)
    threshold_font = load_font(max(13, render_params.tick_font_size_px - 1), bold=False)
    title_rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.title")
    title_probability = float(params.get("scatter_points_title_probability", group_default(RENDER_DEFAULTS, "scatter_points_title_probability", 0.35)))
    title_text = ""
    title_bbox: list[float] = []
    if title_rng.random() < max(0.0, min(1.0, title_probability)):
        title_text = str(title_rng.choice(title_options(params)))
        title_xy = (float(plot_bbox[0]), 34.0)
        draw_text_traced(
            draw,
            title_xy,
            title_text,
            font=title_font,
            fill=render_params.text_color_rgb,
            role="readout",
            required=False,
        )
        title_bbox = text_bbox(draw, title_xy, title_text, title_font)

    x0, y0, x1, y1 = plot_bbox
    tick_values = [0, 20, 40, 60, 80, 100]
    grid_values = [20, 40, 60, 80]
    x_tick_positions = draw_vertical_value_grid_ticks(
        draw,
        plot_bbox,
        tick_values=tick_values,
        domain_min=0,
        domain_max=100,
        grid_rgb=render_params.grid_color_rgb,
        axis_rgb=render_params.axis_color_rgb,
        grid_width_px=render_params.grid_line_width_px,
        tick_width_px=render_params.axis_line_width_px,
        tick_length_px=float(render_params.tick_length_px),
        grid_values=grid_values,
    )
    y_tick_positions = draw_horizontal_value_grid_ticks(
        draw,
        plot_bbox,
        tick_values=tick_values,
        domain_min=0,
        domain_max=100,
        grid_rgb=render_params.grid_color_rgb,
        axis_rgb=render_params.axis_color_rgb,
        grid_width_px=render_params.grid_line_width_px,
        tick_width_px=render_params.axis_line_width_px,
        tick_length_px=float(render_params.tick_length_px),
        grid_values=grid_values,
    )
    for tick in tick_values:
        px = float(x_tick_positions[float(tick)])
        py = float(y_tick_positions[float(tick)])
        draw_text_traced(
            draw,
            (px, y1 + render_params.tick_length_px + 4),
            str(tick),
            font=tick_font,
            fill=render_params.muted_text_rgb,
            anchor="mt",
            role="readout",
            required=False,
        )
        draw_text_traced(
            draw,
            (x0 - render_params.tick_length_px - 8, py),
            str(tick),
            font=tick_font,
            fill=render_params.muted_text_rgb,
            anchor="rm",
            role="readout",
            required=False,
        )

    draw_axis_lines(
        draw,
        plot_bbox,
        axis_rgb=render_params.axis_color_rgb,
        axis_width_px=render_params.axis_line_width_px + 1,
    )

    x_axis_label = str(params.get("scatter_points_x_axis_label", group_default(RENDER_DEFAULTS, "scatter_points_x_axis_label", "X value")))
    y_axis_label = str(params.get("scatter_points_y_axis_label", group_default(RENDER_DEFAULTS, "scatter_points_y_axis_label", "Y value")))
    x_label_xy = ((x0 + x1) / 2.0, height - 47.0)
    draw_text_traced(
        draw,
        x_label_xy,
        x_axis_label,
        font=label_font,
        fill=render_params.text_color_rgb,
        anchor="mm",
        role="readout",
        required=False,
    )
    x_axis_label_bbox = text_bbox(draw, x_label_xy, x_axis_label, label_font, anchor="mm")
    y_axis_label_bbox = draw_rotated_y_label(
        image,
        text=y_axis_label,
        font=label_font,
        fill=render_params.text_color_rgb,
        position=(45.0, (y0 + y1) / 2.0),
    )

    threshold_guide_bbox: list[float] = []
    threshold_axis = dataset.query.trace.get("threshold_axis")
    threshold_value = dataset.query.trace.get("threshold_value")
    if threshold_axis in {"x", "y"} and threshold_value is not None:
        value = float(threshold_value)
        if str(threshold_axis) == "x":
            px = x0 + (value / 100.0) * (x1 - x0)
            draw_dashed_line(draw, (px, y0), (px, y1), fill=render_params.threshold_line_rgb, width=2)
            label_xy = (px + 8.0, y0 + 12.0)
            text = f"x = {int(value)}"
            draw_text_traced(
                draw,
                label_xy,
                text,
                font=threshold_font,
                fill=render_params.threshold_label_rgb,
                stroke_width=0,
                role="readout",
                required=False,
            )
            label_box = text_bbox(draw, label_xy, text, threshold_font, stroke_width=0)
            threshold_guide_bbox = bbox_union([(px - 3, y0, px + 3, y1), label_box])
        else:
            py = y1 - (value / 100.0) * (y1 - y0)
            draw_dashed_line(draw, (x0, py), (x1, py), fill=render_params.threshold_line_rgb, width=2)
            label_xy = (x1 - 70.0, py - 24.0)
            text = f"y = {int(value)}"
            draw_text_traced(
                draw,
                label_xy,
                text,
                font=threshold_font,
                fill=render_params.threshold_label_rgb,
                stroke_width=0,
                role="readout",
                required=False,
            )
            label_box = text_bbox(draw, label_xy, text, threshold_font, stroke_width=0)
            threshold_guide_bbox = bbox_union([(x0, py - 3, x1, py + 3), label_box])

    entities: list[dict[str, Any]] = [
        {"entity_id": "plot_area", "entity_type": "plot_area", "bbox_xyxy": bbox(plot_bbox), "attrs": {}},
    ]
    point_bboxes: dict[str, list[float]] = {}
    point_centers: dict[str, list[float]] = {}
    outline_rgb = (255, 255, 255)
    for point in dataset.points:
        center = data_to_pixel(float(point.x_value), float(point.y_value), plot_bbox)
        point_box = draw_marker(
            draw,
            center=center,
            radius=float(render_params.point_radius_px),
            shape=str(point.marker_shape),
            fill=point.color_rgb,
            outline=outline_rgb,
        )
        point_bboxes[str(point.point_id)] = list(point_box)
        point_centers[str(point.point_id)] = [round(float(center[0]), 3), round(float(center[1]), 3)]
        entities.append(
            {
                "entity_id": str(point.point_id),
                "entity_type": "scatter_point",
                "bbox_xyxy": list(point_box),
                "attrs": {
                    "x_value": round(float(point.x_value), 3),
                    "y_value": round(float(point.y_value), 3),
                    "category_label": str(point.category_label),
                    "marker_shape": str(point.marker_shape),
                },
            }
        )

    legend_bboxes: dict[str, list[float]] = {}
    if dataset.categories:
        legend_x = x1 + float(render_params.legend_gap_px)
        legend_y = y0 + 28.0
        row_gap = max(32.0, float(render_params.legend_font_size_px + 16))
        draw_text_traced(draw, (legend_x, legend_y - 34.0), "Legend", font=legend_font, fill=render_params.text_color_rgb, role="readout", required=False)
        for category_index, category in enumerate(dataset.categories):
            cy = legend_y + float(category_index) * row_gap
            marker_box = draw_marker(
                draw,
                center=(legend_x + 12.0, cy + 8.0),
                radius=max(6.0, float(render_params.point_radius_px)),
                shape=str(category.marker_shape),
                fill=category.color_rgb,
                outline=outline_rgb,
            )
            text_xy = (legend_x + 34.0, cy)
            draw_text_traced(
                draw,
                text_xy,
                str(category.label),
                font=legend_font,
                fill=render_params.text_color_rgb,
                role="readout",
                required=False,
            )
            text_box = text_bbox(draw, text_xy, str(category.label), legend_font)
            row_box = bbox_union([marker_box, text_box])
            legend_bboxes[str(category.label)] = list(row_box)
            entities.append(
                {
                    "entity_id": f"legend_{category.label}",
                    "entity_type": "legend_entry",
                    "bbox_xyxy": list(row_box),
                    "attrs": {"category_label": str(category.label)},
                }
            )

    return RenderedScene(
        image=image,
        entities=tuple(dict(entity) for entity in entities),
        plot_bbox_px=bbox(plot_bbox),
        panel_bbox_px=bbox(panel_bbox),
        point_bboxes=dict(point_bboxes),
        point_centers=dict(point_centers),
        legend_bboxes=dict(legend_bboxes),
        threshold_guide_bbox_px=list(threshold_guide_bbox),
        title_bbox_px=list(title_bbox),
        x_axis_label_bbox_px=list(x_axis_label_bbox),
        y_axis_label_bbox_px=list(y_axis_label_bbox),
        title_text=str(title_text),
    )


def render_scatter_points_dataset(
    *,
    dataset: Dataset,
    params: Mapping[str, Any],
    instance_seed: int,
) -> ScatterPointsRenderResult:
    """Apply background/font/noise wrappers around the identity-free scatter renderer."""

    render_style_params = {**dict(params), "_render_style_seed": int(instance_seed)}
    resolved_params = resolve_render_params(render_style_params)
    protected_colors = [tuple(int(channel) for channel in category.color_rgb) for category in dataset.categories]
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="scatter_points",
        render_params=resolved_params,
        protected_colors=protected_colors,
    )
    chart_font_family = sample_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=SCENE_NAMESPACE,
        params=params,
    )
    with temporary_default_font_family(str(chart_font_family)):
        rendered_scene = render_scatter_points_scene(
            background,
            dataset=dataset,
            render_params=render_params,
            instance_seed=int(instance_seed),
            params=params,
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=NOISE_DEFAULTS,
    )
    return ScatterPointsRenderResult(
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
    "as_rgb",
    "bbox",
    "bbox_union",
    "data_to_pixel",
    "draw_marker",
    "font_assets_payload",
    "render_scatter_points_dataset",
    "resolve_render_params",
]
