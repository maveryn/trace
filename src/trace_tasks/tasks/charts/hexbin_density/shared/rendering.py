"""Rendering primitives for hexbin-density charts."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.shared.cartesian.axes import (
    draw_axis_lines,
    draw_horizontal_value_grid_ticks,
    draw_plot_frame,
    draw_vertical_value_grid_ticks,
)
from trace_tasks.tasks.charts.shared.cartesian.frame import plot_bbox_from_margins
from trace_tasks.tasks.charts.shared.cartesian.geometry import project_xy, round_bbox
from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font
from trace_tasks.tasks.charts.hexbin_density.shared.defaults import (
    AXIS_LABELS,
    BBox,
    POST_IMAGE_NOISE_DEFAULTS,
    RGB,
    SCENE_NAMESPACE,
    jittered_margins,
    render_int,
    render_rgb,
)
from trace_tasks.tasks.charts.hexbin_density.shared.state import HexbinDataset, RenderParams, RenderedHexbinScene


def bbox(values: Sequence[float]) -> List[float]:
    return round_bbox(values)


def resolve_render_params(params: Mapping[str, Any], *, instance_seed: int) -> RenderParams:
    params = {**dict(params), "_render_style_seed": int(instance_seed)}
    left, right, top, bottom, jitter = jittered_margins(params, instance_seed=int(instance_seed))
    return RenderParams(
        canvas_width=render_int(params, "canvas_width", 1280),
        canvas_height=render_int(params, "canvas_height", 820),
        margin_left=int(left),
        margin_right=int(right),
        margin_top=int(top),
        margin_bottom=int(bottom),
        legend_width=render_int(params, "legend_width_px", 150),
        axis_line_width=render_int(params, "axis_line_width_px", 2),
        grid_line_width=render_int(params, "grid_line_width_px", 1),
        hex_outline_width=render_int(params, "hex_outline_width_px", 2),
        tick_font_size=render_int(params, "tick_font_size_px", 15),
        label_font_size=render_int(params, "label_font_size_px", 17),
        title_font_size=render_int(params, "title_font_size_px", 26),
        plot_fill_rgb=render_rgb(params, "plot_fill_rgb", (255, 255, 255)),
        axis_rgb=render_rgb(params, "axis_color_rgb", (55, 60, 70)),
        grid_rgb=render_rgb(params, "grid_color_rgb", (223, 227, 235)),
        text_rgb=render_rgb(params, "text_color_rgb", (35, 40, 52)),
        muted_rgb=render_rgb(params, "muted_text_rgb", (83, 91, 105)),
        hex_outline_rgb=render_rgb(params, "hex_outline_rgb", (255, 255, 255)),
        threshold_guide_fill_rgb=render_rgb(params, "threshold_guide_fill_rgb", (255, 255, 255)),
        layout_jitter=dict(jitter),
    )


def plot_bbox(render_params: RenderParams) -> BBox:
    return tuple(
        plot_bbox_from_margins(
            canvas_width=float(render_params.canvas_width),
            canvas_height=float(render_params.canvas_height),
            margin_left_px=float(render_params.margin_left),
            margin_right_px=float(render_params.margin_right),
            margin_top_px=float(render_params.margin_top),
            margin_bottom_px=float(render_params.margin_bottom),
        )
    )


def legend_bbox(plot_box: BBox, render_params: RenderParams) -> BBox:
    _x0, y0, x1, y1 = (float(value) for value in plot_box)
    return (
        float(x1 + 34.0),
        float(y0 + 58.0),
        float(min(render_params.canvas_width - 34, x1 + 34.0 + render_params.legend_width)),
        float(min(y1, y0 + 286.0)),
    )


def scale_point(x_value: float, y_value: float, *, plot_box: BBox) -> Tuple[float, float]:
    return project_xy(
        x_value=float(x_value),
        y_value=float(y_value),
        plot_bbox=plot_box,
        x_min=0.0,
        x_max=100.0,
        y_min=0.0,
        y_max=100.0,
    )


def draw_axes(draw: ImageDraw.ImageDraw, *, plot_box: BBox, render_params: RenderParams) -> Dict[str, Any]:
    x0, y0, x1, y1 = (float(value) for value in plot_box)
    draw_plot_frame(draw, plot_box, fill=render_params.plot_fill_rgb, outline=render_params.grid_rgb, width=1)
    tick_font = load_font(int(render_params.tick_font_size), bold=False)
    tick_values = (0, 25, 50, 75, 100)
    x_tick_positions = draw_vertical_value_grid_ticks(
        draw,
        plot_box,
        tick_values=tick_values,
        domain_min=0,
        domain_max=100,
        grid_rgb=render_params.grid_rgb,
        axis_rgb=render_params.axis_rgb,
        grid_width_px=max(1, int(render_params.grid_line_width)),
        tick_width_px=1,
        tick_length_px=0.0,
    )
    y_tick_positions = draw_horizontal_value_grid_ticks(
        draw,
        plot_box,
        tick_values=tick_values,
        domain_min=0,
        domain_max=100,
        grid_rgb=render_params.grid_rgb,
        axis_rgb=render_params.axis_rgb,
        grid_width_px=max(1, int(render_params.grid_line_width)),
        tick_width_px=1,
        tick_length_px=0.0,
    )
    for tick in tick_values:
        sx = float(x_tick_positions[float(tick)])
        sy = float(y_tick_positions[float(tick)])
        draw_text_traced(draw, (sx - 8.0, y1 + 10.0), str(tick), font=tick_font, fill=render_params.muted_rgb, role="readout", required=False)
        draw_text_traced(draw, (x0 - 36.0, sy - 8.0), str(tick), font=tick_font, fill=render_params.muted_rgb, role="readout", required=False)
    draw_axis_lines(draw, plot_box, axis_rgb=render_params.axis_rgb, axis_width_px=max(1, int(render_params.axis_line_width)))
    return {"axis_ticks": [0, 25, 50, 75, 100]}


def hex_points(center_x: float, center_y: float, radius: float) -> List[Tuple[float, float]]:
    return [
        (
            float(center_x) + float(radius) * math.cos(math.radians(float(angle))),
            float(center_y) + float(radius) * math.sin(math.radians(float(angle))),
        )
        for angle in (0, 60, 120, 180, 240, 300)
    ]


def hex_bbox(points: Sequence[Tuple[float, float]]) -> List[float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return bbox([min(xs), min(ys), max(xs), max(ys)])


def hex_layout(dataset: HexbinDataset, *, plot_box: BBox) -> Tuple[float, Dict[Tuple[int, int], Tuple[float, float]]]:
    x0, y0, x1, y1 = (float(value) for value in plot_box)
    inner_pad = 36.0
    usable_w = max(10.0, (x1 - x0) - (2.0 * inner_pad))
    usable_h = max(10.0, (y1 - y0) - (2.0 * inner_pad))
    columns = int(dataset.column_count)
    rows = int(dataset.row_count)
    radius_by_width = usable_w / max(1.0, 1.5 * float(columns - 1) + 2.0)
    radius_by_height = usable_h / max(1.0, math.sqrt(3.0) * (float(rows) + 0.5))
    radius = max(8.0, min(float(radius_by_width), float(radius_by_height)))
    span_w = float(radius) * (1.5 * float(columns - 1) + 2.0)
    span_h = float(radius) * math.sqrt(3.0) * (float(rows) + 0.5)
    origin_x = x0 + ((x1 - x0) - span_w) / 2.0 + float(radius)
    origin_y = y0 + ((y1 - y0) - span_h) / 2.0 + (math.sqrt(3.0) * float(radius) / 2.0)
    centers: Dict[Tuple[int, int], Tuple[float, float]] = {}
    for row in range(rows):
        for col in range(columns):
            centers[(int(row), int(col))] = (
                float(origin_x + (1.5 * float(radius) * float(col))),
                float(origin_y + (math.sqrt(3.0) * float(radius) * float(row)) + ((math.sqrt(3.0) * float(radius) / 2.0) if col % 2 else 0.0)),
            )
    return float(radius), centers


def draw_hex_bins(
    draw: ImageDraw.ImageDraw,
    *,
    dataset: HexbinDataset,
    plot_box: BBox,
    render_params: RenderParams,
) -> Tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    radius, centers = hex_layout(dataset, plot_box=plot_box)
    bin_bboxes: Dict[str, List[float]] = {}
    bin_centers: Dict[str, List[float]] = {}
    for bin_item in dataset.bins:
        center = centers[(int(bin_item.row_index), int(bin_item.column_index))]
        points = hex_points(float(center[0]), float(center[1]), float(radius) * 0.94)
        draw.polygon(points, fill=bin_item.fill_rgb, outline=render_params.hex_outline_rgb)
        if int(render_params.hex_outline_width) > 1:
            draw.line([*points, points[0]], fill=render_params.hex_outline_rgb, width=max(1, int(render_params.hex_outline_width)))
        box = hex_bbox(points)
        bin_bboxes[str(bin_item.bin_id)] = list(box)
        bin_centers[str(bin_item.bin_id)] = bbox([float(center[0]), float(center[1])])
    return bin_bboxes, bin_centers


def draw_legend(draw: ImageDraw.ImageDraw, *, legend_box: BBox, palette: Sequence[RGB], render_params: RenderParams) -> None:
    x0, y0, x1, _y1 = (float(value) for value in legend_box)
    title_font = load_font(int(render_params.label_font_size), bold=True)
    label_font = load_font(int(render_params.label_font_size), bold=False)
    draw_text_traced(draw, (x0, y0), "Density level", font=title_font, fill=render_params.text_rgb, role="readout", required=False)
    swatch = 22
    y = y0 + 36.0
    for index, color in enumerate(palette[:5], start=1):
        draw.rectangle([x0, y, x0 + swatch, y + swatch], fill=color, outline=render_params.grid_rgb, width=1)
        draw_text_traced(
            draw,
            (x0 + swatch + 12.0, y + 2.0),
            str(index),
            font=label_font,
            fill=render_params.text_rgb,
            role="readout",
            required=False,
        )
        y += 32.0
    draw.rectangle([x0 - 12.0, y0 - 14.0, x1, y + 4.0], outline=render_params.grid_rgb, width=1)


def draw_threshold_guide(draw: ImageDraw.ImageDraw, *, dataset: HexbinDataset, plot_box: BBox, render_params: RenderParams) -> Tuple[List[float], str]:
    label = f"Count density {dataset.query.threshold_operator} {dataset.query.threshold_level}"
    font = load_font(int(render_params.label_font_size), bold=True)
    x0, y0, x1, _ = (float(value) for value in plot_box)
    text_bbox = draw.textbbox((0, 0), label, font=font)
    width = float(text_bbox[2] - text_bbox[0]) + 24.0
    height = float(text_bbox[3] - text_bbox[1]) + 16.0
    bx1 = x1 - 10.0
    bx0 = max(x0 + 10.0, bx1 - width)
    by0 = y0 + 10.0
    by1 = by0 + height
    draw.rounded_rectangle([bx0, by0, bx1, by1], radius=8, fill=render_params.threshold_guide_fill_rgb, outline=render_params.grid_rgb, width=1)
    draw_text_traced(draw, (bx0 + 12.0, by0 + 7.0), label, font=font, fill=render_params.text_rgb, role="readout", required=False)
    return bbox([bx0, by0, bx1, by1]), str(label)


def render_dataset(dataset: HexbinDataset, *, params: Mapping[str, Any], instance_seed: int) -> RenderedHexbinScene:
    """Render one sampled density field and retain projected hex-bin witnesses."""

    resolved_params = resolve_render_params(params, instance_seed=int(instance_seed))
    render_params, image, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="hexbin_density",
        render_params=resolved_params,
        protected_colors=dataset.density_palette_rgb,
    )
    draw = ImageDraw.Draw(image)
    plot_box = plot_bbox(render_params)
    axes_meta = draw_axes(draw, plot_box=plot_box, render_params=render_params)
    axis_rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.axis_labels")
    x_axis, y_axis = AXIS_LABELS[axis_rng.randrange(len(AXIS_LABELS))]
    label_font = load_font(int(render_params.label_font_size), bold=False)
    x_label_bbox = draw.textbbox((0, 0), x_axis, font=label_font)
    x_label_xy = (
        float((plot_box[0] + plot_box[2]) / 2.0) - float(x_label_bbox[2] - x_label_bbox[0]) / 2.0,
        float(plot_box[3]) + 45.0,
    )
    draw_text_traced(draw, x_label_xy, x_axis, font=label_font, fill=render_params.muted_rgb, role="readout", required=False)
    y_label_xy = (float(plot_box[0]) - 76.0, float(plot_box[1]) - 28.0)
    draw_text_traced(draw, y_label_xy, y_axis, font=label_font, fill=render_params.muted_rgb, role="readout", required=False)
    bin_bboxes, bin_centers = draw_hex_bins(draw, dataset=dataset, plot_box=plot_box, render_params=render_params)
    legend_box = legend_bbox(plot_box, render_params)
    draw_legend(draw, legend_box=legend_box, palette=dataset.density_palette_rgb, render_params=render_params)
    threshold_box, threshold_text = draw_threshold_guide(draw, dataset=dataset, plot_box=plot_box, render_params=render_params)
    image, noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    title_bbox: List[float] = []
    entities: list[Dict[str, Any]] = [
        {"entity_id": "plot_area", "entity_type": "plot", "bbox_px": bbox(plot_box)},
        {"entity_id": "legend", "entity_type": "legend", "bbox_px": bbox(legend_box)},
        {"entity_id": "threshold_guide", "entity_type": "annotation", "bbox_px": threshold_box, "text": threshold_text},
    ]
    for bin_item in dataset.bins:
        entities.append(
            {
                "entity_id": str(bin_item.bin_id),
                "entity_type": "hex_bin",
                "row_index": int(bin_item.row_index),
                "column_index": int(bin_item.column_index),
                "density_level": int(bin_item.density_level),
                "bbox_px": list(bin_bboxes[str(bin_item.bin_id)]),
                "center_px": list(bin_centers[str(bin_item.bin_id)]),
            }
        )
    return RenderedHexbinScene(
        image=image,
        entities=tuple(entities),
        plot_bbox_px=bbox(plot_box),
        legend_bbox_px=bbox(legend_box),
        title_bbox_px=title_bbox,
        threshold_guide_bbox_px=threshold_box,
        bin_bboxes_px=dict(bin_bboxes),
        bin_centers_px=dict(bin_centers),
        render_meta={
            "background_style": {**dict(background_meta), "information_scene_style": dict(information_style_meta)},
            "information_scene_style": dict(information_style_meta),
            "post_image_noise": dict(noise_meta),
            "layout_jitter": dict(render_params.layout_jitter),
            "axis_labels": {"x": str(x_axis), "y": str(y_axis)},
            "title_text": "",
            "density_palette_scheme": str(dataset.density_palette_scheme),
            "density_palette_rgb": [list(color) for color in dataset.density_palette_rgb],
            "density_palette_contrast_policy": dict(dataset.density_palette_trace),
            **dict(axes_meta),
        },
    )


__all__ = [
    "bbox",
    "render_dataset",
]
