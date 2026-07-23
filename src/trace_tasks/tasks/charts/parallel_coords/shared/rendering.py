"""Rendering helpers for the parallel-coordinates chart scene."""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw

from trace_tasks.tasks.charts.shared.dense_text import dense_fit_bold, dense_stroke_width
from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from .....core.visual.noise import apply_post_image_noise
from ....shared.bbox_projection import bbox_union, round_bbox
from ....shared.font_assets import font_asset_version
from ....shared.text_legibility import draw_text_traced
from ....shared.text_rendering import load_font, temporary_default_font_family
from .defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    resolve_render_params,
    sample_chart_font_family,
)
from .state import BBox, ParallelDataset, ParallelRenderParams, ParallelRenderResult, RenderedParallelScene, RGB


def _text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: Any,
    *,
    stroke_width: int = 0,
) -> BBox:
    box = draw.textbbox(tuple(float(value) for value in xy), str(text), font=font, stroke_width=max(0, int(stroke_width)))
    return round_bbox([box[0], box[1], box[2], box[3]])


def _draw_centered_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: Any,
    *,
    fill: RGB,
    stroke_fill: RGB,
    stroke_width: int = 0,
) -> BBox:
    box = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width)))
    width = float(box[2] - box[0])
    height = float(box[3] - box[1])
    x = float(xy[0]) - width / 2.0
    y = float(xy[1]) - height / 2.0
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
    return round_bbox([x, y, x + width, y + height])


def _render_parallel_chart(
    background: Image.Image,
    *,
    dataset: ParallelDataset,
    render_params: ParallelRenderParams,
) -> RenderedParallelScene:
    """Draw one parallel-coordinates chart and record all reusable projections."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = image.size
    panel_margin = 34
    left = float(render_params.plot_margin_left_px)
    right = float(width - int(render_params.plot_margin_right_px))
    top = float(render_params.plot_margin_top_px)
    bottom = float(height - int(render_params.plot_margin_bottom_px))
    plot_bbox = round_bbox([left, top, right, bottom])
    panel_bbox = [panel_margin, 36, width - panel_margin, height - 42]
    draw.rounded_rectangle(
        panel_bbox,
        radius=8,
        fill=render_params.panel_fill_rgb,
        outline=render_params.panel_border_rgb,
        width=1,
    )
    draw.rectangle([left, top, right, bottom], fill=render_params.plot_fill_rgb)

    label_font = load_font(render_params.label_font_size_px, bold=False)
    endpoint_label_font = load_font(max(12, int(render_params.label_font_size_px) - 3), bold=dense_fit_bold())
    tick_font = load_font(render_params.tick_font_size_px, bold=False)
    threshold_font = load_font(render_params.threshold_font_size_px, bold=False)

    def y_px(value: float) -> float:
        ratio = (float(value) - float(dataset.value_min)) / max(1.0, float(dataset.value_max) - float(dataset.value_min))
        return bottom - (ratio * (bottom - top))

    axis_count = len(dataset.metrics)
    axis_x: dict[int, float] = {}
    for axis_index in range(axis_count):
        x = left + (float(axis_index) * (right - left) / max(1, axis_count - 1))
        axis_x[int(axis_index)] = float(x)

    tick_values = [
        int(dataset.value_min),
        int(round((int(dataset.value_min) + int(dataset.value_max)) / 2.0)),
        int(dataset.value_max),
    ]
    for tick in tick_values:
        y = y_px(float(tick))
        draw.line([left, y, right, y], fill=render_params.grid_rgb, width=render_params.grid_line_width_px)
        draw_text_traced(
            draw,
            (left - 46, y - 9),
            str(int(tick)),
            font=tick_font,
            fill=render_params.muted_text_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=dense_stroke_width(),
            role="readout",
            required=False,
        )

    selected_axes = {int(dataset.query.axis_i), int(dataset.query.axis_j)}
    threshold_bboxes: dict[int, BBox] = {}
    for axis_index, metric in enumerate(dataset.metrics):
        x = axis_x[int(axis_index)]
        selected = int(axis_index) in selected_axes
        axis_rgb = render_params.selected_axis_rgb if selected else render_params.axis_rgb
        axis_width = render_params.selected_axis_line_width_px if selected else render_params.axis_line_width_px
        draw.line([x, top, x, bottom], fill=axis_rgb, width=axis_width)
        _draw_centered_text(
            draw,
            (x, bottom + 34),
            str(metric),
            label_font,
            fill=render_params.text_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=dense_stroke_width(),
        )
        for tick in tick_values:
            y = y_px(float(tick))
            draw.line([x - 5, y, x + 5, y], fill=axis_rgb, width=2)
        if dataset.query.threshold is not None and selected:
            y = y_px(float(dataset.query.threshold))
            draw.line([x - 22, y, x + 22, y], fill=render_params.threshold_rgb, width=3)
            text_bbox = _text_bbox(draw, (x + 26, y - 9), str(dataset.query.threshold), threshold_font, stroke_width=dense_stroke_width())
            draw_text_traced(
                draw,
                (x + 26, y - 9),
                str(dataset.query.threshold),
                font=threshold_font,
                fill=render_params.threshold_rgb,
                stroke_fill=render_params.text_stroke_rgb,
                stroke_width=dense_stroke_width(),
                role="readout",
                required=False,
            )
            threshold_bboxes[int(axis_index)] = bbox_union([[x - 22, y - 2, x + 22, y + 2], text_bbox], padding=2)

    point_bboxes: dict[str, BBox] = {}
    segment_bboxes: dict[str, BBox] = {}
    profile_bboxes: dict[str, BBox] = {}
    label_bboxes: dict[str, BBox] = {}
    entities: list[dict[str, Any]] = []
    radius = float(render_params.point_radius_px)

    for profile in dataset.profiles:
        points = [(axis_x[index], y_px(profile.values[index])) for index in range(axis_count)]
        line_rgb = tuple(int(channel) for channel in profile.color_rgb)
        draw.line(points, fill=line_rgb, width=render_params.line_width_px, joint="curve")
        profile_segment_boxes: list[BBox] = []
        for axis_index in range(axis_count):
            x, y = points[int(axis_index)]
            point_box = round_bbox([x - radius, y - radius, x + radius, y + radius])
            point_bboxes[f"{profile.profile_id}:axis_{axis_index}"] = point_box
            draw.ellipse(point_box, fill=line_rgb, outline=(255, 255, 255), width=2)
        for axis_index in range(axis_count - 1):
            x0, y0 = points[int(axis_index)]
            x1, y1 = points[int(axis_index) + 1]
            seg_box = round_bbox([min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1)])
            padded = bbox_union([seg_box], padding=8)
            segment_bboxes[f"{profile.profile_id}:axis_{axis_index}_{axis_index + 1}"] = padded
            profile_segment_boxes.append(padded)
        endpoint_box = draw.textbbox((0, 0), str(profile.label), font=endpoint_label_font, stroke_width=dense_stroke_width())
        endpoint_width = float(endpoint_box[2] - endpoint_box[0])
        endpoint_height = float(endpoint_box[3] - endpoint_box[1])
        left_label_right = float(left) - 58.0
        right_label_left = float(right) + 58.0
        label_positions = (
            ("left", (max(panel_margin + 6.0, left_label_right - endpoint_width), points[0][1] - endpoint_height / 2.0)),
            ("right", (right_label_left, points[-1][1] - endpoint_height / 2.0)),
        )
        for suffix, xy in label_positions:
            box = _text_bbox(draw, xy, profile.label, endpoint_label_font, stroke_width=dense_stroke_width())
            draw_text_traced(
                draw,
                xy,
                profile.label,
                font=endpoint_label_font,
                fill=line_rgb,
                stroke_fill=render_params.text_stroke_rgb,
                stroke_width=dense_stroke_width(),
                role="readout",
                required=False,
            )
            label_bboxes[f"{profile.profile_id}:{suffix}"] = box
        profile_bboxes[profile.profile_id] = bbox_union(
            profile_segment_boxes
            + [point_bboxes[f"{profile.profile_id}:axis_{index}"] for index in range(axis_count)]
            + [label_bboxes[f"{profile.profile_id}:left"], label_bboxes[f"{profile.profile_id}:right"]],
            padding=4,
        )
        entities.append(
            {
                "entity_id": profile.profile_id,
                "entity_type": "parallel_coordinate_profile",
                "label": profile.label,
                "values": [int(value) for value in profile.values],
                "color_rgb": list(line_rgb),
                "bbox_px": list(profile_bboxes[profile.profile_id]),
            }
        )

    return RenderedParallelScene(
        image=image,
        entities=tuple(entities),
        plot_bbox_px=list(plot_bbox),
        axis_x_px={int(key): float(value) for key, value in axis_x.items()},
        point_bboxes_px=dict(point_bboxes),
        segment_bboxes_px=dict(segment_bboxes),
        profile_bboxes_px=dict(profile_bboxes),
        label_bboxes_px=dict(label_bboxes),
        threshold_bboxes_px=dict(threshold_bboxes),
    )


def render_dataset(*, dataset: ParallelDataset, params: dict[str, Any], instance_seed: int) -> ParallelRenderResult:
    """Render the dataset on a sampled chart background."""

    resolved_params = resolve_render_params({**dict(params), "_render_style_seed": int(instance_seed)})
    protected_colors = [tuple(int(channel) for channel in profile.color_rgb) for profile in dataset.profiles]
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="parallel_coords",
        render_params=resolved_params,
        protected_colors=protected_colors,
    )
    chart_font_family = sample_chart_font_family(int(instance_seed), params)
    with temporary_default_font_family(str(chart_font_family)):
        rendered = _render_parallel_chart(background, dataset=dataset, render_params=render_params)
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return ParallelRenderResult(
        image=image,
        rendered_scene=rendered,
        render_params=render_params,
        background_meta={**dict(background_meta), "information_scene_style": dict(information_style_meta)},
        post_noise_meta=dict(post_noise_meta),
        chart_font_family=str(chart_font_family),
    )


def render_spec_payload(result: ParallelRenderResult, dataset: ParallelDataset) -> dict[str, Any]:
    rendered = result.rendered_scene
    render_params = result.render_params
    return {
        "scene_variant": str(dataset.scene_variant),
        "canvas_width": int(render_params.canvas_width),
        "canvas_height": int(render_params.canvas_height),
        "coord_space": "pixel",
        "plot_bbox_px": list(rendered.plot_bbox_px),
        "axis_x_px": {str(key): float(value) for key, value in rendered.axis_x_px.items()},
        "line_width_px": int(render_params.line_width_px),
        "point_radius_px": int(render_params.point_radius_px),
        "layout_jitter": dict(render_params.layout_jitter_meta),
        "font_assets": {
            "font_asset_version": font_asset_version(),
            "chart_font_family": str(result.chart_font_family),
        },
        "background_style": dict(result.background_meta),
        "information_scene_style": dict(result.background_meta.get("information_scene_style", {})),
        "post_image_noise": dict(result.post_noise_meta),
    }


def render_map_payload(result: ParallelRenderResult) -> dict[str, Any]:
    rendered = result.rendered_scene
    return {
        "image_id": "img0",
        "plot_bbox_px": list(rendered.plot_bbox_px),
        "axis_x_px": {str(key): float(value) for key, value in rendered.axis_x_px.items()},
        "point_bboxes_px": dict(rendered.point_bboxes_px),
        "segment_bboxes_px": dict(rendered.segment_bboxes_px),
        "profile_bboxes_px": dict(rendered.profile_bboxes_px),
        "label_bboxes_px": dict(rendered.label_bboxes_px),
        "threshold_bboxes_px": {str(key): list(value) for key, value in rendered.threshold_bboxes_px.items()},
    }


__all__ = [
    "render_dataset",
    "render_map_payload",
    "render_spec_payload",
]
