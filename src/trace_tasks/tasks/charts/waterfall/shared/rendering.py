"""Rendering primitives for waterfall chart scenes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from PIL import Image, ImageDraw

from trace_tasks.tasks.charts.shared.cartesian.axes import draw_axis_lines, draw_horizontal_value_grid_ticks
from trace_tasks.tasks.charts.shared.cartesian.geometry import project_linear_inverted
from trace_tasks.tasks.charts.shared.information_style import (
    make_chart_information_background,
    resolve_chart_information_style,
)
from trace_tasks.tasks.charts.shared.dense_text import dense_fit_bold, dense_stroke_width, dense_text_style_meta
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.shared.visual_defaults import sample_chart_font_family
from trace_tasks.tasks.charts.waterfall.shared.defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    RENDERING_DEFAULTS,
    SCENE_NAMESPACE,
    rendering_default,
)
from trace_tasks.tasks.charts.waterfall.shared.state import (
    BBox,
    RGB,
    RenderedWaterfall,
    WaterfallDataset,
    WaterfallRenderArtifacts,
    WaterfallRenderParams,
)
from trace_tasks.tasks.shared.bbox_projection import round_bbox as _bbox
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.render_variation import apply_layout_jitter_to_margins, resolve_render_rgb
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import draw_text_centered, load_font, temporary_default_font_family


def _text_bbox_at(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: tuple[float, float],
    font: Any,
    stroke_width: int = 1,
) -> BBox:
    raw = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width)))
    width = float(raw[2] - raw[0])
    height = float(raw[3] - raw[1])
    cx, cy = float(center[0]), float(center[1])
    return _bbox([cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0])


def _render_int(params: Mapping[str, Any], key: str, fallback: int) -> int:
    return int(rendering_default(params, str(key), int(fallback)))


def _render_float(params: Mapping[str, Any], key: str, fallback: float) -> float:
    return float(rendering_default(params, str(key), float(fallback)))


def _render_rgb(
    params: Mapping[str, Any],
    key: str,
    fallback: RGB,
    *,
    instance_seed: int,
) -> RGB:
    return resolve_render_rgb(
        params,
        RENDERING_DEFAULTS,
        str(key),
        tuple(int(value) for value in fallback),
        instance_seed=int(instance_seed),
        namespace=SCENE_NAMESPACE,
    )


def resolve_render_params(params: Mapping[str, Any], *, instance_seed: int) -> WaterfallRenderParams:
    """Resolve one deterministic set of waterfall render parameters."""

    margin_left = _render_int(params, "plot_margin_left_px", 82)
    margin_right = _render_int(params, "plot_margin_right_px", 46)
    margin_top = _render_int(params, "plot_margin_top_px", 70)
    margin_bottom = _render_int(params, "plot_margin_bottom_px", 96)
    margin_left, margin_right, margin_top, margin_bottom, jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(margin_left),
        right_px=int(margin_right),
        top_px=int(margin_top),
        bottom_px=int(margin_bottom),
        params=params,
        defaults=RENDERING_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.layout",
    )
    return WaterfallRenderParams(
        canvas_width=_render_int(params, "canvas_width", 1180),
        canvas_height=_render_int(params, "canvas_height", 760),
        plot_margin_left_px=int(margin_left),
        plot_margin_right_px=int(margin_right),
        plot_margin_top_px=int(margin_top),
        plot_margin_bottom_px=int(margin_bottom),
        axis_line_width_px=_render_int(params, "axis_line_width_px", 2),
        grid_line_width_px=_render_int(params, "grid_line_width_px", 1),
        connector_width_px=_render_int(params, "connector_width_px", 2),
        bar_outline_width_px=_render_int(params, "bar_outline_width_px", 2),
        tick_length_px=_render_int(params, "tick_length_px", 8),
        title_font_size_px=_render_int(params, "title_font_size_px", 26),
        tick_font_size_px=_render_int(params, "tick_font_size_px", 16),
        label_font_size_px=_render_int(params, "label_font_size_px", 18),
        value_font_size_px=_render_int(params, "value_font_size_px", 17),
        threshold_font_size_px=_render_int(params, "threshold_font_size_px", 16),
        bar_width_fraction=max(0.35, min(0.78, _render_float(params, "bar_width_fraction", 0.56))),
        axis_color_rgb=_render_rgb(params, "axis_color_rgb", (62, 68, 78), instance_seed=int(instance_seed)),
        grid_color_rgb=_render_rgb(params, "grid_color_rgb", (224, 228, 235), instance_seed=int(instance_seed)),
        plot_fill_rgb=_render_rgb(params, "plot_fill_rgb", (255, 255, 255), instance_seed=int(instance_seed)),
        text_color_rgb=_render_rgb(params, "text_color_rgb", (38, 42, 52), instance_seed=int(instance_seed)),
        muted_text_rgb=_render_rgb(params, "muted_text_rgb", (82, 92, 108), instance_seed=int(instance_seed)),
        text_stroke_rgb=_render_rgb(params, "text_stroke_rgb", (255, 255, 255), instance_seed=int(instance_seed)),
        start_fill_rgb=_render_rgb(params, "start_fill_rgb", (76, 118, 178), instance_seed=int(instance_seed)),
        final_fill_rgb=_render_rgb(params, "final_fill_rgb", (58, 92, 150), instance_seed=int(instance_seed)),
        positive_fill_rgb=_render_rgb(params, "positive_fill_rgb", (74, 154, 98), instance_seed=int(instance_seed)),
        negative_fill_rgb=_render_rgb(params, "negative_fill_rgb", (206, 92, 74), instance_seed=int(instance_seed)),
        connector_rgb=_render_rgb(params, "connector_rgb", (126, 134, 148), instance_seed=int(instance_seed)),
        threshold_rgb=_render_rgb(params, "threshold_rgb", (102, 91, 168), instance_seed=int(instance_seed)),
        layout_jitter_meta=dict(jitter_meta),
    )


def _apply_information_style(render_params: WaterfallRenderParams, style: Any) -> WaterfallRenderParams:
    """Apply non-semantic chart style roles without changing waterfall bar meanings."""

    return replace(
        render_params,
        axis_color_rgb=tuple(int(value) for value in style.axis_rgb),
        grid_color_rgb=tuple(int(value) for value in style.grid_rgb),
        plot_fill_rgb=tuple(int(value) for value in style.surface_rgb),
        text_color_rgb=tuple(int(value) for value in style.text_rgb),
        muted_text_rgb=tuple(int(value) for value in style.muted_text_rgb),
        text_stroke_rgb=tuple(int(value) for value in style.text_stroke_rgb),
        connector_rgb=tuple(int(value) for value in style.connector_rgb),
        threshold_rgb=tuple(int(value) for value in style.highlight_rgb),
    )


def _draw_waterfall(
    base_image: Image.Image,
    *,
    dataset: WaterfallDataset,
    render_params: WaterfallRenderParams,
    threshold_value: int | None,
) -> RenderedWaterfall:
    """Draw the chart and record all bar, value-label, and x-label boxes."""

    image = base_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = int(render_params.canvas_width), int(render_params.canvas_height)
    left = int(render_params.plot_margin_left_px)
    right = width - int(render_params.plot_margin_right_px)
    top = int(render_params.plot_margin_top_px)
    bottom = height - int(render_params.plot_margin_bottom_px)
    plot_bbox = _bbox([left, top, right, bottom])
    draw.rounded_rectangle(
        [left, top, right, bottom],
        radius=8,
        fill=tuple(render_params.plot_fill_rgb),
        outline=tuple(render_params.axis_color_rgb),
        width=1,
    )

    title_font = load_font(int(render_params.title_font_size_px), bold=False)
    tick_font = load_font(int(render_params.tick_font_size_px), bold=False)
    label_font = load_font(int(render_params.label_font_size_px), bold=dense_fit_bold())
    value_font = load_font(int(render_params.value_font_size_px), bold=dense_fit_bold())
    threshold_font = load_font(int(render_params.threshold_font_size_px), bold=dense_fit_bold())

    draw_text_traced(
        draw,
        (left, max(12, top - 48)),
        "Waterfall chart",
        font=title_font,
        fill=tuple(render_params.text_color_rgb),
        role="readout",
        required=False,
    )

    y_axis_max = int(group_default(RENDERING_DEFAULTS, "y_axis_max", 100))
    y_axis_max = int(max(80, y_axis_max))
    plot_w = float(right - left)

    def y_for(value: int | float) -> float:
        return project_linear_inverted(
            float(value),
            domain_min=0.0,
            domain_max=float(y_axis_max),
            pixel_top=float(top),
            pixel_bottom=float(bottom),
        )

    tick_step = int(group_default(RENDERING_DEFAULTS, "y_tick_step", 20))
    y_tick_values = range(0, int(y_axis_max) + 1, int(tick_step))
    y_tick_positions = draw_horizontal_value_grid_ticks(
        draw,
        plot_bbox,
        tick_values=y_tick_values,
        domain_min=0,
        domain_max=int(y_axis_max),
        grid_rgb=render_params.grid_color_rgb,
        axis_rgb=render_params.axis_color_rgb,
        grid_width_px=int(render_params.grid_line_width_px),
        tick_width_px=1,
        tick_length_px=float(render_params.tick_length_px),
    )
    for tick in y_tick_values:
        y = float(y_tick_positions[float(tick)])
        text = str(tick)
        bbox = draw.textbbox((0, 0), text, font=tick_font)
        draw_text_traced(
            draw,
            (left - int(render_params.tick_length_px) - 8 - (bbox[2] - bbox[0]), y - (bbox[3] - bbox[1]) / 2),
            text,
            font=tick_font,
            fill=tuple(render_params.muted_text_rgb),
            role="readout",
            required=False,
        )
    draw_axis_lines(draw, plot_bbox, axis_rgb=render_params.axis_color_rgb, axis_width_px=int(render_params.axis_line_width_px))

    items: list[tuple[str, str, int, int, int, RGB]] = [
        ("start", "Start", int(dataset.start_value), 0, int(dataset.start_value), tuple(render_params.start_fill_rgb)),
    ]
    for step in dataset.steps:
        fill = tuple(render_params.positive_fill_rgb if int(step.delta) > 0 else render_params.negative_fill_rgb)
        items.append((str(step.step_id), str(step.label), int(step.delta), int(step.running_before), int(step.running_after), fill))
    items.append(("final", "Final", int(dataset.final_value), 0, int(dataset.final_value), tuple(render_params.final_fill_rgb)))

    slot_count = len(items)
    slot_w = plot_w / float(slot_count)
    bar_w = max(26.0, min(74.0, slot_w * float(render_params.bar_width_fraction)))
    bar_bboxes: dict[str, BBox] = {}
    value_label_bboxes: dict[str, BBox] = {}
    x_label_bboxes: dict[str, BBox] = {}
    connector_bboxes: dict[str, BBox] = {}
    entities: list[dict[str, Any]] = []

    centers: dict[str, float] = {}
    for index, (bar_id, label, raw_value, before, after, fill) in enumerate(items):
        cx = float(left) + (float(index) + 0.5) * slot_w
        centers[str(bar_id)] = float(cx)
        zero_y = y_for(0)
        if str(bar_id) in {"start", "final"}:
            y_top = y_for(int(after))
            y_bottom = zero_y
            value_text = str(int(after))
        else:
            y_top = min(y_for(int(before)), y_for(int(after)))
            y_bottom = max(y_for(int(before)), y_for(int(after)))
            value_text = f"{int(raw_value):+d}"
        box = _bbox([cx - bar_w / 2.0, y_top, cx + bar_w / 2.0, y_bottom])
        draw.rectangle(box, fill=fill, outline=tuple(render_params.axis_color_rgb), width=int(render_params.bar_outline_width_px))
        bar_bboxes[str(bar_id)] = box

        value_center = (float(cx), float(y_top) - 14.0)
        if y_bottom - y_top >= 30.0:
            value_center = (float(cx), float(y_top + y_bottom) / 2.0)
        value_box = _text_bbox_at(draw, text=value_text, center=value_center, font=value_font, stroke_width=dense_stroke_width())
        draw_text_centered(
            draw,
            text=value_text,
            center=value_center,
            font=value_font,
            fill=tuple(render_params.text_color_rgb),
            stroke_fill=tuple(render_params.text_stroke_rgb),
            stroke_width=dense_stroke_width(),
        )
        value_label_bboxes[str(bar_id)] = value_box

        x_center = (float(cx), float(bottom) + 26.0)
        x_box = _text_bbox_at(draw, text=str(label), center=x_center, font=label_font, stroke_width=dense_stroke_width())
        draw_text_centered(
            draw,
            text=str(label),
            center=x_center,
            font=label_font,
            fill=tuple(render_params.text_color_rgb),
            stroke_fill=tuple(render_params.text_stroke_rgb),
            stroke_width=dense_stroke_width(),
        )
        x_label_bboxes[str(bar_id)] = x_box
        entities.append(
            {
                "entity_id": str(bar_id),
                "entity_type": "waterfall_bar",
                "label": str(label),
                "value": int(raw_value),
                "running_before": int(before),
                "running_after": int(after),
                "bar_bbox_px": list(box),
                "value_label_bbox_px": list(value_box),
                "x_label_bbox_px": list(x_box),
            }
        )

    previous_id = "start"
    for step in dataset.steps:
        y = y_for(int(step.running_before))
        x1 = centers[str(previous_id)] + bar_w / 2.0
        x2 = centers[str(step.step_id)] - bar_w / 2.0
        draw.line([x1, y, x2, y], fill=tuple(render_params.connector_rgb), width=int(render_params.connector_width_px))
        connector_bboxes[f"{previous_id}->{step.step_id}"] = _bbox([x1, y - 3, x2, y + 3])
        previous_id = str(step.step_id)
    final_y = y_for(int(dataset.final_value))
    x1 = centers[str(previous_id)] + bar_w / 2.0
    x2 = centers["final"] - bar_w / 2.0
    draw.line([x1, final_y, x2, final_y], fill=tuple(render_params.connector_rgb), width=int(render_params.connector_width_px))
    connector_bboxes[f"{previous_id}->final"] = _bbox([x1, final_y - 3, x2, final_y + 3])

    extra_bboxes: dict[str, BBox] = {}
    if threshold_value is not None:
        threshold_int = int(threshold_value)
        threshold_y = y_for(threshold_int)
        dash = 12
        x = float(left)
        while x < float(right):
            draw.line(
                [x, threshold_y, min(float(right), x + dash), threshold_y],
                fill=tuple(render_params.threshold_rgb),
                width=max(1, int(render_params.connector_width_px)),
            )
            x += dash * 1.8
        threshold_text = f"T={threshold_int}"
        center = (float(right) - 34.0, threshold_y - 14.0)
        threshold_box = _text_bbox_at(draw, text=threshold_text, center=center, font=threshold_font, stroke_width=dense_stroke_width())
        draw_text_centered(
            draw,
            text=threshold_text,
            center=center,
            font=threshold_font,
            fill=tuple(render_params.threshold_rgb),
            stroke_fill=tuple(render_params.text_stroke_rgb),
            stroke_width=dense_stroke_width(),
        )
        extra_bboxes["threshold_line"] = _bbox([left, threshold_y - 3, right, threshold_y + 3])
        extra_bboxes["threshold_label"] = threshold_box

    return RenderedWaterfall(
        image=image,
        entities=tuple(entities),
        plot_bbox_px=list(plot_bbox),
        bar_bboxes_px=dict(bar_bboxes),
        value_label_bboxes_px=dict(value_label_bboxes),
        x_label_bboxes_px=dict(x_label_bboxes),
        connector_bboxes_px=dict(connector_bboxes),
        extra_bboxes_px=dict(extra_bboxes),
        threshold_value=int(threshold_value) if threshold_value is not None else None,
        y_axis_max=int(y_axis_max),
    )


def render_waterfall_dataset(
    *,
    dataset: WaterfallDataset,
    params: Mapping[str, Any],
    instance_seed: int,
    threshold_value: int | None = None,
) -> WaterfallRenderArtifacts:
    """Render a sampled waterfall scene and return traceable artifacts."""

    render_params = resolve_render_params(params, instance_seed=int(instance_seed))
    protected_colors = (
        tuple(int(value) for value in render_params.start_fill_rgb),
        tuple(int(value) for value in render_params.final_fill_rgb),
        tuple(int(value) for value in render_params.positive_fill_rgb),
        tuple(int(value) for value in render_params.negative_fill_rgb),
        tuple(int(value) for value in render_params.threshold_rgb),
    )
    information_style, information_style_meta = resolve_chart_information_style(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="waterfall",
        protected_colors=protected_colors,
    )
    render_params = _apply_information_style(render_params, information_style)
    background, background_meta = make_chart_information_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=information_style,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.information_scene_background",
    )
    background_meta = dict(background_meta)
    background_meta["information_scene_style"] = dict(information_style_meta)
    chart_font_family = sample_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
    )
    with temporary_default_font_family(str(chart_font_family)):
        rendered = _draw_waterfall(
            background,
            dataset=dataset,
            render_params=render_params,
            threshold_value=threshold_value,
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return WaterfallRenderArtifacts(
        image=image,
        rendered_scene=rendered,
        render_params=render_params,
        background_style=dict(background_meta),
        post_image_noise=dict(post_noise_meta),
        chart_font_family=str(chart_font_family),
    )


__all__ = ["render_waterfall_dataset", "resolve_render_params"]
