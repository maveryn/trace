"""Rendering primitives for candlestick/OHLC chart scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.tasks.charts.shared.cartesian.frame import plot_bbox_from_margins
from trace_tasks.tasks.charts.shared.cartesian.geometry import round_bbox, round_point
from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.candlestick.shared.defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    RENDERING_DEFAULTS,
    SCENE_NAMESPACE,
)
from trace_tasks.tasks.charts.candlestick.shared.state import BBox, Dataset, Point, RenderArtifacts, RenderParams, Rendered
from trace_tasks.tasks.charts.shared.visual_defaults import chart_font_asset_metadata, sample_chart_font_family
from trace_tasks.tasks.shared.bbox_projection import bbox_union_raw as _bbox_union
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.render_variation import apply_layout_jitter_to_margins, resolve_render_rgb
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import draw_text_centered, load_font, temporary_default_font_family


def _bbox(values: Sequence[float]) -> BBox:
    return round_bbox(values)


def bbox_center(box: Sequence[float]) -> Point:
    return round_point((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


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
    return int(params.get(str(key), group_default(RENDERING_DEFAULTS, str(key), int(fallback))))


def _render_float(params: Mapping[str, Any], key: str, fallback: float) -> float:
    return float(params.get(str(key), group_default(RENDERING_DEFAULTS, str(key), float(fallback))))


def resolve_render_params(params: Mapping[str, Any], *, instance_seed: int) -> RenderParams:
    """Resolve reusable chart geometry, colors, and jitter before any candle target is chosen."""

    margin_left = _render_int(params, "plot_margin_left_px", 94)
    margin_right = _render_int(params, "plot_margin_right_px", 70)
    margin_top = _render_int(params, "plot_margin_top_px", 82)
    margin_bottom = _render_int(params, "plot_margin_bottom_px", 108)
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
    return RenderParams(
        canvas_width=_render_int(params, "canvas_width", 1440),
        canvas_height=_render_int(params, "canvas_height", 820),
        plot_margin_left_px=int(margin_left),
        plot_margin_right_px=int(margin_right),
        plot_margin_top_px=int(margin_top),
        plot_margin_bottom_px=int(margin_bottom),
        axis_line_width_px=_render_int(params, "axis_line_width_px", 2),
        grid_line_width_px=_render_int(params, "grid_line_width_px", 1),
        wick_line_width_px=_render_int(params, "wick_line_width_px", 4),
        body_outline_width_px=_render_int(params, "body_outline_width_px", 2),
        tick_length_px=_render_int(params, "tick_length_px", 8),
        title_font_size_px=_render_int(params, "title_font_size_px", 26),
        tick_font_size_px=_render_int(params, "tick_font_size_px", 16),
        label_font_size_px=_render_int(params, "label_font_size_px", 18),
        value_font_size_px=_render_int(params, "value_font_size_px", 15),
        candle_width_fraction=max(0.26, min(0.62, _render_float(params, "candle_width_fraction", 0.36))),
        y_axis_min=_render_int(params, "y_axis_min", 0),
        y_axis_max=max(80, _render_int(params, "y_axis_max", 100)),
        y_tick_step=max(5, _render_int(params, "y_tick_step", 20)),
        axis_color_rgb=resolve_render_rgb(
            params, RENDERING_DEFAULTS, "axis_color_rgb", (62, 68, 78), instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE
        ),
        grid_color_rgb=resolve_render_rgb(
            params, RENDERING_DEFAULTS, "grid_color_rgb", (224, 228, 235), instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE
        ),
        plot_fill_rgb=resolve_render_rgb(
            params, RENDERING_DEFAULTS, "plot_fill_rgb", (255, 255, 255), instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE
        ),
        text_color_rgb=resolve_render_rgb(
            params, RENDERING_DEFAULTS, "text_color_rgb", (38, 42, 52), instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE
        ),
        muted_text_rgb=resolve_render_rgb(
            params, RENDERING_DEFAULTS, "muted_text_rgb", (82, 92, 108), instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE
        ),
        text_stroke_rgb=resolve_render_rgb(
            params, RENDERING_DEFAULTS, "text_stroke_rgb", (255, 255, 255), instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE
        ),
        up_fill_rgb=resolve_render_rgb(
            params, RENDERING_DEFAULTS, "up_fill_rgb", (59, 151, 119), instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE
        ),
        down_fill_rgb=resolve_render_rgb(
            params, RENDERING_DEFAULTS, "down_fill_rgb", (210, 93, 83), instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE
        ),
        wick_rgb=resolve_render_rgb(
            params, RENDERING_DEFAULTS, "wick_rgb", (54, 60, 70), instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE
        ),
        body_outline_rgb=resolve_render_rgb(
            params, RENDERING_DEFAULTS, "body_outline_rgb", (38, 44, 54), instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE
        ),
        layout_jitter_meta=dict(jitter_meta),
    )


def draw_candlesticks(
    base_image: Image.Image,
    *,
    dataset: Dataset,
    render_params: RenderParams,
) -> Rendered:
    """Draw candles and record every candle/body/wick/value-label projection used by tasks."""

    image = base_image.convert("RGB")
    draw = ImageDraw.Draw(image)
    width, height = int(render_params.canvas_width), int(render_params.canvas_height)
    left = int(render_params.plot_margin_left_px)
    right = width - int(render_params.plot_margin_right_px)
    top = int(render_params.plot_margin_top_px)
    bottom = height - int(render_params.plot_margin_bottom_px)
    plot_bbox = plot_bbox_from_margins(
        canvas_width=float(width),
        canvas_height=float(height),
        margin_left_px=float(render_params.plot_margin_left_px),
        margin_right_px=float(render_params.plot_margin_right_px),
        margin_top_px=float(render_params.plot_margin_top_px),
        margin_bottom_px=float(render_params.plot_margin_bottom_px),
    )
    draw.rounded_rectangle(
        [left, top, right, bottom],
        radius=8,
        fill=tuple(render_params.plot_fill_rgb),
        outline=(202, 208, 218),
        width=1,
    )

    tick_font = load_font(int(render_params.tick_font_size_px), bold=False)
    label_font = load_font(int(render_params.label_font_size_px), bold=True)
    value_font = load_font(int(render_params.value_font_size_px), bold=False)

    y_min = int(render_params.y_axis_min)
    y_max = int(render_params.y_axis_max)
    plot_h = float(bottom - top)
    plot_w = float(right - left)

    def y_for(value: int | float) -> float:
        span = max(1.0, float(y_max - y_min))
        return float(bottom) - ((float(value) - float(y_min)) / span) * plot_h

    for tick in range(int(y_min), int(y_max) + 1, int(render_params.y_tick_step)):
        y = y_for(int(tick))
        draw.line([left, y, right, y], fill=tuple(render_params.grid_color_rgb), width=int(render_params.grid_line_width_px))
        draw.line([left - int(render_params.tick_length_px), y, left, y], fill=tuple(render_params.axis_color_rgb), width=1)
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
    draw.line([left, bottom, right, bottom], fill=tuple(render_params.axis_color_rgb), width=int(render_params.axis_line_width_px))
    draw.line([left, top, left, bottom], fill=tuple(render_params.axis_color_rgb), width=int(render_params.axis_line_width_px))

    slot_count = len(dataset.candles)
    slot_w = plot_w / float(slot_count)
    body_w = max(30.0, min(54.0, slot_w * float(render_params.candle_width_fraction)))
    candle_bboxes: dict[str, BBox] = {}
    body_bboxes: dict[str, BBox] = {}
    wick_bboxes: dict[str, BBox] = {}
    value_label_bboxes: dict[str, BBox] = {}
    x_label_bboxes: dict[str, BBox] = {}
    entities: list[dict[str, Any]] = []

    for index, candle in enumerate(dataset.candles):
        cx = float(left) + (float(index) + 0.5) * slot_w
        high_y = y_for(int(candle.high_value))
        low_y = y_for(int(candle.low_value))
        open_y = y_for(int(candle.open_value))
        close_y = y_for(int(candle.close_value))
        body_top = min(open_y, close_y)
        body_bottom = max(open_y, close_y)
        if body_bottom - body_top < 8:
            midpoint = (body_top + body_bottom) / 2.0
            body_top = midpoint - 4.0
            body_bottom = midpoint + 4.0

        wick_box = _bbox([cx - 4, high_y, cx + 4, low_y])
        draw.line([cx, high_y, cx, low_y], fill=tuple(render_params.wick_rgb), width=int(render_params.wick_line_width_px))
        tick_half = max(8.0, body_w * 0.24)
        draw.line([cx - tick_half, high_y, cx + tick_half, high_y], fill=tuple(render_params.wick_rgb), width=2)
        draw.line([cx - tick_half, low_y, cx + tick_half, low_y], fill=tuple(render_params.wick_rgb), width=2)
        fill = tuple(render_params.up_fill_rgb if candle.direction == "up" else render_params.down_fill_rgb)
        body_box = _bbox([cx - body_w / 2.0, body_top, cx + body_w / 2.0, body_bottom])
        draw.rounded_rectangle(
            body_box,
            radius=4,
            fill=fill,
            outline=tuple(render_params.body_outline_rgb),
            width=int(render_params.body_outline_width_px),
        )
        candle_box = _bbox_union([wick_box, body_box], padding=3.0)
        wick_bboxes[str(candle.candle_id)] = wick_box
        body_bboxes[str(candle.candle_id)] = body_box
        candle_bboxes[str(candle.candle_id)] = candle_box

        side_label_offset = max(16.0, body_w * 0.32)
        label_specs = [
            ("high", f"H{int(candle.high_value)}", (cx, max(float(top) + 14.0, high_y - 18.0))),
            ("low", f"L{int(candle.low_value)}", (cx, min(float(bottom) - 14.0, low_y + 18.0))),
            ("open", f"O{int(candle.open_value)}", (cx - body_w / 2.0 - side_label_offset, open_y)),
            ("close", f"C{int(candle.close_value)}", (cx + body_w / 2.0 + side_label_offset, close_y)),
        ]
        for value_kind, text, center in label_specs:
            box = _text_bbox_at(draw, text=str(text), center=center, font=value_font, stroke_width=0)
            draw_text_centered(
                draw,
                text=str(text),
                center=center,
                font=value_font,
                fill=tuple(render_params.text_color_rgb),
                stroke_fill=tuple(render_params.text_stroke_rgb),
                stroke_width=0,
            )
            value_label_bboxes[f"{candle.candle_id}:{value_kind}"] = list(box)

        x_center = (float(cx), float(bottom) + 28.0)
        x_box = _text_bbox_at(draw, text=str(candle.label), center=x_center, font=label_font, stroke_width=1)
        draw_text_centered(
            draw,
            text=str(candle.label),
            center=x_center,
            font=label_font,
            fill=tuple(render_params.text_color_rgb),
            stroke_fill=tuple(render_params.text_stroke_rgb),
            stroke_width=1,
        )
        x_label_bboxes[str(candle.candle_id)] = list(x_box)
        entities.append(
            {
                "entity_id": str(candle.candle_id),
                "entity_type": "ohlc_candle",
                "label": str(candle.label),
                "open": int(candle.open_value),
                "high": int(candle.high_value),
                "low": int(candle.low_value),
                "close": int(candle.close_value),
                "direction": str(candle.direction),
                "body_size": int(candle.body_size),
                "wick_range": int(candle.wick_range),
                "candle_bbox_px": list(candle_box),
                "body_bbox_px": list(body_box),
                "wick_bbox_px": list(wick_box),
                "x_label_bbox_px": list(x_box),
            }
        )

    return Rendered(
        image=image,
        entities=tuple(entities),
        plot_bbox_px=list(plot_bbox),
        candle_bboxes_px=dict(candle_bboxes),
        body_bboxes_px=dict(body_bboxes),
        wick_bboxes_px=dict(wick_bboxes),
        value_label_bboxes_px=dict(value_label_bboxes),
        x_label_bboxes_px=dict(x_label_bboxes),
    )


def render_dataset(
    *,
    dataset: Dataset,
    params: Mapping[str, Any],
    instance_seed: int,
) -> RenderArtifacts:
    """Render a prepared OHLC dataset without choosing any task target or answer."""

    render_params = resolve_render_params(params, instance_seed=int(instance_seed))
    protected_colors = (
        render_params.up_fill_rgb,
        render_params.down_fill_rgb,
        render_params.wick_rgb,
        render_params.body_outline_rgb,
    )
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=dict(params),
        scene_id="candlestick",
        render_params=render_params,
        protected_colors=protected_colors,
    )
    chart_font_family = sample_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
    )
    with temporary_default_font_family(str(chart_font_family)):
        rendered = draw_candlesticks(background, dataset=dataset, render_params=render_params)
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    rendered = Rendered(
        image=image,
        entities=tuple(dict(entity) for entity in rendered.entities),
        plot_bbox_px=list(rendered.plot_bbox_px),
        candle_bboxes_px=dict(rendered.candle_bboxes_px),
        body_bboxes_px=dict(rendered.body_bboxes_px),
        wick_bboxes_px=dict(rendered.wick_bboxes_px),
        value_label_bboxes_px=dict(rendered.value_label_bboxes_px),
        x_label_bboxes_px=dict(rendered.x_label_bboxes_px),
    )
    return RenderArtifacts(
        rendered=rendered,
        render_params=render_params,
        background_style={**dict(background_meta), "information_scene_style": dict(information_style_meta)},
        font_assets=chart_font_asset_metadata(str(chart_font_family)),
        post_image_noise=dict(post_noise_meta),
    )


__all__ = ["bbox_center", "draw_candlesticks", "render_dataset", "resolve_render_params"]
