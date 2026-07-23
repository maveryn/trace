"""Rendering helpers for radial-progress charts."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from PIL import Image, ImageDraw

from trace_tasks.tasks.charts.shared.dense_text import dense_fit_bold, dense_stroke_width
from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.shared.panel.grid_layout import layout_panel_grid
from trace_tasks.tasks.shared.drawing import draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font, temporary_default_font_family

from .defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    font_assets_payload,
    render_int,
    render_rgb,
    sample_chart_font_family,
)
from .state import (
    BBox,
    FULL_PROGRESS_RINGS,
    ProgressDataset,
    ProgressItem,
    RGB,
    SEGMENTED_RADIAL_BARS,
    SEMICIRCLE_GAUGES,
    RadialProgressRenderResult,
    RenderedProgressScene,
    RenderParams,
)


def bbox(values: Sequence[float]) -> BBox:
    return [round(float(value), 3) for value in values]


def resolve_render_params(params: Mapping[str, Any], *, instance_seed: int) -> RenderParams:
    return RenderParams(
        canvas_width=render_int(params, "canvas_width", 1320),
        canvas_height=render_int(params, "canvas_height", 900),
        outer_margin_px=render_int(params, "outer_margin_px", 48),
        title_band_height_px=render_int(params, "title_band_height_px", 68),
        card_gap_px=render_int(params, "card_gap_px", 18),
        card_corner_radius_px=render_int(params, "card_corner_radius_px", 12),
        card_outline_width_px=render_int(params, "card_outline_width_px", 2),
        title_font_size_px=render_int(params, "title_font_size_px", 30),
        label_font_size_px=render_int(params, "label_font_size_px", 20),
        tick_font_size_px=render_int(params, "tick_font_size_px", 13),
        ring_width_px=render_int(params, "ring_width_px", 16),
        gauge_width_px=render_int(params, "gauge_width_px", 16),
        segment_width_px=render_int(params, "segment_width_px", 14),
        tick_length_px=render_int(params, "tick_length_px", 8),
        text_rgb=render_rgb(params, "text_rgb", [36, 42, 52], instance_seed=int(instance_seed)),
        muted_text_rgb=render_rgb(params, "muted_text_rgb", [88, 96, 112], instance_seed=int(instance_seed)),
        text_stroke_rgb=render_rgb(params, "text_stroke_rgb", [255, 255, 255], instance_seed=int(instance_seed)),
        card_fill_rgb=render_rgb(params, "card_fill_rgb", [255, 255, 255], instance_seed=int(instance_seed)),
        card_alt_fill_rgb=render_rgb(params, "card_alt_fill_rgb", [248, 250, 252], instance_seed=int(instance_seed)),
        card_outline_rgb=render_rgb(params, "card_outline_rgb", [194, 202, 214], instance_seed=int(instance_seed)),
        track_rgb=render_rgb(params, "track_rgb", [225, 230, 238], instance_seed=int(instance_seed)),
        tick_rgb=render_rgb(params, "tick_rgb", [96, 107, 122], instance_seed=int(instance_seed)),
        needle_rgb=render_rgb(params, "needle_rgb", [44, 52, 64], instance_seed=int(instance_seed)),
    )


def _rgb_luminance(rgb: Sequence[int]) -> float:
    """Return perceptual luminance for simple light/dark style decisions."""

    red, green, blue = (float(int(channel)) for channel in rgb[:3])
    return (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)


def _blend_rgb(source: Sequence[int], target: Sequence[int], target_weight: float) -> RGB:
    weight = min(1.0, max(0.0, float(target_weight)))
    return tuple(
        max(0, min(255, int(round((float(source[index]) * (1.0 - weight)) + (float(target[index]) * weight)))))
        for index in range(3)
    )


def _style_role_rgb(information_style_meta: Mapping[str, Any], role: str, fallback: RGB) -> RGB:
    raw_roles = information_style_meta.get("roles_rgb", {})
    if not isinstance(raw_roles, Mapping):
        return tuple(int(channel) for channel in fallback)
    raw_value = raw_roles.get(str(role))
    if not isinstance(raw_value, Sequence) or len(raw_value) < 3:
        return tuple(int(channel) for channel in fallback)
    return tuple(int(raw_value[index]) for index in range(3))


def _is_dark_information_style(render_params: RenderParams, information_style_meta: Mapping[str, Any]) -> bool:
    style_text = " ".join(
        str(information_style_meta.get(key, ""))
        for key in ("treatment", "palette_id", "style_pack")
    ).casefold()
    return "dark" in style_text or _rgb_luminance(render_params.card_fill_rgb) < 96.0


def _with_radial_progress_track_contrast(
    render_params: RenderParams,
    *,
    information_style_meta: Mapping[str, Any],
) -> tuple[RenderParams, dict[str, Any]]:
    """Make inactive progress tracks visually subordinate on dark themes."""

    if not _is_dark_information_style(render_params, information_style_meta):
        return render_params, {
            "progress_track_policy": "style_track",
            "track_rgb": list(render_params.track_rgb),
        }
    shadow_rgb = _style_role_rgb(information_style_meta, "shadow", render_params.text_stroke_rgb)
    track_rgb = _blend_rgb(render_params.card_fill_rgb, shadow_rgb, 0.52)
    return replace(render_params, track_rgb=track_rgb), {
        "progress_track_policy": "dark_inactive_track",
        "track_rgb": list(track_rgb),
        "base_track_rgb": list(render_params.track_rgb),
        "track_source_rgb": list(render_params.card_fill_rgb),
        "track_shadow_rgb": list(shadow_rgb),
    }


def point(cx: float, cy: float, radius: float, degrees: float) -> tuple[float, float]:
    radians = math.radians(float(degrees))
    return float(cx + (radius * math.cos(radians))), float(cy + (radius * math.sin(radians)))


def draw_readout_text(
    draw: ImageDraw.ImageDraw,
    *,
    xy: tuple[float, float],
    text: str,
    font: Any,
    fill: RGB,
    stroke_fill: RGB,
    anchor: str = "mm",
    stroke_width: int = 0,
) -> BBox:
    draw_text_traced(
        draw,
        (float(xy[0]), float(xy[1])),
        str(text),
        font=font,
        fill=tuple(fill),
        stroke_width=max(0, int(stroke_width)),
        stroke_fill=tuple(stroke_fill),
        anchor=str(anchor),
        role="readout",
        required=False,
    )
    text_bbox = draw.textbbox(
        (float(xy[0]), float(xy[1])),
        str(text),
        font=font,
        stroke_width=max(0, int(stroke_width)),
        anchor=str(anchor),
    )
    return bbox(text_bbox)


def draw_ticks(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    radius: float,
    values: Sequence[int],
    start_degrees: float,
    span_degrees: float,
    p: RenderParams,
    label_values: Sequence[int],
) -> None:
    tick_font = load_font(p.tick_font_size_px, bold=False)
    label_set = {int(value) for value in label_values}
    cx, cy = float(center[0]), float(center[1])
    for value in values:
        angle = float(start_degrees + (span_degrees * (int(value) / 100.0)))
        outer = point(cx, cy, float(radius) + 3.0, angle)
        inner = point(cx, cy, max(1.0, float(radius) - float(p.tick_length_px)), angle)
        draw.line([inner, outer], fill=tuple(p.tick_rgb), width=2)
        if int(value) in label_set:
            label_point = point(cx, cy, float(radius) + 19.0, angle)
            text = "0/100" if int(value) == 0 and 100 in label_set else str(value)
            if int(value) == 100 and 0 in label_set:
                continue
            draw_readout_text(
                draw,
                xy=label_point,
                text=text,
                font=tick_font,
                fill=p.muted_text_rgb,
                stroke_fill=p.text_stroke_rgb,
                anchor="mm",
                stroke_width=dense_stroke_width(),
            )


def draw_full_ring(
    draw: ImageDraw.ImageDraw,
    *,
    item: ProgressItem,
    center: tuple[float, float],
    radius: float,
    p: RenderParams,
) -> BBox:
    cx, cy = float(center[0]), float(center[1])
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    width = max(6, int(p.ring_width_px))
    draw.arc(box, start=-90, end=270, fill=tuple(p.track_rgb), width=width)
    end_angle = float(-90.0 + (360.0 * (int(item.value) / 100.0)))
    draw.arc(box, start=-90, end=end_angle, fill=tuple(item.color_rgb), width=width)
    draw_ticks(
        draw,
        center=(cx, cy),
        radius=radius,
        values=(0, 25, 50, 75, 100),
        start_degrees=-90,
        span_degrees=360,
        p=p,
        label_values=(0, 50),
    )
    return bbox([cx - radius - 24, cy - radius - 24, cx + radius + 24, cy + radius + 24])


def draw_semicircle_gauge(
    draw: ImageDraw.ImageDraw,
    *,
    item: ProgressItem,
    center: tuple[float, float],
    radius: float,
    p: RenderParams,
) -> BBox:
    cx, cy = float(center[0]), float(center[1])
    box = (cx - radius, cy - radius, cx + radius, cy + radius)
    width = max(6, int(p.gauge_width_px))
    draw.arc(box, start=180, end=360, fill=tuple(p.track_rgb), width=width)
    end_angle = float(180.0 + (180.0 * (int(item.value) / 100.0)))
    draw.arc(box, start=180, end=end_angle, fill=tuple(item.color_rgb), width=width)
    draw_ticks(
        draw,
        center=(cx, cy),
        radius=radius,
        values=(0, 25, 50, 75, 100),
        start_degrees=180,
        span_degrees=180,
        p=p,
        label_values=(0, 50, 100),
    )
    needle_end = point(cx, cy, max(1.0, radius - 22.0), end_angle)
    draw.line([(cx, cy), needle_end], fill=tuple(p.needle_rgb), width=3)
    r = 5.0
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=tuple(p.needle_rgb), outline=tuple(p.text_stroke_rgb), width=1)
    return bbox([cx - radius - 24, cy - radius - 28, cx + radius + 24, cy + 24])


def paste_annular_segment(
    image: Image.Image,
    *,
    center: tuple[float, float],
    radius: float,
    width: float,
    start_degrees: float,
    end_degrees: float,
    fill: RGB,
) -> None:
    """Render one anti-aliased annular wedge used by segmented progress bars."""

    cx, cy = float(center[0]), float(center[1])
    outer_radius = float(radius) + (float(width) * 0.5)
    inner_radius = max(1.0, float(radius) - (float(width) * 0.5))
    pad = int(math.ceil(float(outer_radius) + 5.0))
    size = int((2 * pad) + 1)
    scale = 4
    scaled_size = int(size * scale)
    local_center = float(pad * scale)

    mask = Image.new("L", (scaled_size, scaled_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    outer_box = (
        local_center - (outer_radius * scale),
        local_center - (outer_radius * scale),
        local_center + (outer_radius * scale),
        local_center + (outer_radius * scale),
    )
    inner_box = (
        local_center - (inner_radius * scale),
        local_center - (inner_radius * scale),
        local_center + (inner_radius * scale),
        local_center + (inner_radius * scale),
    )
    mask_draw.pieslice(outer_box, start=float(start_degrees), end=float(end_degrees), fill=255)
    mask_draw.pieslice(inner_box, start=float(start_degrees) - 1.0, end=float(end_degrees) + 1.0, fill=0)
    try:
        resample = Image.Resampling.LANCZOS
    except AttributeError:  # pragma: no cover - Pillow compatibility
        resample = Image.LANCZOS
    mask = mask.resize((size, size), resample)
    color_patch = Image.new("RGB", (size, size), tuple(int(v) for v in fill))
    image.paste(color_patch, (int(round(cx - pad)), int(round(cy - pad))), mask)


def draw_segmented_radial_bar(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    *,
    item: ProgressItem,
    center: tuple[float, float],
    radius: float,
    p: RenderParams,
) -> BBox:
    """Draw one segmented ring; the value is encoded as filled 5-percent wedges."""

    cx, cy = float(center[0]), float(center[1])
    width = max(10, int(p.segment_width_px))
    segment_count = 20
    filled_count = int(round(int(item.value) / 5.0))
    gap_degrees = 2.2
    for index in range(segment_count):
        start = float(-90.0 + (index * 360.0 / segment_count) + gap_degrees * 0.5)
        end = float(-90.0 + ((index + 1) * 360.0 / segment_count) - gap_degrees * 0.5)
        fill = item.color_rgb if int(index) < int(filled_count) else p.track_rgb
        paste_annular_segment(
            image,
            center=(cx, cy),
            radius=float(radius),
            width=float(width),
            start_degrees=float(start),
            end_degrees=float(end),
            fill=tuple(int(v) for v in fill),
        )
    outline_width = 1
    outer_radius = float(radius) + (float(width) * 0.5)
    inner_radius = max(1.0, float(radius) - (float(width) * 0.5))
    for outline_radius in (inner_radius, outer_radius):
        outline_box = (
            cx - outline_radius,
            cy - outline_radius,
            cx + outline_radius,
            cy + outline_radius,
        )
        draw.ellipse(outline_box, outline=tuple(p.card_outline_rgb), width=outline_width)
    draw_ticks(
        draw,
        center=(cx, cy),
        radius=radius,
        values=(0, 25, 50, 75, 100),
        start_degrees=-90,
        span_degrees=360,
        p=p,
        label_values=(0, 50, 100),
    )
    return bbox([cx - radius - 24, cy - radius - 24, cx + radius + 24, cy + radius + 24])


def entity_record(item: ProgressItem, *, item_bbox: Sequence[float], progress_bbox: Sequence[float]) -> dict[str, Any]:
    return {
        "entity_id": str(item.item_id),
        "entity_type": "radial_progress_item",
        "label": str(item.label),
        "value": int(item.value),
        "remaining": int(100 - int(item.value)),
        "bbox_px": list(item_bbox),
        "progress_bbox_px": list(progress_bbox),
    }


def render_chart(
    *,
    background: Image.Image,
    dataset: ProgressDataset,
    params: Mapping[str, Any],
    instance_seed: int,
    render_params: RenderParams | None = None,
) -> RenderedProgressScene:
    """Render the grid of cards and preserve per-item boxes for annotation projection."""

    p = render_params or resolve_render_params(params, instance_seed=int(instance_seed))
    image = background.convert("RGB")
    if image.size != (int(p.canvas_width), int(p.canvas_height)):
        image = image.resize((int(p.canvas_width), int(p.canvas_height)))
    draw = ImageDraw.Draw(image)

    title_font = load_font(p.title_font_size_px, bold=False)
    label_font = load_font(p.label_font_size_px, bold=dense_fit_bold())
    margin = float(p.outer_margin_px)
    draw_centered_text(
        draw,
        text=str(dataset.title),
        center=(p.canvas_width / 2.0, margin + p.title_band_height_px * 0.38),
        font=title_font,
        fill=p.text_rgb,
        stroke_fill=p.text_stroke_rgb,
        stroke_width=dense_stroke_width(),
    )
    plot_left = margin
    plot_top = margin + float(p.title_band_height_px)
    plot_right = float(p.canvas_width) - margin
    plot_bottom = float(p.canvas_height) - margin
    plot_bbox = bbox([plot_left, plot_top, plot_right, plot_bottom])

    gap = float(p.card_gap_px)
    card_boxes = layout_panel_grid(plot_bbox, panel_count=len(dataset.items), gap_x=float(gap), gap_y=float(gap))
    row_tops = sorted({round(float(box[1]), 6) for box in card_boxes})
    item_bboxes: dict[str, BBox] = {}
    progress_bboxes: dict[str, BBox] = {}
    entities: list[dict[str, Any]] = []

    for index, (item, card_box) in enumerate(zip(dataset.items, card_boxes)):
        x0, y0, x1, y1 = (float(value) for value in card_box)
        card_w = float(x1 - x0)
        card_h = float(y1 - y0)
        row = int(row_tops.index(round(float(y0), 6)))
        card_fill = p.card_fill_rgb if (index + row) % 2 == 0 else p.card_alt_fill_rgb
        draw_rounded_rect(
            draw,
            (x0, y0, x1, y1),
            radius=int(p.card_corner_radius_px),
            fill=card_fill,
            outline=p.card_outline_rgb,
            width=int(p.card_outline_width_px),
        )
        draw_readout_text(
            draw,
            xy=((x0 + x1) / 2.0, y0 + 24.0),
            text=item.label,
            font=label_font,
            fill=p.text_rgb,
            stroke_fill=p.text_stroke_rgb,
            anchor="mm",
            stroke_width=dense_stroke_width(),
        )
        center_x = float((x0 + x1) / 2.0)
        if dataset.scene_variant == SEMICIRCLE_GAUGES:
            center_y = float(y0 + card_h * 0.72)
            radius = max(38.0, min(card_w * 0.36, card_h * 0.44))
            progress_bbox = draw_semicircle_gauge(draw, item=item, center=(center_x, center_y), radius=radius, p=p)
        else:
            center_y = float(y0 + card_h * 0.55)
            radius = max(34.0, min(card_w * 0.32, card_h * 0.31))
            if dataset.scene_variant == FULL_PROGRESS_RINGS:
                progress_bbox = draw_full_ring(draw, item=item, center=(center_x, center_y), radius=radius, p=p)
            elif dataset.scene_variant == SEGMENTED_RADIAL_BARS:
                progress_bbox = draw_segmented_radial_bar(image, draw, item=item, center=(center_x, center_y), radius=radius, p=p)
            else:
                raise ValueError(f"unsupported radial progress scene variant: {dataset.scene_variant}")
        item_bbox = bbox([x0, y0, x1, y1])
        item_bboxes[item.item_id] = item_bbox
        progress_bboxes[item.item_id] = progress_bbox
        entities.append(entity_record(item, item_bbox=item_bbox, progress_bbox=progress_bbox))

    render_meta = {
        "scene_variant": str(dataset.scene_variant),
        "value_scale_min": 0,
        "value_scale_max": 100,
        "style": {
            "card_fill_rgb": list(p.card_fill_rgb),
            "card_alt_fill_rgb": list(p.card_alt_fill_rgb),
            "card_outline_rgb": list(p.card_outline_rgb),
            "track_rgb": list(p.track_rgb),
            "tick_rgb": list(p.tick_rgb),
        },
    }
    return RenderedProgressScene(
        image=image,
        entities=tuple(entities),
        plot_bbox_px=list(plot_bbox),
        item_bboxes_px=dict(item_bboxes),
        progress_bboxes_px=dict(progress_bboxes),
        render_meta=render_meta,
    )


def render_radial_progress_dataset(
    *,
    dataset: ProgressDataset,
    params: Mapping[str, Any],
    instance_seed: int,
) -> RadialProgressRenderResult:
    """Render the full scene using the sampled font/background/noise contracts."""

    resolved_params = resolve_render_params(params, instance_seed=int(instance_seed))
    protected_colors = [tuple(int(channel) for channel in item.color_rgb) for item in dataset.items]
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="radial_progress",
        render_params=resolved_params,
        protected_colors=protected_colors,
    )
    render_params, track_style_meta = _with_radial_progress_track_contrast(
        render_params,
        information_style_meta=information_style_meta,
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
    final_scene = RenderedProgressScene(
        image=image,
        entities=tuple(rendered_scene.entities),
        plot_bbox_px=list(rendered_scene.plot_bbox_px),
        item_bboxes_px=dict(rendered_scene.item_bboxes_px),
        progress_bboxes_px=dict(rendered_scene.progress_bboxes_px),
        render_meta={
            **dict(rendered_scene.render_meta),
            "background_style": {**dict(background_meta), "information_scene_style": dict(information_style_meta)},
            "information_scene_style": dict(information_style_meta),
            "progress_track_style": dict(track_style_meta),
            "post_image_noise": dict(post_noise_meta),
        },
    )
    return RadialProgressRenderResult(
        rendered_scene=final_scene,
        background_meta={**dict(background_meta), "information_scene_style": dict(information_style_meta)},
        post_noise_meta=dict(post_noise_meta),
        chart_font_family=str(chart_font_family),
    )


__all__ = [
    "font_assets_payload",
    "render_radial_progress_dataset",
]
