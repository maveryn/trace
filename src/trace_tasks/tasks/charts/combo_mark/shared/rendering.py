"""Rendering helpers for combo-mark chart tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.charts.shared.cartesian.geometry import project_linear_inverted
from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.combo_mark.shared.defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    RENDERING_DEFAULTS,
    SCENE_NAMESPACE,
    scene_default,
)
from trace_tasks.tasks.charts.combo_mark.shared.state import ComboDataset, ComboRenderArtifacts, ComboScene, RenderParams
from trace_tasks.tasks.charts.shared.visual_defaults import (
    chart_font_asset_metadata,
    sample_chart_font_family,
)
from trace_tasks.tasks.shared.bbox_projection import bbox_union
from trace_tasks.tasks.shared.render_variation import apply_layout_jitter_to_margins, resolve_render_int, resolve_render_rgb
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import draw_text_centered, load_font, temporary_default_font_family


def render_params(params: Mapping[str, Any], *, instance_seed: int) -> RenderParams:
    """Resolve combo chart dimensions, style colors, fonts, and jitter.

    The helper is intentionally visual-only: public task files provide sampled
    data and semantic target bindings, while this function only derives the
    scene's rendering parameters from config and seed.
    """

    canvas_width = int(params.get("canvas_width", scene_default(RENDERING_DEFAULTS, "canvas_width", 1080)))
    canvas_height = int(params.get("canvas_height", scene_default(RENDERING_DEFAULTS, "canvas_height", 660)))
    plot_left = int(params.get("plot_left_px", scene_default(RENDERING_DEFAULTS, "plot_left_px", 86)))
    plot_right_margin = int(
        params.get("plot_right_margin_px", scene_default(RENDERING_DEFAULTS, "plot_right_margin_px", 176))
    )
    plot_top = int(params.get("plot_top_px", scene_default(RENDERING_DEFAULTS, "plot_top_px", 58)))
    plot_bottom_margin = int(
        params.get("plot_bottom_margin_px", scene_default(RENDERING_DEFAULTS, "plot_bottom_margin_px", 92))
    )
    plot_left, plot_right_margin, plot_top, plot_bottom_margin, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(plot_left),
        right_px=int(plot_right_margin),
        top_px=int(plot_top),
        bottom_px=int(plot_bottom_margin),
        params=params,
        defaults=RENDERING_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=SCENE_NAMESPACE,
    )
    return RenderParams(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        plot_left=int(plot_left),
        plot_right=int(canvas_width - int(plot_right_margin)),
        plot_top=int(plot_top),
        plot_bottom=int(canvas_height - int(plot_bottom_margin)),
        axis_width=resolve_render_int(
            params, RENDERING_DEFAULTS, "axis_line_width_px", 2, instance_seed=instance_seed, namespace=SCENE_NAMESPACE
        ),
        grid_width=resolve_render_int(
            params, RENDERING_DEFAULTS, "grid_line_width_px", 1, instance_seed=instance_seed, namespace=SCENE_NAMESPACE
        ),
        line_width=resolve_render_int(
            params, RENDERING_DEFAULTS, "line_width_px", 4, instance_seed=instance_seed, namespace=SCENE_NAMESPACE
        ),
        point_radius=resolve_render_int(
            params, RENDERING_DEFAULTS, "point_radius_px", 7, instance_seed=instance_seed, namespace=SCENE_NAMESPACE
        ),
        tick_font_size=int(params.get("tick_font_size_px", scene_default(RENDERING_DEFAULTS, "tick_font_size_px", 16))),
        label_font_size=int(
            params.get("label_font_size_px", scene_default(RENDERING_DEFAULTS, "label_font_size_px", 18))
        ),
        value_font_size=int(
            params.get("value_font_size_px", scene_default(RENDERING_DEFAULTS, "value_font_size_px", 15))
        ),
        legend_font_size=int(
            params.get("legend_font_size_px", scene_default(RENDERING_DEFAULTS, "legend_font_size_px", 18))
        ),
        primary_rgb=resolve_render_rgb(
            params, RENDERING_DEFAULTS, "primary_rgb", (73, 125, 203), instance_seed=instance_seed, namespace=SCENE_NAMESPACE
        ),
        primary_alt_rgb=resolve_render_rgb(
            params,
            RENDERING_DEFAULTS,
            "primary_alt_rgb",
            (104, 171, 130),
            instance_seed=instance_seed,
            namespace=SCENE_NAMESPACE,
        ),
        line_rgb=resolve_render_rgb(
            params, RENDERING_DEFAULTS, "line_rgb", (220, 115, 55), instance_seed=instance_seed, namespace=SCENE_NAMESPACE
        ),
        area_rgb=resolve_render_rgb(
            params, RENDERING_DEFAULTS, "area_rgb", (117, 166, 214), instance_seed=instance_seed, namespace=SCENE_NAMESPACE
        ),
        axis_rgb=resolve_render_rgb(
            params,
            RENDERING_DEFAULTS,
            "axis_color_rgb",
            (66, 72, 82),
            instance_seed=instance_seed,
            namespace=SCENE_NAMESPACE,
        ),
        grid_rgb=resolve_render_rgb(
            params,
            RENDERING_DEFAULTS,
            "grid_color_rgb",
            (224, 228, 234),
            instance_seed=instance_seed,
            namespace=SCENE_NAMESPACE,
        ),
        text_rgb=resolve_render_rgb(
            params,
            RENDERING_DEFAULTS,
            "text_color_rgb",
            (36, 42, 52),
            instance_seed=instance_seed,
            namespace=SCENE_NAMESPACE,
        ),
        text_stroke_rgb=resolve_render_rgb(
            params,
            RENDERING_DEFAULTS,
            "text_stroke_rgb",
            (255, 255, 255),
            instance_seed=instance_seed,
            namespace=SCENE_NAMESPACE,
        ),
        panel_rgb=resolve_render_rgb(
            params,
            RENDERING_DEFAULTS,
            "plot_fill_rgb",
            (255, 255, 255),
            instance_seed=instance_seed,
            namespace=SCENE_NAMESPACE,
        ),
        layout_jitter_meta=dict(layout_jitter_meta),
    )


def _round_axis_max(value: int) -> int:
    return int(max(20, int(math.ceil((float(value) + 8.0) / 10.0) * 10)))


def _value_to_y(value: int, *, top: int, bottom: int, axis_max: int) -> float:
    return project_linear_inverted(
        float(value),
        domain_min=0.0,
        domain_max=float(axis_max),
        pixel_top=float(top),
        pixel_bottom=float(bottom),
    )


def _draw_label_box(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font,
    fill: Tuple[int, int, int],
    stroke_fill: Tuple[int, int, int] = (255, 255, 255),
) -> Tuple[float, float, float, float]:
    bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=1)
    w = float(bbox[2] - bbox[0])
    h = float(bbox[3] - bbox[1])
    x0 = float(center[0]) - (0.5 * w) - 4.0
    y0 = float(center[1]) - (0.5 * h) - 2.0
    x1 = float(center[0]) + (0.5 * w) + 4.0
    y1 = float(center[1]) + (0.5 * h) + 2.0
    draw.rounded_rectangle((x0, y0, x1, y1), radius=4, fill=stroke_fill, outline=(214, 218, 226), width=1)
    draw_text_centered(draw, text=str(text), center=center, font=font, fill=fill, stroke_width=0)
    return (x0, y0, x1, y1)


def _draw_axes(
    draw: ImageDraw.ImageDraw,
    *,
    p: RenderParams,
    labels: Sequence[str],
    primary_axis_max: int,
    line_axis_max: int,
    dual_axis: bool,
) -> None:
    """Draw shared combo-chart axes and labels without binding task semantics."""

    tick_font = load_font(p.tick_font_size, bold=False)
    label_font = load_font(p.label_font_size, bold=True)
    left, right, top, bottom = p.plot_left, p.plot_right, p.plot_top, p.plot_bottom
    draw.rounded_rectangle(
        (left - 14, top - 18, right + 14, bottom + 12),
        radius=10,
        fill=p.panel_rgb,
        outline=(211, 216, 224),
        width=1,
    )
    for i in range(5):
        value = int(round(primary_axis_max * i / 4))
        y = _value_to_y(value, top=top, bottom=bottom, axis_max=primary_axis_max)
        draw.line((left, y, right, y), fill=p.grid_rgb, width=p.grid_width)
        draw_text_traced(draw, (left - 12, y), str(value), anchor="rm", font=tick_font, fill=p.text_rgb, role="readout", required=False)
        if dual_axis:
            line_value = int(round(line_axis_max * i / 4))
            draw_text_traced(
                draw, (right + 12, y), str(line_value), anchor="lm", font=tick_font, fill=p.text_rgb, role="readout", required=False
            )
    draw.line((left, top, left, bottom), fill=p.axis_rgb, width=p.axis_width)
    draw.line((left, bottom, right, bottom), fill=p.axis_rgb, width=p.axis_width)
    if dual_axis:
        draw.line((right, top, right, bottom), fill=p.axis_rgb, width=p.axis_width)
    step = float(right - left) / float(max(1, len(labels)))
    for idx, label in enumerate(labels):
        x = float(left) + (float(idx) + 0.5) * step
        draw.line((x, bottom, x, bottom + 6), fill=p.axis_rgb, width=1)
        draw_text_centered(
            draw,
            text=str(label),
            center=(x, bottom + 24),
            font=label_font,
            fill=p.text_rgb,
            stroke_fill=p.text_stroke_rgb,
            stroke_width=1,
        )


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    *,
    p: RenderParams,
    scene_variant: str,
    primary_name: str,
    line_name: str,
) -> Tuple[float, float, float, float]:
    font = load_font(p.legend_font_size, bold=True)
    x0 = p.plot_right + 38
    y0 = p.plot_top + 24
    boxes: list[Sequence[float]] = []
    title_bbox = draw.textbbox((x0, y0), "Legend", font=font)
    boxes.append(title_bbox)
    draw_text_traced(draw, (x0, y0), "Legend", font=font, fill=p.text_rgb, role="readout", required=False)
    y = y0 + 34
    if scene_variant == "area_line_overlay":
        primary_swatch_bbox = (x0, y + 2, x0 + 28, y + 18)
        draw.rectangle(primary_swatch_bbox, fill=p.area_rgb, outline=p.primary_rgb)
    elif scene_variant == "stacked_bar_line":
        primary_swatch_bbox = (x0, y + 2, x0 + 28, y + 20)
        draw.rectangle((x0, y + 2, x0 + 13, y + 20), fill=p.primary_rgb, outline=(55, 62, 72))
        draw.rectangle((x0 + 15, y + 2, x0 + 28, y + 20), fill=p.primary_alt_rgb, outline=(55, 62, 72))
    else:
        primary_swatch_bbox = (x0, y + 2, x0 + 28, y + 20)
        draw.rectangle(primary_swatch_bbox, fill=p.primary_rgb, outline=(55, 62, 72))
    boxes.append(primary_swatch_bbox)
    primary_text_bbox = draw.textbbox((x0 + 38, y), str(primary_name), font=font)
    boxes.append(primary_text_bbox)
    draw_text_traced(draw, (x0 + 38, y), str(primary_name), font=font, fill=p.text_rgb, role="readout", required=False)
    y += 34
    line_swatch_bbox = (x0, y + 2, x0 + 30, y + 12 + max(0, p.line_width // 2))
    boxes.append(line_swatch_bbox)
    draw.line((x0, y + 11, x0 + 30, y + 11), fill=p.line_rgb, width=p.line_width)
    draw.ellipse((x0 + 11, y + 2, x0 + 21, y + 12), fill=(255, 255, 255), outline=p.line_rgb, width=3)
    line_text_bbox = draw.textbbox((x0 + 38, y), str(line_name), font=font)
    boxes.append(line_text_bbox)
    draw_text_traced(draw, (x0 + 38, y), str(line_name), font=font, fill=p.text_rgb, role="readout", required=False)
    return tuple(float(value) for value in bbox_union(boxes, padding=8.0))


def render_combo_scene(
    base: Image.Image,
    *,
    labels: Sequence[str],
    primary_values: Sequence[int],
    line_values: Sequence[int],
    scene_variant: str,
    primary_name: str,
    line_name: str,
    params: Mapping[str, Any],
    instance_seed: int,
    resolved_render_params: RenderParams | None = None,
) -> ComboScene:
    """Render one combo-mark chart and return projected mark centers."""

    p = resolved_render_params or render_params(params, instance_seed=int(instance_seed))
    image = base.convert("RGB")
    draw = ImageDraw.Draw(image)
    primary_axis_max = _round_axis_max(max(primary_values))
    line_axis_max = (
        _round_axis_max(max(line_values))
        if scene_variant == "bar_line_dual_axis"
        else _round_axis_max(max(max(primary_values), max(line_values)))
    )
    if scene_variant != "bar_line_dual_axis":
        primary_axis_max = line_axis_max
    _draw_axes(
        draw,
        p=p,
        labels=labels,
        primary_axis_max=primary_axis_max,
        line_axis_max=line_axis_max,
        dual_axis=(scene_variant == "bar_line_dual_axis"),
    )
    value_font = load_font(p.value_font_size, bold=True)
    n = len(labels)
    step = float(p.plot_right - p.plot_left) / float(max(1, n))
    bar_width = max(22.0, min(56.0, 0.44 * float(step)))
    primary_points: list[Tuple[float, float]] = []
    line_points: list[Tuple[float, float]] = []
    entities: list[Dict[str, Any]] = []

    if scene_variant == "area_line_overlay":
        area_points = []
        for idx, value in enumerate(primary_values):
            x = float(p.plot_left) + (float(idx) + 0.5) * step
            y = _value_to_y(int(value), top=p.plot_top, bottom=p.plot_bottom, axis_max=primary_axis_max)
            area_points.append((x, y))
        polygon = [(area_points[0][0], float(p.plot_bottom))] + area_points + [(area_points[-1][0], float(p.plot_bottom))]
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        odraw = ImageDraw.Draw(overlay)
        odraw.polygon(polygon, fill=tuple(p.area_rgb) + (112,))
        odraw.line(area_points, fill=tuple(p.primary_rgb) + (255,), width=3)
        image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
        draw = ImageDraw.Draw(image)

    for idx, (label, primary_value) in enumerate(zip(labels, primary_values)):
        x = float(p.plot_left) + (float(idx) + 0.5) * step
        y = _value_to_y(int(primary_value), top=p.plot_top, bottom=p.plot_bottom, axis_max=primary_axis_max)
        primary_points.append((x, y))
        if scene_variant in {"bar_line_shared_axis", "bar_line_dual_axis"}:
            box = (x - 0.5 * bar_width, y, x + 0.5 * bar_width, p.plot_bottom)
            draw.rectangle(box, fill=p.primary_rgb, outline=(55, 62, 72), width=1)
        elif scene_variant == "stacked_bar_line":
            split = int(round(int(primary_value) * (0.42 + 0.16 * (idx % 3) / 2.0)))
            split = max(2, min(int(primary_value) - 2, split))
            y_split = _value_to_y(split, top=p.plot_top, bottom=p.plot_bottom, axis_max=primary_axis_max)
            draw.rectangle(
                (x - 0.5 * bar_width, y_split, x + 0.5 * bar_width, p.plot_bottom),
                fill=p.primary_rgb,
                outline=(55, 62, 72),
                width=1,
            )
            draw.rectangle(
                (x - 0.5 * bar_width, y, x + 0.5 * bar_width, y_split),
                fill=p.primary_alt_rgb,
                outline=(55, 62, 72),
                width=1,
            )
        _draw_label_box(draw, text=str(int(primary_value)), center=(x, max(p.plot_top + 12, y - 15)), font=value_font, fill=p.primary_rgb)
        entities.append(
            {
                "type": "combo_primary_mark",
                "metric": str(primary_name),
                "x_label": str(label),
                "value": int(primary_value),
                "center_px": [float(x), float(y)],
                "bbox_px": [float(x - bar_width * 0.55), float(y), float(x + bar_width * 0.55), float(p.plot_bottom)],
            }
        )

    for idx, line_value in enumerate(line_values):
        x = float(p.plot_left) + (float(idx) + 0.5) * step
        y = _value_to_y(int(line_value), top=p.plot_top, bottom=p.plot_bottom, axis_max=line_axis_max)
        line_points.append((x, y))
    if len(line_points) >= 2:
        draw.line(line_points, fill=p.line_rgb, width=p.line_width, joint="curve")
    for idx, (label, line_value) in enumerate(zip(labels, line_values)):
        x, y = line_points[idx]
        r = float(p.point_radius)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(255, 255, 255), outline=p.line_rgb, width=3)
        offset = -21 if float(y) > float(primary_points[idx][1]) - 22 else 22
        _draw_label_box(
            draw,
            text=str(int(line_value)),
            center=(x, min(p.plot_bottom - 14, max(p.plot_top + 12, y + offset))),
            font=value_font,
            fill=p.line_rgb,
        )
        entities.append(
            {
                "type": "combo_line_point",
                "metric": str(line_name),
                "x_label": str(label),
                "value": int(line_value),
                "center_px": [float(x), float(y)],
                "bbox_px": [float(x - r), float(y - r), float(x + r), float(y + r)],
            }
        )
    legend_bbox = _draw_legend(draw, p=p, scene_variant=str(scene_variant), primary_name=str(primary_name), line_name=str(line_name))
    return ComboScene(
        image=image,
        labels=tuple(str(label) for label in labels),
        primary_values=tuple(int(value) for value in primary_values),
        line_values=tuple(int(value) for value in line_values),
        primary_points=tuple(primary_points),
        line_points=tuple(line_points),
        entities=tuple(entities),
        scene_variant=str(scene_variant),
        primary_name=str(primary_name),
        line_name=str(line_name),
        primary_axis_max=int(primary_axis_max),
        line_axis_max=int(line_axis_max),
        plot_bbox=(int(p.plot_left), int(p.plot_top), int(p.plot_right), int(p.plot_bottom)),
        legend_bbox=tuple(float(value) for value in legend_bbox),
    )


def render_dataset(
    *,
    dataset: ComboDataset,
    params: Mapping[str, Any],
    instance_seed: int,
) -> ComboRenderArtifacts:
    """Render a sampled combo dataset without binding task semantics."""

    resolved_render_params = render_params(params, instance_seed=int(instance_seed))
    protected_colors = (
        resolved_render_params.primary_rgb,
        resolved_render_params.primary_alt_rgb,
        resolved_render_params.line_rgb,
        resolved_render_params.area_rgb,
    )
    resolved_render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="combo_mark",
        render_params=resolved_render_params,
        protected_colors=protected_colors,
    )
    chart_font_family = sample_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
    )
    with temporary_default_font_family(str(chart_font_family)):
        scene = render_combo_scene(
            background,
            labels=dataset.labels,
            primary_values=dataset.primary_values,
            line_values=dataset.line_values,
            scene_variant=str(dataset.scene_variant),
            primary_name=str(dataset.primary_name),
            line_name=str(dataset.line_name),
            params=params,
            instance_seed=int(instance_seed),
            resolved_render_params=resolved_render_params,
        )
    image, post_noise_meta = apply_post_image_noise(
        scene.image,
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    scene = ComboScene(
        image=image,
        labels=tuple(scene.labels),
        primary_values=tuple(scene.primary_values),
        line_values=tuple(scene.line_values),
        primary_points=tuple(scene.primary_points),
        line_points=tuple(scene.line_points),
        entities=tuple(dict(entity) for entity in scene.entities),
        scene_variant=str(scene.scene_variant),
        primary_name=str(scene.primary_name),
        line_name=str(scene.line_name),
        primary_axis_max=int(scene.primary_axis_max),
        line_axis_max=int(scene.line_axis_max),
        plot_bbox=tuple(int(value) for value in scene.plot_bbox),
        legend_bbox=tuple(float(value) for value in scene.legend_bbox),
    )
    return ComboRenderArtifacts(
        scene=scene,
        render_params=resolved_render_params,
        background_style={**dict(background_meta), "information_scene_style": dict(information_style_meta)},
        font_assets=chart_font_asset_metadata(str(chart_font_family)),
        post_image_noise=dict(post_noise_meta),
    )


__all__ = ["render_combo_scene", "render_dataset", "render_params"]
