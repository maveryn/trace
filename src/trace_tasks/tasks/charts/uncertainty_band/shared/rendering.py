"""Rendering helpers for uncertainty-band charts."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.charts.shared.cartesian.axes import draw_axis_lines, draw_horizontal_value_grid_ticks
from trace_tasks.tasks.charts.shared.cartesian.geometry import project_index, project_linear_inverted
from trace_tasks.tasks.charts.shared.information_style import make_chart_information_background, resolve_chart_information_style
from trace_tasks.tasks.charts.shared.visual_defaults import (
    chart_font_asset_metadata,
    relative_luminance,
    sample_chart_font_family,
)
from trace_tasks.tasks.shared.render_variation import apply_layout_jitter_to_margins, resolve_render_int, resolve_render_rgb
from trace_tasks.tasks.shared.text_legibility import draw_traced_text, text_legibility_metadata_for_surfaces
from trace_tasks.tasks.shared.text_rendering import draw_text_centered, fit_font_to_box, load_font

from .defaults import POST_IMAGE_NOISE_DEFAULTS, RENDER_DEFAULTS, rendering_default
from .state import Dataset, RGB, RenderArtifacts, RenderParams, Rendered, SCENE_ID, SCENE_NAMESPACE


def resolve_uncertainty_band_render_params(params: Mapping[str, Any], *, instance_seed: int) -> RenderParams:
    """Resolve style, font, and jittered plot geometry for one render pass."""

    left = int(params.get("plot_margin_left_px", rendering_default("plot_margin_left_px", 84)))
    right = int(params.get("plot_margin_right_px", rendering_default("plot_margin_right_px", 172)))
    top = int(params.get("plot_margin_top_px", rendering_default("plot_margin_top_px", 84)))
    bottom = int(params.get("plot_margin_bottom_px", rendering_default("plot_margin_bottom_px", 92)))
    left, right, top, bottom, jitter_meta = apply_layout_jitter_to_margins(
        left_px=left,
        right_px=right,
        top_px=top,
        bottom_px=bottom,
        params=params,
        defaults=RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.layout",
    )
    font_family = sample_chart_font_family(
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.font",
        params=params,
        exclude_tags=("display",),
    )
    return RenderParams(
        canvas_width=int(params.get("canvas_width", rendering_default("canvas_width", 1180))),
        canvas_height=int(params.get("canvas_height", rendering_default("canvas_height", 760))),
        margin_left_px=int(left),
        margin_right_px=int(right),
        margin_top_px=int(top),
        margin_bottom_px=int(bottom),
        title_band_height_px=int(params.get("title_band_height_px", rendering_default("title_band_height_px", 58))),
        legend_width_px=int(params.get("legend_width_px", rendering_default("legend_width_px", 156))),
        axis_line_width_px=int(resolve_render_int(params, RENDER_DEFAULTS, "axis_line_width_px", 2, instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE)),
        grid_line_width_px=int(resolve_render_int(params, RENDER_DEFAULTS, "grid_line_width_px", 1, instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE)),
        band_outline_width_px=int(resolve_render_int(params, RENDER_DEFAULTS, "band_outline_width_px", 2, instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE)),
        center_line_width_px=int(resolve_render_int(params, RENDER_DEFAULTS, "center_line_width_px", 4, instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE)),
        point_radius_px=int(resolve_render_int(params, RENDER_DEFAULTS, "point_radius_px", 5, instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE)),
        title_font_size_px=int(params.get("title_font_size_px", rendering_default("title_font_size_px", 28))),
        label_font_size_px=int(params.get("label_font_size_px", rendering_default("label_font_size_px", 17))),
        tick_font_size_px=int(params.get("tick_font_size_px", rendering_default("tick_font_size_px", 15))),
        legend_font_size_px=int(params.get("legend_font_size_px", rendering_default("legend_font_size_px", 17))),
        axis_min=int(params.get("axis_min", rendering_default("axis_min", 0))),
        axis_max=int(params.get("axis_max", rendering_default("axis_max", 100))),
        tick_step=int(params.get("tick_step", rendering_default("tick_step", 20))),
        band_alpha=max(32, min(120, int(params.get("band_alpha", rendering_default("band_alpha", 58))))),
        panel_fill_rgb=resolve_render_rgb(params, RENDER_DEFAULTS, "panel_fill_rgb", (255, 255, 255), instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE),
        panel_border_rgb=resolve_render_rgb(params, RENDER_DEFAULTS, "panel_border_rgb", (190, 198, 208), instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE),
        axis_rgb=resolve_render_rgb(params, RENDER_DEFAULTS, "axis_rgb", (64, 68, 76), instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE),
        grid_rgb=resolve_render_rgb(params, RENDER_DEFAULTS, "grid_rgb", (224, 227, 232), instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE),
        text_rgb=resolve_render_rgb(params, RENDER_DEFAULTS, "text_rgb", (38, 41, 48), instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE),
        muted_text_rgb=resolve_render_rgb(params, RENDER_DEFAULTS, "muted_text_rgb", (88, 96, 112), instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE),
        text_stroke_rgb=resolve_render_rgb(params, RENDER_DEFAULTS, "text_stroke_rgb", (255, 255, 255), instance_seed=int(instance_seed), namespace=SCENE_NAMESPACE),
        font_family=str(font_family),
        layout_jitter_meta=dict(jitter_meta),
    )


def _darken(color: RGB, factor: float = 0.72) -> RGB:
    return tuple(max(0, min(255, int(round(float(channel) * float(factor))))) for channel in color)  # type: ignore[return-value]


def _readable_chart_text_colors(surface_rgb: Sequence[int]) -> Tuple[RGB, RGB, RGB]:
    if relative_luminance(surface_rgb) >= 0.55:
        return (34, 42, 54), (72, 84, 100), (34, 42, 54)
    return (246, 250, 255), (205, 218, 232), (18, 24, 32)


def _scale_y(value: int | float, *, plot_bottom: float, plot_height: float, axis_min: int, axis_max: int) -> float:
    return project_linear_inverted(
        float(value),
        domain_min=float(axis_min),
        domain_max=float(axis_max),
        pixel_top=float(plot_bottom) - float(plot_height),
        pixel_bottom=float(plot_bottom),
    )


def _draw_text(
    draw: ImageDraw.ImageDraw,
    *,
    xy: tuple[float, float],
    text: str,
    font: Any,
    fill: RGB,
    stroke: RGB,
    stroke_width: int = 1,
    surface_rgbs: Sequence[Sequence[int]] | None = None,
) -> None:
    draw_traced_text(
        draw,
        xy=(float(xy[0]), float(xy[1])),
        text=str(text),
        font=font,
        fill_rgb=tuple(int(value) for value in fill),
        stroke_rgb=tuple(int(value) for value in stroke),
        stroke_width=int(stroke_width),
        role="chart_text",
        required=True,
        extra_metadata=(
            text_legibility_metadata_for_surfaces(fill_rgb=fill, surface_rgbs=surface_rgbs)
            if surface_rgbs is not None
            else None
        ),
    )


def _render_chart(dataset: Dataset, params: Mapping[str, Any], *, instance_seed: int) -> Rendered:
    """Render the neutral uncertainty-band scene and projection maps.

    The renderer only draws the sampled bands and records visual coordinates; it
    does not know which public task or query will consume those coordinates.
    """

    render_params = resolve_uncertainty_band_render_params(params, instance_seed=int(instance_seed))
    style, style_meta = resolve_chart_information_style(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        protected_colors=[series.color_rgb for series in dataset.series],
        allow_colored_surface=True,
    )
    image, background_meta = make_chart_information_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=style,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.background",
    )
    panel_fill_rgb = tuple(int(value) for value in style.surface_rgb)
    panel_border_rgb = tuple(int(value) for value in style.panel_border_rgb)
    axis_rgb = tuple(int(value) for value in style.axis_rgb)
    grid_rgb = tuple(int(value) for value in style.grid_rgb)
    text_rgb, _muted_text_rgb, text_stroke_rgb = _readable_chart_text_colors(panel_fill_rgb)
    text_stroke_width = 0
    if image.mode != "RGB":
        image = image.convert("RGB")
    draw = ImageDraw.Draw(image)

    plot_left = float(render_params.margin_left_px)
    plot_right = float(render_params.canvas_width - render_params.margin_right_px)
    plot_top = float(render_params.margin_top_px + render_params.title_band_height_px)
    plot_bottom = float(render_params.canvas_height - render_params.margin_bottom_px)
    plot_width = float(plot_right - plot_left)
    plot_height = float(plot_bottom - plot_top)
    plot_bbox = [float(plot_left), float(plot_top), float(plot_right), float(plot_bottom)]
    if plot_width <= 10 or plot_height <= 10:
        raise ValueError("uncertainty band plot area is too small")
    panel_bbox = [
        float(plot_left - 42),
        float(render_params.margin_top_px),
        float(min(render_params.canvas_width - 26, plot_right + render_params.legend_width_px)),
        float(plot_bottom + 54),
    ]
    draw.rounded_rectangle(
        tuple(panel_bbox),
        radius=14,
        fill=panel_fill_rgb,
        outline=panel_border_rgb,
        width=2,
    )

    title_font = load_font(render_params.title_font_size_px, bold=True, font_family=render_params.font_family)
    tick_font = load_font(render_params.tick_font_size_px, bold=False, font_family=render_params.font_family)
    legend_font = load_font(render_params.legend_font_size_px, bold=True, font_family=render_params.font_family)
    _draw_text(
        draw,
        xy=(plot_left, float(render_params.margin_top_px + 12)),
        text=str(dataset.title),
        font=title_font,
        fill=text_rgb,
        stroke=text_stroke_rgb,
        stroke_width=int(text_stroke_width),
        surface_rgbs=(panel_fill_rgb,),
    )

    y_tick_values = range(int(render_params.axis_min), int(render_params.axis_max) + 1, int(render_params.tick_step))
    y_tick_positions = draw_horizontal_value_grid_ticks(
        draw,
        plot_bbox,
        tick_values=y_tick_values,
        domain_min=int(render_params.axis_min),
        domain_max=int(render_params.axis_max),
        grid_rgb=grid_rgb,
        axis_rgb=axis_rgb,
        grid_width_px=int(render_params.grid_line_width_px),
        tick_width_px=1,
    )
    for tick in y_tick_values:
        y = float(y_tick_positions[float(tick)])
        label = str(tick)
        bbox = draw.textbbox((0, 0), label, font=tick_font, stroke_width=1)
        _draw_text(
            draw,
            xy=(plot_left - 14 - float(bbox[2] - bbox[0]), y - (0.5 * float(bbox[3] - bbox[1]))),
            text=label,
            font=tick_font,
            fill=text_rgb,
            stroke=text_stroke_rgb,
            stroke_width=int(text_stroke_width),
            surface_rgbs=(panel_fill_rgb,),
        )
    draw_axis_lines(draw, plot_bbox, axis_rgb=axis_rgb, axis_width_px=int(render_params.axis_line_width_px))

    x_positions = [
        project_index(index, pixel_min=float(plot_left), pixel_max=float(plot_right), count=len(dataset.x_labels))
        for index in range(len(dataset.x_labels))
    ]

    for index, label in enumerate(dataset.x_labels):
        x = float(x_positions[index])
        draw.line((x, plot_bottom, x, plot_bottom + 7), fill=axis_rgb, width=1)
        max_label_width = max(42.0, plot_width / max(1, len(dataset.x_labels)) - 6.0)
        fitted = fit_font_to_box(
            draw,
            text=str(label),
            max_width=float(max_label_width),
            max_height=28.0,
            bold=True,
            font_family=render_params.font_family,
            min_size_px=9,
            max_size_px=render_params.label_font_size_px,
        )
        draw_text_centered(
            draw,
            text=str(label),
            center=(x, plot_bottom + 25),
            font=fitted,
            fill=text_rgb,
            stroke_fill=text_stroke_rgb,
            stroke_width=int(text_stroke_width),
        )

    point_map: dict[str, dict[str, dict[str, list[float]]]] = {}
    series_band_bboxes: dict[str, list[float]] = {}
    entities: list[dict[str, Any]] = []

    base_rgba = image.convert("RGBA")
    series_geometries: list[dict[str, Any]] = []

    for series in dataset.series:
        upper_points: list[tuple[float, float]] = []
        mid_points: list[tuple[float, float]] = []
        lower_points: list[tuple[float, float]] = []
        point_map[str(series.series_id)] = {}
        for index, x_label in enumerate(dataset.x_labels):
            x = float(x_positions[index])
            upper_y = _scale_y(series.upper_values[index], plot_bottom=plot_bottom, plot_height=plot_height, axis_min=render_params.axis_min, axis_max=render_params.axis_max)
            mid_y = _scale_y(series.mid_values[index], plot_bottom=plot_bottom, plot_height=plot_height, axis_min=render_params.axis_min, axis_max=render_params.axis_max)
            lower_y = _scale_y(series.lower_values[index], plot_bottom=plot_bottom, plot_height=plot_height, axis_min=render_params.axis_min, axis_max=render_params.axis_max)
            upper_points.append((x, upper_y))
            mid_points.append((x, mid_y))
            lower_points.append((x, lower_y))
            point_map[str(series.series_id)][str(x_label)] = {
                "upper_bound": [round(float(x), 3), round(float(upper_y), 3)],
                "center_line": [round(float(x), 3), round(float(mid_y), 3)],
                "lower_bound": [round(float(x), 3), round(float(lower_y), 3)],
            }
            entities.append(
                {
                    "entity_id": f"{series.series_id}:{index}",
                    "entity_type": "uncertainty_band_point",
                    "series_id": str(series.series_id),
                    "series_label": str(series.label),
                    "x_index": int(index),
                    "x_label": str(x_label),
                    "lower": int(series.lower_values[index]),
                    "midpoint": int(series.mid_values[index]),
                    "upper": int(series.upper_values[index]),
                    "upper_point_px": point_map[str(series.series_id)][str(x_label)]["upper_bound"],
                    "center_point_px": point_map[str(series.series_id)][str(x_label)]["center_line"],
                    "lower_point_px": point_map[str(series.series_id)][str(x_label)]["lower_bound"],
                }
            )
        polygon = list(upper_points) + list(reversed(lower_points))
        xs = [point[0] for point in polygon]
        ys = [point[1] for point in polygon]
        series_band_bboxes[str(series.series_id)] = [
            round(min(xs), 3),
            round(min(ys), 3),
            round(max(xs), 3),
            round(max(ys), 3),
        ]
        series_geometries.append(
            {
                "series": series,
                "polygon": polygon,
                "upper_points": list(upper_points),
                "lower_points": list(lower_points),
                "mid_points": list(mid_points),
            }
        )

    band_rgba = base_rgba
    for geometry in series_geometries:
        series = geometry["series"]
        fill_layer = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
        fill_draw = ImageDraw.Draw(fill_layer)
        fill_draw.polygon(
            geometry["polygon"],
            fill=(*tuple(int(value) for value in series.color_rgb), int(render_params.band_alpha)),
        )
        band_rgba = Image.alpha_composite(band_rgba, fill_layer)

    line_layer = Image.new("RGBA", base_rgba.size, (0, 0, 0, 0))
    line_draw = ImageDraw.Draw(line_layer)
    for geometry in series_geometries:
        series = geometry["series"]
        outline_rgb = _darken(series.color_rgb, 0.55)
        line_draw.line(
            geometry["upper_points"],
            fill=(*outline_rgb, 235),
            width=int(render_params.band_outline_width_px),
        )
        line_draw.line(
            geometry["lower_points"],
            fill=(*outline_rgb, 235),
            width=int(render_params.band_outline_width_px),
        )
        line_draw.line(
            geometry["mid_points"],
            fill=(*outline_rgb, 255),
            width=int(render_params.center_line_width_px),
            joint="curve",
        )

    image = Image.alpha_composite(band_rgba, line_layer).convert("RGB")
    draw = ImageDraw.Draw(image)

    for series in dataset.series:
        for x_label in dataset.x_labels:
            center = point_map[str(series.series_id)][str(x_label)]["center_line"]
            r = int(render_params.point_radius_px)
            x, y = float(center[0]), float(center[1])
            draw.ellipse(
                (x - r, y - r, x + r, y + r),
                fill=tuple(int(value) for value in _darken(series.color_rgb, 0.60)),
                outline=text_stroke_rgb,
                width=1,
            )

    legend_x = float(plot_right + 26)
    legend_y = float(plot_top + 8)
    for index, series in enumerate(dataset.series):
        y = float(legend_y + (index * 34))
        draw.rounded_rectangle((legend_x, y + 3, legend_x + 24, y + 17), radius=4, fill=series.color_rgb, outline=_darken(series.color_rgb, 0.62), width=1)
        _draw_text(
            draw,
            xy=(legend_x + 34, y),
            text=str(series.label),
            font=legend_font,
            fill=text_rgb,
            stroke=text_stroke_rgb,
            stroke_width=int(text_stroke_width),
        )

    overlap_points: dict[str, list[float]] = {}
    series_a, series_b = dataset.series
    for index, x_label in enumerate(dataset.x_labels):
        low = max(int(series_a.lower_values[index]), int(series_b.lower_values[index]))
        high = min(int(series_a.upper_values[index]), int(series_b.upper_values[index]))
        if int(low) <= int(high):
            value = 0.5 * (float(low) + float(high))
            overlap_points[str(x_label)] = [
                round(float(x_positions[index]), 3),
                round(_scale_y(value, plot_bottom=plot_bottom, plot_height=plot_height, axis_min=render_params.axis_min, axis_max=render_params.axis_max), 3),
            ]

    return Rendered(
        image=image,
        entities=tuple(entities),
        plot_bbox_px=[round(plot_left, 3), round(plot_top, 3), round(plot_right, 3), round(plot_bottom, 3)],
        series_band_bboxes_px=dict(series_band_bboxes),
        point_map_px=dict(point_map),
        overlap_points_px=dict(overlap_points),
        render_meta={
            "panel_bbox_px": [round(float(value), 3) for value in panel_bbox],
            "font_assets": chart_font_asset_metadata(render_params.font_family),
            "layout_jitter": dict(render_params.layout_jitter_meta),
            "background": dict(background_meta),
            "information_style": dict(style_meta),
            "axis_min": int(render_params.axis_min),
            "axis_max": int(render_params.axis_max),
            "tick_step": int(render_params.tick_step),
        },
    )


def render_dataset(dataset: Dataset, *, params: Mapping[str, Any], instance_seed: int) -> RenderArtifacts:
    rendered = _render_chart(dataset, params, instance_seed=int(instance_seed))
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=dict(params),
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderArtifacts(
        rendered=replace(rendered, image=image),
        post_image_noise=dict(post_noise_meta),
    )


__all__ = ["render_dataset", "resolve_uncertainty_band_render_params"]
