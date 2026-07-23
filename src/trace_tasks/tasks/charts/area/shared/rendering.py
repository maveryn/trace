"""Rendering helpers for the area chart scene."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.charts.area.shared.defaults import (
    DEFAULT_PALETTE,
    POST_IMAGE_NOISE_DEFAULTS,
    RENDERING_DEFAULTS,
    SCENE_ID,
    SCENE_NAMESPACE,
    scene_default,
)
from trace_tasks.tasks.charts.area.shared.state import AreaRenderParams, AreaRenderResult, RenderedAreaPanel
from trace_tasks.tasks.charts.shared.information_style import (
    make_chart_information_background,
    resolve_chart_information_style,
)
from trace_tasks.tasks.charts.shared.visual_defaults import chart_font_asset_metadata
from trace_tasks.tasks.shared.bbox_projection import round_bbox
from trace_tasks.tasks.shared.color_distance import sample_color_palette_with_distance_constraints
from trace_tasks.tasks.shared.font_assets import font_asset_version, sample_font_family
from trace_tasks.tasks.shared.render_variation import (
    apply_layout_jitter_to_margins,
    resolve_render_int,
    resolve_render_rgb,
)
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import draw_text_centered, load_font, temporary_default_font_family


def _as_rgb(value: Sequence[int] | None, fallback: tuple[int, int, int]) -> tuple[int, int, int]:
    if value is None or len(value) < 3:
        return tuple(int(channel) for channel in fallback)
    return tuple(max(0, min(255, int(channel))) for channel in value[:3])


def _semantic_palette(params: Mapping[str, Any]) -> tuple[tuple[int, int, int], ...]:
    raw_palette = params.get("series_palette_rgb", scene_default(params, RENDERING_DEFAULTS, "series_palette_rgb", DEFAULT_PALETTE))
    if isinstance(raw_palette, Sequence) and not isinstance(raw_palette, (str, bytes)):
        colors = tuple(
            _as_rgb(item, DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)])
            for index, item in enumerate(raw_palette)
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes))
        )
        if colors:
            return colors
    return tuple(tuple(int(channel) for channel in color) for color in DEFAULT_PALETTE)


def _params_with_information_style(params: Mapping[str, Any], style: Any) -> dict[str, Any]:
    styled = dict(params)
    styled.update(
        {
            "axis_color_rgb": list(style.axis_rgb),
            "grid_color_rgb": list(style.grid_rgb),
            "plot_fill_rgb": list(style.surface_rgb),
            "text_color_rgb": list(style.text_rgb),
            "text_stroke_rgb": list(style.text_stroke_rgb),
            "legend_border_rgb": list(style.panel_border_rgb),
        }
    )
    return styled


def resolve_area_render_params(params: Mapping[str, Any], *, instance_seed: int) -> AreaRenderParams:
    """Resolve scene-level render dimensions, style, and jitter.

    This helper owns only visual area-chart layout defaults; public tasks pass
    objective semantics separately through sampled data and annotations.
    """

    canvas_width = int(params.get("canvas_width", scene_default(params, RENDERING_DEFAULTS, "canvas_width", 1060)))
    canvas_height = int(params.get("canvas_height", scene_default(params, RENDERING_DEFAULTS, "canvas_height", 680)))
    margin_left = int(params.get("plot_margin_left_px", scene_default(params, RENDERING_DEFAULTS, "plot_margin_left_px", 84)))
    margin_right = int(params.get("plot_margin_right_px", scene_default(params, RENDERING_DEFAULTS, "plot_margin_right_px", 172)))
    margin_top = int(params.get("plot_margin_top_px", scene_default(params, RENDERING_DEFAULTS, "plot_margin_top_px", 48)))
    margin_bottom = int(params.get("plot_margin_bottom_px", scene_default(params, RENDERING_DEFAULTS, "plot_margin_bottom_px", 84)))
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
    return AreaRenderParams(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        plot_margin_left_px=int(margin_left),
        plot_margin_right_px=int(margin_right),
        plot_margin_top_px=int(margin_top),
        plot_margin_bottom_px=int(margin_bottom),
        axis_line_width_px=int(
            resolve_render_int(
                params,
                RENDERING_DEFAULTS,
                "axis_line_width_px",
                2,
                instance_seed=int(instance_seed),
                namespace=SCENE_NAMESPACE,
            )
        ),
        grid_line_width_px=int(
            resolve_render_int(
                params,
                RENDERING_DEFAULTS,
                "grid_line_width_px",
                1,
                instance_seed=int(instance_seed),
                namespace=SCENE_NAMESPACE,
            )
        ),
        tick_length_px=int(params.get("tick_length_px", scene_default(params, RENDERING_DEFAULTS, "tick_length_px", 8))),
        label_font_size_px=int(params.get("label_font_size_px", scene_default(params, RENDERING_DEFAULTS, "label_font_size_px", 19))),
        tick_font_size_px=int(params.get("tick_font_size_px", scene_default(params, RENDERING_DEFAULTS, "tick_font_size_px", 16))),
        value_font_size_px=int(params.get("value_font_size_px", scene_default(params, RENDERING_DEFAULTS, "value_font_size_px", 14))),
        legend_font_size_px=int(params.get("legend_font_size_px", scene_default(params, RENDERING_DEFAULTS, "legend_font_size_px", 17))),
        label_stroke_width_px=int(params.get("label_stroke_width_px", scene_default(params, RENDERING_DEFAULTS, "label_stroke_width_px", 0))),
        area_outline_width_px=int(
            resolve_render_int(
                params,
                RENDERING_DEFAULTS,
                "area_outline_width_px",
                3,
                instance_seed=int(instance_seed),
                namespace=SCENE_NAMESPACE,
            )
        ),
        point_radius_px=int(
            resolve_render_int(
                params,
                RENDERING_DEFAULTS,
                "point_radius_px",
                6,
                instance_seed=int(instance_seed),
                namespace=SCENE_NAMESPACE,
            )
        ),
        axis_color_rgb=resolve_render_rgb(
            params,
            RENDERING_DEFAULTS,
            "axis_color_rgb",
            (68, 72, 82),
            instance_seed=int(instance_seed),
            namespace=SCENE_NAMESPACE,
        ),
        grid_color_rgb=resolve_render_rgb(
            params,
            RENDERING_DEFAULTS,
            "grid_color_rgb",
            (222, 226, 232),
            instance_seed=int(instance_seed),
            namespace=SCENE_NAMESPACE,
        ),
        plot_fill_rgb=resolve_render_rgb(
            params,
            RENDERING_DEFAULTS,
            "plot_fill_rgb",
            (255, 255, 255),
            instance_seed=int(instance_seed),
            namespace=SCENE_NAMESPACE,
        ),
        text_color_rgb=resolve_render_rgb(
            params,
            RENDERING_DEFAULTS,
            "text_color_rgb",
            (38, 42, 50),
            instance_seed=int(instance_seed),
            namespace=SCENE_NAMESPACE,
        ),
        text_stroke_rgb=resolve_render_rgb(
            params,
            RENDERING_DEFAULTS,
            "text_stroke_rgb",
            (255, 255, 255),
            instance_seed=int(instance_seed),
            namespace=SCENE_NAMESPACE,
        ),
        legend_border_rgb=resolve_render_rgb(
            params,
            RENDERING_DEFAULTS,
            "legend_border_rgb",
            (188, 196, 210),
            instance_seed=int(instance_seed),
            namespace=SCENE_NAMESPACE,
        ),
        layout_jitter_meta=dict(jitter_meta),
    )


def _text_bbox(draw: ImageDraw.ImageDraw, text: str, center: tuple[float, float], font: Any) -> list[float]:
    raw = draw.textbbox((0, 0), str(text), font=font)
    width = float(raw[2] - raw[0])
    height = float(raw[3] - raw[1])
    return round_bbox(
        (
            float(center[0]) - 0.5 * width,
            float(center[1]) - 0.5 * height,
            float(center[0]) + 0.5 * width,
            float(center[1]) + 0.5 * height,
        )
    )


def _axis_max(max_value: int, *, tick_step: int) -> int:
    step = max(1, int(tick_step))
    return max(step, int(math.ceil(float(max_value) / float(step)) * step))


def _y_for_value(value: float, *, plot_top: int, plot_bottom: int, y_axis_max: int) -> float:
    span = float(max(1, int(plot_bottom) - int(plot_top)))
    return float(plot_bottom) - ((float(value) / float(max(1, int(y_axis_max)))) * span)


def _sample_palette(*, instance_seed: int, count: int) -> tuple[tuple[int, int, int], ...]:
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.palette")
    raw_palette = RENDERING_DEFAULTS.get("series_palette_rgb")
    if isinstance(raw_palette, Sequence) and len(raw_palette) >= int(count):
        palette = [
            _as_rgb(item, DEFAULT_PALETTE[index % len(DEFAULT_PALETTE)])
            for index, item in enumerate(raw_palette)
            if isinstance(item, Sequence) and not isinstance(item, (str, bytes))
        ]
        rng.shuffle(palette)
        return tuple(palette[: int(count)])
    palette = sample_color_palette_with_distance_constraints(
        rng,
        palette_size=int(count),
        channel_min=30,
        channel_max=220,
        anchor_colors=((255, 255, 255), (248, 248, 248)),
        min_distance=42.0,
        distance_space="lab",
    )
    return tuple(tuple(int(channel) for channel in color) for color in palette)


def render_area_panel(
    background: Image.Image,
    *,
    x_labels: Sequence[str],
    series_labels: Sequence[str],
    series_values: Mapping[str, Sequence[int]],
    stacked: bool,
    highlight_points: Sequence[tuple[str, str]],
    instance_seed: int,
    params: Mapping[str, Any],
    render_params: AreaRenderParams | None = None,
) -> RenderedAreaPanel:
    """Draw the reusable area-chart panel and record mark projections.

    The renderer owns axes, bands, labels, legends, and point traces for the
    scene; it deliberately does not choose answers or annotation semantics.
    """

    render_params = render_params or resolve_area_render_params(params, instance_seed=int(instance_seed))
    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    plot_left = int(render_params.plot_margin_left_px)
    plot_top = int(render_params.plot_margin_top_px)
    plot_right = int(render_params.canvas_width - render_params.plot_margin_right_px)
    plot_bottom = int(render_params.canvas_height - render_params.plot_margin_bottom_px)
    plot_bbox = (int(plot_left), int(plot_top), int(plot_right), int(plot_bottom))
    draw.rectangle(plot_bbox, fill=render_params.plot_fill_rgb)

    x_label_list = [str(label) for label in x_labels]
    series_label_list = [str(label) for label in series_labels]
    point_count = len(x_label_list)
    if int(point_count) < 2:
        raise ValueError("area chart requires at least two x labels")
    x_edge_pad = float(max(18, int(render_params.label_font_size_px)))
    x_start = float(plot_left) + float(x_edge_pad)
    x_end = float(plot_right) - float(x_edge_pad)
    x_step = float(x_end - x_start) / float(max(1, int(point_count) - 1))
    x_centers = [float(x_start) + float(index) * float(x_step) for index in range(int(point_count))]
    totals_by_x = [
        sum(int(series_values[str(series_label)][index]) for series_label in series_label_list)
        for index in range(int(point_count))
    ]
    max_visible = max(totals_by_x if bool(stacked) else [int(series_values[series_label_list[0]][i]) for i in range(point_count)])
    tick_step = int(params.get("y_axis_tick_step", scene_default(params, RENDERING_DEFAULTS, "y_axis_tick_step", 10)))
    y_axis_max = _axis_max(int(max_visible), tick_step=int(tick_step))
    y_ticks = tuple(range(0, int(y_axis_max) + 1, max(1, int(tick_step))))
    tick_font = load_font(int(render_params.tick_font_size_px), bold=False)
    label_font = load_font(int(render_params.label_font_size_px), bold=True)
    value_font = load_font(int(render_params.value_font_size_px), bold=False)
    legend_font = load_font(int(render_params.legend_font_size_px), bold=True)

    for tick in y_ticks:
        y = _y_for_value(float(tick), plot_top=plot_top, plot_bottom=plot_bottom, y_axis_max=int(y_axis_max))
        draw.line(
            [(float(plot_left), float(y)), (float(plot_right), float(y))],
            fill=render_params.grid_color_rgb,
            width=int(render_params.grid_line_width_px),
        )
        draw_text_centered(
            draw,
            text=str(int(tick)),
            center=(float(plot_left - 28), float(y)),
            font=tick_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=1,
        )
    draw.line(
        [(plot_left, plot_top), (plot_left, plot_bottom), (plot_right, plot_bottom)],
        fill=render_params.axis_color_rgb,
        width=int(render_params.axis_line_width_px),
    )

    palette = _sample_palette(instance_seed=int(instance_seed), count=max(1, len(series_label_list)))
    point_traces: list[dict[str, Any]] = []
    entities: list[dict[str, Any]] = []
    highlight_set = {(str(series), str(label)) for series, label in highlight_points}

    cumulative_lower = [0 for _ in range(int(point_count))]
    for series_index, series_label in enumerate(series_label_list):
        values = [int(value) for value in series_values[str(series_label)]]
        if len(values) != int(point_count):
            raise ValueError("each area series must match x label count")
        lower = list(cumulative_lower) if bool(stacked) else [0 for _ in values]
        upper = [int(low) + int(value) for low, value in zip(lower, values)]
        fill_rgb = tuple(int(channel) for channel in palette[int(series_index) % len(palette)])
        outline_rgb = tuple(max(0, int(channel * 0.58)) for channel in fill_rgb)
        upper_points = [
            (
                float(x_centers[index]),
                _y_for_value(float(upper[index]), plot_top=plot_top, plot_bottom=plot_bottom, y_axis_max=int(y_axis_max)),
            )
            for index in range(int(point_count))
        ]
        lower_points = [
            (
                float(x_centers[index]),
                _y_for_value(float(lower[index]), plot_top=plot_top, plot_bottom=plot_bottom, y_axis_max=int(y_axis_max)),
            )
            for index in range(int(point_count) - 1, -1, -1)
        ]
        draw.polygon(list(upper_points) + list(lower_points), fill=fill_rgb, outline=None)
        if not bool(stacked):
            selected_indices = [
                index
                for index, x_label in enumerate(x_label_list)
                if (str(series_label), str(x_label)) in highlight_set
            ]
            if len(selected_indices) >= 2:
                start_index = min(int(index) for index in selected_indices)
                end_index = max(int(index) for index in selected_indices)
                highlight_upper = [upper_points[index] for index in range(int(start_index), int(end_index) + 1)]
                highlight_lower = [
                    (float(x_centers[index]), float(plot_bottom))
                    for index in range(int(end_index), int(start_index) - 1, -1)
                ]
                overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
                overlay_draw = ImageDraw.Draw(overlay)
                overlay_draw.polygon(
                    list(highlight_upper) + list(highlight_lower),
                    fill=(255, 224, 92, 78),
                )
                boundary_color = (118, 92, 12, 190)
                for boundary_index in (int(start_index), int(end_index)):
                    overlay_draw.line(
                        [
                            (float(x_centers[boundary_index]), float(plot_bottom)),
                            (
                                float(x_centers[boundary_index]),
                                float(
                                    _y_for_value(
                                        float(upper[boundary_index]),
                                        plot_top=plot_top,
                                        plot_bottom=plot_bottom,
                                        y_axis_max=int(y_axis_max),
                                    )
                                ),
                            ),
                        ],
                        fill=boundary_color,
                        width=max(2, int(render_params.area_outline_width_px)),
                    )
                image = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
                draw = ImageDraw.Draw(image)
        draw.line(upper_points, fill=outline_rgb, width=int(render_params.area_outline_width_px), joint="curve")
        if bool(stacked):
            draw.line(
                [
                    (
                        float(x_centers[index]),
                        _y_for_value(float(lower[index]), plot_top=plot_top, plot_bottom=plot_bottom, y_axis_max=int(y_axis_max)),
                    )
                    for index in range(int(point_count))
                ],
                fill=outline_rgb,
                width=max(1, int(render_params.area_outline_width_px) - 1),
                joint="curve",
            )
        for index, x_label in enumerate(x_label_list):
            center_y_value = float(lower[index]) + 0.5 * float(values[index])
            band_center = (
                float(x_centers[index]),
                _y_for_value(center_y_value, plot_top=plot_top, plot_bottom=plot_bottom, y_axis_max=int(y_axis_max)),
            )
            point_y = _y_for_value(float(upper[index]), plot_top=plot_top, plot_bottom=plot_bottom, y_axis_max=int(y_axis_max))
            radius = float(render_params.point_radius_px)
            mark_center = tuple(float(value) for value in band_center)
            value_center = tuple(float(value) for value in band_center)
            point_bbox: list[float] | None = None
            if not bool(stacked):
                mark_center = (float(x_centers[index]), float(point_y))
                value_center = (float(x_centers[index]), float(point_y - 20.0))
                point_bbox = round_bbox(
                    (
                        float(mark_center[0] - radius),
                        float(mark_center[1] - radius),
                        float(mark_center[0] + radius),
                        float(mark_center[1] + radius),
                    )
                )
                draw.ellipse(
                    (
                        float(mark_center[0] - radius),
                        float(mark_center[1] - radius),
                        float(mark_center[0] + radius),
                        float(mark_center[1] + radius),
                    ),
                    fill=render_params.plot_fill_rgb,
                    outline=outline_rgb,
                    width=max(1, int(render_params.area_outline_width_px) - 1),
                )
            text = str(int(values[index]))
            draw_text_centered(
                draw,
                text=text,
                center=value_center,
                font=value_font,
                fill=render_params.text_color_rgb,
                stroke_fill=render_params.text_stroke_rgb,
                stroke_width=int(render_params.label_stroke_width_px),
            )
            text_bbox = _text_bbox(draw, text=text, center=value_center, font=value_font)
            segment_bbox = [
                round(float(x_centers[index] - 0.5 * max(20.0, x_step * 0.55)), 3),
                round(float(_y_for_value(float(upper[index]), plot_top=plot_top, plot_bottom=plot_bottom, y_axis_max=int(y_axis_max))), 3),
                round(float(x_centers[index] + 0.5 * max(20.0, x_step * 0.55)), 3),
                round(float(_y_for_value(float(lower[index]), plot_top=plot_top, plot_bottom=plot_bottom, y_axis_max=int(y_axis_max))), 3),
            ]
            trace = {
                "entity_id": f"area_{series_index}_{index}",
                "series_label": str(series_label),
                "x_label": str(x_label),
                "x_rank": int(index),
                "series_rank": int(series_index),
                "value": int(values[index]),
                "lower_value": int(lower[index]),
                "upper_value": int(upper[index]),
                "mark_center_px": [round(float(mark_center[0]), 3), round(float(mark_center[1]), 3)],
                "mark_bbox_px": list(point_bbox if point_bbox is not None else segment_bbox),
                "value_center_px": [round(float(value_center[0]), 3), round(float(value_center[1]), 3)],
                "value_bbox_px": list(text_bbox),
                "band_segment_bbox_px": list(segment_bbox),
                "queried": bool((str(series_label), str(x_label)) in highlight_set),
                "fill_rgb": [int(channel) for channel in fill_rgb],
                "outline_rgb": [int(channel) for channel in outline_rgb],
            }
            point_traces.append(dict(trace))
            entities.append(
                {
                    "entity_id": str(trace["entity_id"]),
                    "entity_type": "area_band_value" if bool(stacked) else "area_point_value",
                    "attrs": dict(trace),
                }
            )
        cumulative_lower = list(upper)

    for index, x_label in enumerate(x_label_list):
        draw.line(
            [
                (float(x_centers[index]), float(plot_bottom)),
                (float(x_centers[index]), float(plot_bottom + render_params.tick_length_px)),
            ],
            fill=render_params.axis_color_rgb,
            width=max(1, int(render_params.axis_line_width_px)),
        )
        draw_text_centered(
            draw,
            text=str(x_label),
            center=(float(x_centers[index]), float(plot_bottom + render_params.tick_length_px + render_params.label_font_size_px)),
            font=label_font,
            fill=render_params.text_color_rgb,
            stroke_fill=render_params.text_stroke_rgb,
            stroke_width=int(render_params.label_stroke_width_px),
        )

    legend_traces: list[dict[str, Any]] = []
    legend_left = float(plot_right + 24)
    legend_top = float(plot_top + 18)
    swatch = float(max(18, int(render_params.legend_font_size_px)))
    row_h = float(max(34, int(render_params.legend_font_size_px) + 16))
    if bool(stacked):
        legend_width = float(max(130.0, render_params.canvas_width - legend_left - 24.0))
        legend_height = float(len(series_label_list) * row_h + 16.0)
        legend_bbox = (legend_left - 10.0, legend_top - 10.0, legend_left + legend_width, legend_top + legend_height)
        draw.rounded_rectangle(
            legend_bbox,
            radius=6,
            fill=render_params.plot_fill_rgb,
            outline=render_params.legend_border_rgb,
            width=1,
        )
        for series_index, series_label in enumerate(series_label_list):
            row_y = legend_top + series_index * row_h
            fill_rgb = tuple(int(channel) for channel in palette[int(series_index) % len(palette)])
            swatch_bbox = (legend_left, row_y, legend_left + swatch, row_y + swatch)
            draw.rectangle(
                swatch_bbox,
                fill=fill_rgb,
                outline=tuple(max(0, int(channel * 0.58)) for channel in fill_rgb),
                width=2,
            )
            text_center = (legend_left + swatch + 12 + 52, row_y + swatch * 0.5)
            draw_text_traced(
                draw,
                (legend_left + swatch + 12, row_y - 1),
                str(series_label),
                font=legend_font,
                fill=render_params.text_color_rgb,
                stroke_width=1,
                stroke_fill=render_params.text_stroke_rgb,
                role="readout",
                required=False,
            )
            text_bbox = draw.textbbox((legend_left + swatch + 12, row_y - 1), str(series_label), font=legend_font)
            legend_trace = {
                "entity_id": f"legend_{series_index}",
                "series_label": str(series_label),
                "series_rank": int(series_index),
                "swatch_bbox_px": round_bbox(swatch_bbox),
                "label_center_px": [round(float(text_center[0]), 3), round(float(text_center[1]), 3)],
                "label_bbox_px": round_bbox(text_bbox),
                "fill_rgb": [int(channel) for channel in fill_rgb],
            }
            legend_traces.append(dict(legend_trace))
            entities.append(
                {
                    "entity_id": str(legend_trace["entity_id"]),
                    "entity_type": "legend_entry",
                    "attrs": dict(legend_trace),
                }
            )

    return RenderedAreaPanel(
        image=image,
        plot_bbox_px=tuple(int(value) for value in plot_bbox),
        y_axis_max=int(y_axis_max),
        y_ticks=tuple(int(value) for value in y_ticks),
        entities=tuple(dict(entity) for entity in entities),
        point_traces=tuple(dict(trace) for trace in point_traces),
        legend_traces=tuple(dict(trace) for trace in legend_traces),
    )


def render_area_scene(
    *,
    x_labels: Sequence[str],
    series_labels: Sequence[str],
    series_values: Mapping[str, Sequence[int]],
    stacked: bool,
    highlight_points: Sequence[tuple[str, str]],
    params: Mapping[str, Any],
    instance_seed: int,
) -> AreaRenderResult:
    """Assemble shared information style, font, panel rendering, and post-noise.

    This scene-level helper returns rendered geometry for task-owned answer and
    annotation binding without branching on task or query identity.
    """

    information_style, information_style_meta = resolve_chart_information_style(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        protected_colors=_semantic_palette(params),
    )
    render_style_params = _params_with_information_style(params, information_style)
    render_params = resolve_area_render_params(render_style_params, instance_seed=int(instance_seed))
    background, background_meta = make_chart_information_background(
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
        style=information_style,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.information_scene_background",
    )
    chart_font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.chart_font",
        params=params,
        explicit_key="chart_font_family",
        weights_key="chart_font_family_weights",
    )
    with temporary_default_font_family(str(chart_font_family)):
        panel = render_area_panel(
            background,
            x_labels=x_labels,
            series_labels=series_labels,
            series_values=series_values,
            stacked=bool(stacked),
            highlight_points=highlight_points,
            instance_seed=int(instance_seed),
            params=render_style_params,
            render_params=render_params,
        )
    image, post_noise_meta = apply_post_image_noise(
        panel.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return AreaRenderResult(
        image=image,
        panel=panel,
        render_params=render_params,
        information_style_meta=dict(information_style_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        chart_font_family=str(chart_font_family),
    )


def render_trace_sections(
    *,
    rendered: AreaRenderResult,
    scene_variant: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Export renderer metadata and mark maps for task trace payloads.

    The trace export describes scene geometry, font/style choices, and rendered
    marks; objective fields remain owned by public task modules.
    """

    render_params = rendered.render_params
    panel = rendered.panel
    font_assets = chart_font_asset_metadata(str(rendered.chart_font_family))
    render_spec = {
        "canvas_width": int(rendered.image.size[0]),
        "canvas_height": int(rendered.image.size[1]),
        "coord_space": "pixel",
        "scene_id": SCENE_ID,
        "scene_variant": str(scene_variant),
        "information_scene_style": dict(rendered.information_style_meta),
        "background_style": dict(rendered.background_meta),
        "post_image_noise": dict(rendered.post_noise_meta),
        "plot_bbox_px": list(panel.plot_bbox_px),
        "y_axis_max": int(panel.y_axis_max),
        "y_ticks": [int(tick) for tick in panel.y_ticks],
        "layout_jitter": dict(render_params.layout_jitter_meta),
        "text_style": {
            "label_font_size_px": int(render_params.label_font_size_px),
            "tick_font_size_px": int(render_params.tick_font_size_px),
            "value_font_size_px": int(render_params.value_font_size_px),
            "legend_font_size_px": int(render_params.legend_font_size_px),
            "label_stroke_width_px": int(render_params.label_stroke_width_px),
            "font_asset_version": str(font_asset_version()),
            "chart_font_family": str(rendered.chart_font_family),
            "chart_font_exclude_tags": [],
        },
        "font_assets": dict(font_assets),
        "axis_style": {
            "axis_line_width_px": int(render_params.axis_line_width_px),
            "grid_line_width_px": int(render_params.grid_line_width_px),
            "tick_length_px": int(render_params.tick_length_px),
        },
    }
    render_map = {
        "image_id": "img0",
        "plot_bbox_px": list(panel.plot_bbox_px),
        "point_traces": [dict(trace) for trace in panel.point_traces],
        "legend_traces": [dict(trace) for trace in panel.legend_traces],
    }
    return render_spec, render_map
