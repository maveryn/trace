"""Low-level chart scene rendering primitives."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageChops, ImageDraw

from ....core.sampling import uniform_choice
from ....core.seed import spawn_rng
from .chart_scene_types import ChartColor, ChartRenderParams, RenderedChartScene


def _text_bbox(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font,
) -> Tuple[float, float, float, float]:
    """Return one text bbox centered around `center`."""

    try:
        raw = draw.textbbox((0, 0), str(text), font=font)
        width = float(raw[2] - raw[0])
        height = float(raw[3] - raw[1])
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        width = float(width)
        height = float(height)
    center_x = float(center[0])
    center_y = float(center[1])
    return (
        float(center_x - (0.5 * width)),
        float(center_y - (0.5 * height)),
        float(center_x + (0.5 * width)),
        float(center_y + (0.5 * height)),
    )


def _text_size(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    font,
) -> Tuple[float, float]:
    """Return one text width/height pair in pixels."""

    try:
        raw = draw.textbbox((0, 0), str(text), font=font)
        return (float(raw[2] - raw[0]), float(raw[3] - raw[1]))
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        return (float(width), float(height))


def _resolve_plot_bbox(params: ChartRenderParams) -> Tuple[int, int, int, int]:
    """Resolve one axis-aligned plot bbox within the chart canvas."""

    left = int(params.plot_margin_left_px)
    top = int(params.plot_margin_top_px)
    right = int(params.canvas_width) - int(params.plot_margin_right_px)
    bottom = int(params.canvas_height) - int(params.plot_margin_bottom_px)
    if right <= left or bottom <= top:
        raise ValueError("chart plot margins leave no drawable plot area")
    return (int(left), int(top), int(right), int(bottom))


def _tick_y(
    value: int,
    *,
    y_axis_min: int = 0,
    y_axis_max: int,
    plot_top: int,
    plot_bottom: int,
) -> float:
    """Map one integer chart value to plot-space pixel y coordinate."""

    span = max(1, int(y_axis_max) - int(y_axis_min))
    plot_height = float(max(1, int(plot_bottom) - int(plot_top)))
    return float(plot_bottom) - ((float(value) - float(y_axis_min)) / float(span)) * float(plot_height)


def _tick_x(
    value: int,
    *,
    x_axis_min: int = 0,
    x_axis_max: int,
    plot_left: int,
    plot_right: int,
) -> float:
    """Map one integer chart value to plot-space pixel x coordinate."""

    span = max(1, int(x_axis_max) - int(x_axis_min))
    plot_width = float(max(1, int(plot_right) - int(plot_left)))
    return float(plot_left) + ((float(value) - float(x_axis_min)) / float(span)) * float(plot_width)


def _axis_ticks(axis_max: int, *, axis_min: int = 0, step: int | None = None) -> Tuple[int, ...]:
    """Return readable integer ticks for a chart axis."""

    min_value = int(axis_min)
    max_value = max(1, int(axis_max))
    if int(max_value) < int(min_value):
        min_value, max_value = int(max_value), int(min_value)
    if step is None:
        span = int(max_value) - int(min_value)
        if int(span) <= 25:
            return tuple(range(int(min_value), int(max_value) + 1))
        target_step = max(1, int(math.ceil(float(span) / 10.0)))
        nice_steps = (2, 5, 10, 20, 25, 50, 100)
        resolved_step = next((candidate for candidate in nice_steps if int(candidate) >= int(target_step)), int(target_step))
    else:
        resolved_step = max(1, int(step))
    ticks = [int(value) for value in range(int(min_value), int(max_value) + 1, int(resolved_step))]
    if not ticks or ticks[0] != int(min_value):
        ticks.insert(0, int(min_value))
    if ticks[-1] != int(max_value):
        ticks.append(int(max_value))
    return tuple(int(value) for value in ticks)


def _resolve_value_axis(
    values: Sequence[int],
    *,
    render_params: ChartRenderParams,
) -> Tuple[int, int, Tuple[int, ...], Tuple[int, ...], bool]:
    """Resolve one readable value-axis window for axis-based charts."""

    if not values:
        return 0, 4, _axis_ticks(4), _axis_ticks(4), False
    data_min = int(min(int(value) for value in values))
    data_max = int(max(int(value) for value in values))
    if not bool(render_params.value_axis_window_enabled):
        axis_min = 0
        axis_max = max(4, int(data_max) + 1)
        ticks = _axis_ticks(axis_max)
        return int(axis_min), int(axis_max), ticks, ticks, False

    hard_max = int(max(1, render_params.value_axis_hard_max))
    major_step = max(1, int(render_params.value_axis_major_tick_step))
    minor_step = max(1, int(render_params.value_axis_minor_tick_step))
    span_min = max(1, int(render_params.value_axis_span_min))
    span_max = max(int(span_min), int(render_params.value_axis_span_max))

    if bool(render_params.value_axis_allow_nonzero_min):
        axis_min = int(math.floor(float(data_min) / float(major_step)) * int(major_step))
        if int(data_max) - int(axis_min) > int(span_max) and int(data_max) - int(data_min) <= int(span_max):
            axis_min = int(data_max) - int(span_max)
    else:
        axis_min = 0
    axis_min = max(0, int(axis_min))

    required_span = max(int(span_min), int(data_max) - int(axis_min))
    if int(required_span) <= int(span_max):
        axis_span = int(span_max)
    else:
        axis_span = int(required_span)
    axis_max = int(axis_min) + int(axis_span)

    if int(axis_max) > int(hard_max):
        axis_max = int(hard_max)
        axis_min = max(0, int(axis_max) - int(axis_span))
        if int(data_min) < int(axis_min):
            axis_min = int(data_min)
        if int(data_max) > int(axis_max):
            axis_max = int(data_max)

    if int(axis_min) == int(axis_max):
        axis_max = int(axis_min) + 1
    major_ticks = _axis_ticks(int(axis_max), axis_min=int(axis_min), step=int(major_step))
    minor_ticks = _axis_ticks(int(axis_max), axis_min=int(axis_min), step=int(minor_step))
    return int(axis_min), int(axis_max), major_ticks, minor_ticks, True


def _draw_styled_line(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Tuple[float, float]],
    *,
    fill: ChartColor,
    width: int,
    style: str,
) -> None:
    """Draw a solid, dashed, or dotted line segment."""

    if len(points) < 2:
        return
    x0, y0 = float(points[0][0]), float(points[0][1])
    x1, y1 = float(points[1][0]), float(points[1][1])
    resolved_width = max(1, int(width))
    resolved_style = str(style).strip().lower()
    if resolved_style == "solid":
        draw.line([(x0, y0), (x1, y1)], fill=fill, width=resolved_width)
        return

    dx = float(x1 - x0)
    dy = float(y1 - y0)
    length = math.hypot(dx, dy)
    if length <= 0.0:
        return
    ux = dx / length
    uy = dy / length
    if resolved_style == "dotted":
        gap = max(4.0, float(resolved_width) * 4.0)
        radius = max(1.0, float(resolved_width))
        distance = 0.0
        while distance <= length:
            cx = x0 + ux * distance
            cy = y0 + uy * distance
            draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=fill)
            distance += gap
        return

    dash = max(8.0, float(resolved_width) * 8.0)
    gap = max(5.0, float(resolved_width) * 5.0)
    distance = 0.0
    while distance < length:
        end = min(length, distance + dash)
        draw.line(
            [
                (x0 + ux * distance, y0 + uy * distance),
                (x0 + ux * end, y0 + uy * end),
            ],
            fill=fill,
            width=resolved_width,
        )
        distance += dash + gap


def _guide_lines_enabled(render_params: ChartRenderParams) -> bool:
    """Return whether value guide lines should be drawn for this render."""

    return str(render_params.guide_line_mode).strip().lower() in {"always", "variant"}


def value_axis_render_metadata(rendered_scene: RenderedChartScene) -> Dict[str, Any]:
    """Return trace/render metadata for the chart value axis."""

    return {
        "value_axis_min": int(rendered_scene.value_axis_min),
        "value_axis_max": int(rendered_scene.value_axis_max),
        "value_axis_span": int(rendered_scene.value_axis_span),
        "value_axis_major_ticks": [int(value) for value in rendered_scene.value_axis_major_ticks],
        "value_axis_minor_ticks": [int(value) for value in rendered_scene.value_axis_minor_ticks],
        "value_axis_window_enabled": bool(rendered_scene.value_axis_window_enabled),
        "guide_line_style": str(rendered_scene.guide_line_style),
        "guide_lines": [dict(line) for line in rendered_scene.guide_lines],
    }


def _label_center_for_variant(
    *,
    scene_variant: str,
    mark_center: Tuple[float, float],
    bar_bbox: Tuple[float, float, float, float] | None,
    point_radius_px: int,
    label_font_size_px: int,
) -> Tuple[float, float]:
    """Resolve one readable label center for the selected chart type."""

    center_x = float(mark_center[0])
    center_y = float(mark_center[1])
    if str(scene_variant) == "bar":
        if bar_bbox is None:
            raise ValueError("bar chart labels require bar bbox")
        return (
            float(center_x),
            float(bar_bbox[3]) + float(max(16, int(label_font_size_px) + 4)),
        )
    if str(scene_variant) == "horizontal_bar":
        if bar_bbox is None:
            raise ValueError("horizontal_bar labels require bar bbox")
        return (
            float(bar_bbox[0]) - float(max(20, int(label_font_size_px) + 8)),
            float(center_y),
        )
    return (
        float(center_x),
        float(center_y) - float(max(int(point_radius_px) + 14, int(label_font_size_px))),
    )


def _scatter_slot_centers(
    rng,
    *,
    count: int,
    plot_left: int,
    plot_right: int,
) -> Tuple[float, ...]:
    """Return stable per-point scatter x coordinates with bounded jitter."""

    usable_width = float(max(1, int(plot_right) - int(plot_left)))
    slot_width = float(usable_width / max(1, int(count)))
    centers: List[float] = []
    for index in range(int(count)):
        slot_left = float(plot_left) + float(index) * float(slot_width)
        slot_center = float(slot_left) + 0.5 * float(slot_width)
        jitter = float(rng.uniform(-0.18, 0.18)) * float(slot_width)
        centers.append(float(slot_center + jitter))
    return tuple(float(value) for value in centers)


def _slot_centers(*, count: int, plot_left: int, plot_right: int) -> Tuple[float, ...]:
    """Return equally spaced x-slot centers for ordered chart marks."""

    usable_width = float(max(1, int(plot_right) - int(plot_left)))
    slot_width = float(usable_width / max(1, int(count)))
    return tuple(
        float(plot_left) + (float(index) + 0.5) * float(slot_width)
        for index in range(int(count))
    )


def _axis_slot_centers(*, count: int, start_px: int, end_px: int) -> Tuple[float, ...]:
    """Return equally spaced slot centers along one arbitrary axis span."""

    usable_span = float(max(1, int(end_px) - int(start_px)))
    slot_width = float(usable_span / max(1, int(count)))
    return tuple(
        float(start_px) + (float(index) + 0.5) * float(slot_width)
        for index in range(int(count))
    )


def _clamp_text_center_to_canvas(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font,
    canvas_width: int,
    canvas_height: int,
    margin_px: float = 4.0,
) -> Tuple[float, float]:
    """Clamp one centered text label so its bbox stays inside the image canvas."""

    bbox = _text_bbox(draw, text=str(text), center=center, font=font)
    margin = float(max(0.0, float(margin_px)))
    dx = 0.0
    dy = 0.0
    if float(bbox[0]) < float(margin):
        dx = float(margin) - float(bbox[0])
    elif float(bbox[2]) > float(canvas_width) - float(margin):
        dx = (float(canvas_width) - float(margin)) - float(bbox[2])
    if float(bbox[1]) < float(margin):
        dy = float(margin) - float(bbox[1])
    elif float(bbox[3]) > float(canvas_height) - float(margin):
        dy = (float(canvas_height) - float(margin)) - float(bbox[3])
    return (float(center[0]) + float(dx), float(center[1]) + float(dy))


def _bbox_from_points(points: Sequence[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    """Return the axis-aligned bbox enclosing one point sequence."""

    if not points:
        raise ValueError("bbox requires at least one point")
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


def _union_bboxes(
    bboxes: Sequence[Tuple[float, float, float, float]],
) -> Tuple[float, float, float, float]:
    """Return one enclosing bbox for a non-empty bbox list."""

    if not bboxes:
        raise ValueError("union requires at least one bbox")
    return _bbox_from_points(
        [
            (float(x0), float(y0))
            for x0, y0, _, _ in bboxes
        ]
        + [
            (float(x1), float(y1))
            for _, _, x1, y1 in bboxes
        ]
    )


def _radar_polygon_points(
    *,
    center_x: float,
    center_y: float,
    radius: float,
    angles_rad: Sequence[float],
) -> List[Tuple[float, float]]:
    """Return polygon vertices for one radar ring or data polygon."""

    return [
        (
            float(center_x + (float(radius) * math.cos(float(angle)))),
            float(center_y + (float(radius) * math.sin(float(angle)))),
        )
        for angle in angles_rad
    ]


def _normalize_color(value: Sequence[int], fallback: ChartColor) -> ChartColor:
    """Normalize RGB-like input into a clamped 3-channel tuple."""

    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (
            max(0, min(255, int(value[0]))),
            max(0, min(255, int(value[1]))),
            max(0, min(255, int(value[2]))),
        )
    return tuple(int(channel) for channel in fallback)


def _blend_rgb(foreground: ChartColor, background: ChartColor, foreground_weight: float) -> ChartColor:
    """Blend two RGB colors with a clamped foreground weight."""

    weight = max(0.0, min(1.0, float(foreground_weight)))
    return tuple(
        int(round((weight * float(fg)) + ((1.0 - weight) * float(bg))))
        for fg, bg in zip(foreground, background)
    )


def _darken_rgb(color: ChartColor, factor: float = 0.58) -> ChartColor:
    """Return a darker RGB color for outlines/hatching."""

    scale = max(0.0, min(1.0, float(factor)))
    return tuple(max(0, min(255, int(round(float(channel) * scale)))) for channel in color)


def _violin_palette_color(
    base_fill: ChartColor,
    *,
    index: int,
    offset: int,
    plot_fill: ChartColor,
) -> ChartColor:
    """Return one muted per-violin color while preserving readable contrast."""

    muted_palette: Tuple[ChartColor, ...] = (
        (92, 133, 196),
        (193, 111, 86),
        (94, 155, 116),
        (170, 126, 190),
        (196, 154, 77),
        (90, 154, 168),
        (184, 103, 135),
        (128, 139, 92),
    )
    palette_color = muted_palette[(int(index) + int(offset)) % len(muted_palette)]
    return _blend_rgb(palette_color, base_fill, 0.82) if base_fill else _blend_rgb(palette_color, plot_fill, 0.90)


def _draw_violin_polygon(
    image: Image.Image,
    *,
    points: Sequence[Tuple[float, float]],
    fill_rgb: ChartColor,
    outline_rgb: ChartColor,
    fill_style: str,
    outline_width: int,
    plot_fill_rgb: ChartColor,
) -> None:
    """Draw one violin body with optional lightweight fill styling."""

    style = str(fill_style).strip().lower()
    draw = ImageDraw.Draw(image)
    if style == "light":
        body_fill = _blend_rgb(fill_rgb, plot_fill_rgb, 0.58)
    elif style == "outline":
        body_fill = _blend_rgb(fill_rgb, plot_fill_rgb, 0.28)
    elif style == "hatch":
        body_fill = _blend_rgb(fill_rgb, plot_fill_rgb, 0.34)
    else:
        body_fill = fill_rgb

    draw.polygon(points, fill=body_fill)
    if style == "hatch":
        mask = Image.new("L", image.size, 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.polygon(points, fill=255)
        hatch_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
        hatch_draw = ImageDraw.Draw(hatch_layer)
        x0, y0, x1, y1 = [int(round(value)) for value in _bbox_from_points(points)]
        hatch_color = _darken_rgb(outline_rgb, 0.86)
        spacing = max(9, int(round(0.75 * float(outline_width + 10))))
        for start_x in range(x0 - (y1 - y0) - spacing, x1 + spacing, spacing):
            hatch_draw.line(
                [(start_x, y1 + spacing), (start_x + (y1 - y0) + spacing, y0 - spacing)],
                fill=(*hatch_color, 95),
                width=max(1, int(outline_width)),
            )
        hatch_alpha = ImageChops.multiply(hatch_layer.getchannel("A"), mask)
        hatch_layer.putalpha(hatch_alpha)
        composited = Image.alpha_composite(image.convert("RGBA"), hatch_layer)
        image.paste(composited.convert("RGB"))
        draw = ImageDraw.Draw(image)

    draw.line(
        list(points) + [tuple(points[0])],
        fill=outline_rgb,
        width=max(1, int(outline_width)),
        joint="curve",
    )


def resolve_chart_render_params(params: Mapping[str, Any]) -> ChartRenderParams:
    """Resolve one chart render-parameter block from config-like values."""

    def _bool_value(key: str, fallback: bool) -> bool:
        value = params.get(str(key), bool(fallback))
        if isinstance(value, str):
            return str(value).strip().lower() in {"1", "true", "yes", "on", "always"}
        return bool(value)

    def _style_value() -> str:
        explicit = params.get("guide_line_style")
        if explicit is not None:
            return str(explicit)
        styles = params.get("guide_line_styles", ())
        if isinstance(styles, Sequence) and styles and not isinstance(styles, (str, bytes)):
            seed = int(params.get("_guide_style_seed", 0))
            return str(
                uniform_choice(
                    spawn_rng(int(seed), "charts.shared.guide_line_style"),
                    tuple(str(style) for style in styles),
                )
            )
        return "dashed"

    return ChartRenderParams(
        canvas_width=int(params["canvas_width"]),
        canvas_height=int(params["canvas_height"]),
        plot_margin_left_px=int(params["plot_margin_left_px"]),
        plot_margin_right_px=int(params["plot_margin_right_px"]),
        plot_margin_top_px=int(params["plot_margin_top_px"]),
        plot_margin_bottom_px=int(params["plot_margin_bottom_px"]),
        axis_line_width_px=int(params["axis_line_width_px"]),
        grid_line_width_px=int(params["grid_line_width_px"]),
        tick_length_px=int(params["tick_length_px"]),
        label_font_size_px=int(params["label_font_size_px"]),
        tick_font_size_px=int(params["tick_font_size_px"]),
        label_stroke_width_px=int(params["label_stroke_width_px"]),
        label_bold=_bool_value("label_bold", True),
        mark_outline_width_px=int(params["mark_outline_width_px"]),
        line_width_px=int(params["line_width_px"]),
        point_radius_px=int(params["point_radius_px"]),
        bar_width_fraction=float(params["bar_width_fraction"]),
        axis_color_rgb=_normalize_color(params.get("axis_color_rgb", (74, 78, 86)), (74, 78, 86)),
        grid_color_rgb=_normalize_color(params.get("grid_color_rgb", (224, 227, 232)), (224, 227, 232)),
        mark_fill_rgb=_normalize_color(params.get("mark_fill_rgb", (86, 138, 214)), (86, 138, 214)),
        mark_outline_rgb=_normalize_color(params.get("mark_outline_rgb", (50, 76, 116)), (50, 76, 116)),
        text_color_rgb=_normalize_color(params.get("text_color_rgb", (38, 41, 48)), (38, 41, 48)),
        text_stroke_rgb=_normalize_color(params.get("text_stroke_rgb", (255, 255, 255)), (255, 255, 255)),
        plot_fill_rgb=_normalize_color(params.get("plot_fill_rgb", (255, 255, 255)), (255, 255, 255)),
        value_axis_window_enabled=_bool_value("value_axis_window_enabled", False),
        value_axis_span_min=int(params.get("value_axis_span_min", 10)),
        value_axis_span_max=int(params.get("value_axis_span_max", 25)),
        value_axis_hard_max=int(params.get("value_axis_hard_max", 99)),
        value_axis_major_tick_step=int(params.get("value_axis_major_tick_step", 5)),
        value_axis_minor_tick_step=int(params.get("value_axis_minor_tick_step", 1)),
        value_axis_allow_nonzero_min=_bool_value("value_axis_allow_nonzero_min", True),
        guide_line_mode=str(params.get("guide_line_mode", "off")),
        guide_line_prob=float(params.get("guide_line_prob", 0.0)),
        guide_line_style=str(_style_value()),
        guide_line_width_px=int(params.get("guide_line_width_px", 1)),
        guide_line_color_rgb=_normalize_color(params.get("guide_line_color_rgb", (150, 156, 166)), (150, 156, 166)),
        layout_jitter_px=(
            int(params.get("layout_jitter_dx_px", 0)),
            int(params.get("layout_jitter_dy_px", 0)),
        ),
        layout_jitter_meta=dict(params.get("layout_jitter_meta", {}))
        if isinstance(params.get("layout_jitter_meta", {}), dict)
        else None,
        violin_mode_line_style=str(params.get("violin_mode_line_style", "full")),
        violin_fill_style=str(params.get("violin_fill_style", "solid")),
        violin_width_scale=float(params.get("violin_width_scale", 1.0)),
        violin_smoothing_scale=float(params.get("violin_smoothing_scale", 1.0)),
        violin_palette_mode=str(params.get("violin_palette_mode", "single")),
        violin_palette_offset=int(params.get("violin_palette_offset", 0)),
    )


__all__ = [
    "resolve_chart_render_params",
    "value_axis_render_metadata",
]
