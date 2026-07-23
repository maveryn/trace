"""Rendering helpers for curve-panel chart tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw, ImageFont

from trace_tasks.tasks.charts.shared.cartesian.geometry import project_linear, project_linear_inverted, round_bbox
from trace_tasks.tasks.charts.shared.cartesian.lines import draw_dashed_line as draw_cartesian_dashed_line
from trace_tasks.tasks.charts.shared.panel.grid_layout import layout_panel_grid_list
from trace_tasks.tasks.charts.shared.information_style import (
    make_chart_information_background,
    resolve_chart_information_style,
)
from .....core.visual.noise import apply_post_image_noise
from ....shared.render_variation import apply_layout_jitter_to_margins
from ....shared.text_legibility import draw_text_traced
from ....shared.text_rendering import load_font
from .defaults import (
    POST_IMAGE_NOISE_DEFAULTS,
    SCENE_NAMESPACE,
    RENDER_DEFAULTS,
    resolve_int,
    resolve_rgb,
)
from .sampling import point_id
from .state import (
    BBox,
    RGB,
    CurvePanelDataset,
    Panel,
    RenderedCurvePanels,
)


def _bbox(values: Sequence[float]) -> List[float]:
    return round_bbox(values)


def _text_bbox(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    *,
    stroke_width: int = 0,
) -> List[float]:
    try:
        box = draw.textbbox(
            (float(xy[0]), float(xy[1])),
            str(text),
            font=font,
            stroke_width=max(0, int(stroke_width)),
        )
        return _bbox(box)
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        return _bbox(
            [
                float(xy[0]),
                float(xy[1]),
                float(xy[0]) + float(width),
                float(xy[1]) + float(height),
            ]
        )


def _center_text(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    text: str,
    font: ImageFont.ImageFont,
    fill: RGB,
    stroke_fill: RGB = (255, 255, 255),
    stroke_width: int = 0,
) -> List[float]:
    """Draw centered text and return the actual bbox used by trace metadata."""

    try:
        box = draw.textbbox(
            (0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width))
        )
        width = float(box[2] - box[0])
        height = float(box[3] - box[1])
        x = float(center[0]) - (0.5 * width) - float(box[0])
        y = float(center[1]) - (0.5 * height) - float(box[1])
    except Exception:
        width, height = draw.textsize(str(text), font=font)
        x = float(center[0]) - (0.5 * float(width))
        y = float(center[1]) - (0.5 * float(height))
    draw_text_traced(
        draw,
        (float(x), float(y)),
        str(text),
        font=font,
        fill=fill,
        stroke_fill=stroke_fill,
        stroke_width=max(0, int(stroke_width)),
        role="readout",
        required=False,
    )
    return _text_bbox(
        draw,
        (float(x), float(y)),
        str(text),
        font,
        stroke_width=max(0, int(stroke_width)),
    )


def _panel_layout(plot_bbox: BBox, panel_count: int, gap: float) -> List[BBox]:
    return layout_panel_grid_list(
        plot_bbox,
        panel_count=int(panel_count),
        gap_x=float(gap),
        gap_y=float(gap),
    )


def _scale_point(
    *,
    x_value: float,
    y_value: float,
    x_values: Sequence[int],
    y_min: int,
    y_max: int,
    plot_bbox: BBox,
) -> Tuple[float, float]:
    x1, y1, x2, y2 = (float(value) for value in plot_bbox)
    x_min = float(min(x_values))
    x_max = float(max(x_values))
    return (
        project_linear(
            float(x_value),
            domain_min=float(x_min),
            domain_max=float(x_max),
            pixel_min=float(x1),
            pixel_max=float(x2),
        ),
        project_linear_inverted(
            float(y_value),
            domain_min=float(y_min),
            domain_max=float(y_max),
            pixel_top=float(y1),
            pixel_bottom=float(y2),
        ),
    )


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    *,
    xy0: Tuple[float, float],
    xy1: Tuple[float, float],
    fill: RGB,
    width: int,
    dash_px: float = 8.0,
    gap_px: float = 6.0,
) -> None:
    draw_cartesian_dashed_line(
        draw,
        (float(xy0[0]), float(xy0[1])),
        (float(xy1[0]), float(xy1[1])),
        fill=fill,
        width=int(width),
        dash_px=float(dash_px),
        gap_px=float(gap_px),
    )


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    *,
    method_labels: Sequence[str],
    colors: Sequence[RGB],
    params: Mapping[str, Any],
    origin: Tuple[float, float],
) -> Dict[str, List[float]]:
    font = load_font(resolve_int(params, "legend_font_size_px", 17), bold=True)
    text_rgb = resolve_rgb(params, "text_color_rgb", (36, 42, 54))
    bboxes: Dict[str, List[float]] = {}
    x = float(origin[0])
    y = float(origin[1])
    for index, method in enumerate(method_labels):
        color = tuple(colors[int(index) % len(colors)])
        swatch = [x, y + 5.0, x + 24.0, y + 18.0]
        draw.rounded_rectangle(
            swatch, radius=3, fill=color, outline=(255, 255, 255), width=1
        )
        text_xy = (x + 31.0, y)
        draw_text_traced(
            draw,
            text_xy,
            str(method),
            font=font,
            fill=text_rgb,
            role="readout",
            required=False,
        )
        text_box = _text_bbox(draw, text_xy, str(method), font)
        bboxes[str(method)] = _bbox([swatch[0], swatch[1], text_box[2], text_box[3]])
        x = float(text_box[2]) + 28.0
    return bboxes


def _draw_panel(
    draw: ImageDraw.ImageDraw,
    *,
    panel: Panel,
    panel_bbox: BBox,
    dataset: CurvePanelDataset,
    params: Mapping[str, Any],
) -> Tuple[
    List[Dict[str, Any]], Dict[str, List[float]], Dict[str, List[float]], List[float]
]:
    """Render one subplot and register every curve mark in shared pixel space."""

    x1, y1, x2, y2 = (float(value) for value in panel_bbox)
    panel_fill = resolve_rgb(params, "panel_fill_rgb", (255, 255, 255))
    panel_border = resolve_rgb(params, "panel_border_rgb", (190, 199, 212))
    axis_rgb = resolve_rgb(params, "axis_color_rgb", (68, 72, 82))
    grid_rgb = resolve_rgb(params, "grid_color_rgb", (225, 229, 235))
    text_rgb = resolve_rgb(params, "text_color_rgb", (36, 42, 54))
    muted_rgb = resolve_rgb(params, "muted_text_rgb", (91, 102, 120))
    text_stroke = resolve_rgb(params, "text_stroke_rgb", (255, 255, 255))
    threshold_rgb = resolve_rgb(params, "threshold_rgb", (178, 74, 74))
    panel_title_font = load_font(
        resolve_int(params, "panel_title_font_size_px", 18), bold=True
    )
    tick_font = load_font(resolve_int(params, "tick_font_size_px", 12), bold=False)

    draw.rounded_rectangle(
        [x1, y1, x2, y2],
        radius=resolve_int(params, "panel_corner_radius_px", 6),
        fill=panel_fill,
        outline=panel_border,
        width=resolve_int(params, "panel_border_width_px", 2),
    )
    title_text = str(panel.panel_label)
    title_xy = (x1 + 12.0, y1 + 7.0)
    draw_text_traced(
        draw,
        title_xy,
        title_text,
        font=panel_title_font,
        fill=text_rgb,
        role="readout",
        required=False,
    )

    left_pad = 48.0
    right_pad = 18.0
    top_pad = 44.0
    bottom_pad = 42.0
    plot_bbox = (x1 + left_pad, y1 + top_pad, x2 - right_pad, y2 - bottom_pad)
    px1, py1, px2, py2 = plot_bbox
    draw.rectangle(
        [px1, py1, px2, py2], fill=(255, 255, 255), outline=grid_rgb, width=1
    )

    y_ticks = [0, 25, 50, 75, 100]
    for tick in y_ticks:
        sx, sy = _scale_point(
            x_value=dataset.x_values[0],
            y_value=float(tick),
            x_values=dataset.x_values,
            y_min=dataset.y_min,
            y_max=dataset.y_max,
            plot_bbox=plot_bbox,
        )
        draw.line(
            [px1, sy, px2, sy],
            fill=grid_rgb,
            width=resolve_int(params, "grid_line_width_px", 1),
        )
        draw_text_traced(
            draw,
            (px1 - 34.0, sy - 7.0),
            str(tick),
            font=tick_font,
            fill=muted_rgb,
            role="readout",
            required=False,
        )

    tick_stride = max(1, int(math.ceil(float(len(dataset.x_values)) / 6.0)))
    x_ticks_to_draw = list(dataset.x_values[::tick_stride])
    if dataset.x_values[-1] not in x_ticks_to_draw:
        x_ticks_to_draw.append(int(dataset.x_values[-1]))
    for x_tick in x_ticks_to_draw:
        sx, _ = _scale_point(
            x_value=float(x_tick),
            y_value=float(dataset.y_min),
            x_values=dataset.x_values,
            y_min=dataset.y_min,
            y_max=dataset.y_max,
            plot_bbox=plot_bbox,
        )
        draw.line(
            [sx, py1, sx, py2],
            fill=grid_rgb,
            width=resolve_int(params, "grid_line_width_px", 1),
        )
        _center_text(
            draw,
            center=(sx, py2 + 15.0),
            text=str(x_tick),
            font=tick_font,
            fill=muted_rgb,
            stroke_fill=text_stroke,
            stroke_width=1,
        )

    draw.line(
        [px1, py2, px2, py2],
        fill=axis_rgb,
        width=resolve_int(params, "axis_line_width_px", 2),
    )
    draw.line(
        [px1, py1, px1, py2],
        fill=axis_rgb,
        width=resolve_int(params, "axis_line_width_px", 2),
    )

    draw_threshold_line = str(panel.panel_label) in {
        str(label) for label in dataset.query.threshold_panel_labels
    }
    if draw_threshold_line:
        _, threshold_y = _scale_point(
            x_value=float(dataset.x_values[0]),
            y_value=float(dataset.query.threshold_value),
            x_values=dataset.x_values,
            y_min=dataset.y_min,
            y_max=dataset.y_max,
            plot_bbox=plot_bbox,
        )
        _draw_dashed_line(
            draw,
            xy0=(px1, threshold_y),
            xy1=(px2, threshold_y),
            fill=threshold_rgb,
            width=2,
        )

    point_radius = float(resolve_int(params, "point_radius_px", 5))
    point_bboxes: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = []
    for curve in panel.curves:
        points: List[Tuple[float, float]] = []
        for x_value, y_value in zip(dataset.x_values, curve.values):
            points.append(
                _scale_point(
                    x_value=float(x_value),
                    y_value=float(y_value),
                    x_values=dataset.x_values,
                    y_min=dataset.y_min,
                    y_max=dataset.y_max,
                    plot_bbox=plot_bbox,
                )
            )
        if len(points) >= 2:
            draw.line(
                points,
                fill=curve.color_rgb,
                width=resolve_int(params, "line_width_px", 3),
                joint="curve",
            )
        for x_value, y_value, (cx, cy) in zip(dataset.x_values, curve.values, points):
            marker = [
                float(cx) - float(point_radius),
                float(cy) - float(point_radius),
                float(cx) + float(point_radius),
                float(cy) + float(point_radius),
            ]
            draw.ellipse(marker, fill=curve.color_rgb, outline=(255, 255, 255), width=1)
            marker_id = point_id(
                str(panel.panel_label), str(curve.method_label), int(x_value)
            )
            point_bboxes[str(marker_id)] = _bbox(marker)
            entities.append(
                {
                    "entity_id": str(marker_id),
                    "entity_type": "scientific_curve_marker",
                    "bbox_px": _bbox(marker),
                    "attrs": {
                        "panel_label": str(panel.panel_label),
                        "method_label": str(curve.method_label),
                        "x_value": int(x_value),
                        "y_value": int(y_value),
                        "center_px": [round(float(cx), 3), round(float(cy), 3)],
                    },
                }
            )

    return entities, point_bboxes, {}, _bbox(plot_bbox)


def _render_dataset(
    dataset: CurvePanelDataset, *, params: Mapping[str, Any], instance_seed: int
) -> RenderedCurvePanels:
    """Render the full small-multiple chart without adding objective semantics."""

    params = {**dict(params), "_render_style_seed": int(instance_seed)}
    protected_colors = tuple(
        tuple(int(channel) for channel in curve.color_rgb)
        for panel in dataset.panels
        for curve in panel.curves
    )
    information_style, information_style_meta = resolve_chart_information_style(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="curve_panels",
        protected_colors=protected_colors,
    )
    params = {
        **params,
        "axis_color_rgb": tuple(information_style.axis_rgb),
        "grid_color_rgb": tuple(information_style.grid_rgb),
        "panel_fill_rgb": tuple(information_style.panel_fill_rgb),
        "panel_border_rgb": tuple(information_style.panel_border_rgb),
        "text_color_rgb": tuple(information_style.text_rgb),
        "muted_text_rgb": tuple(information_style.muted_text_rgb),
        "text_stroke_rgb": tuple(information_style.text_stroke_rgb),
        "threshold_rgb": tuple(information_style.highlight_rgb),
    }
    canvas_width = resolve_int(params, "canvas_width", 1200)
    canvas_height = resolve_int(params, "canvas_height", 1000)
    background, background_meta = make_chart_information_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=information_style,
        instance_seed=int(instance_seed),
        namespace=f"charts.curve_panels.information_scene_background",
    )
    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    outer = resolve_int(params, "outer_margin_px", 42)
    margin_left, margin_right, margin_top, margin_bottom, layout_jitter_meta = (
        apply_layout_jitter_to_margins(
            left_px=int(outer),
            right_px=int(outer),
            top_px=int(outer),
            bottom_px=int(outer),
            params=params,
            defaults=RENDER_DEFAULTS,
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.layout",
        )
    )
    title_band = resolve_int(params, "title_band_height_px", 92)
    panel_gap = resolve_int(params, "panel_gap_px", 24)
    text_rgb = resolve_rgb(params, "text_color_rgb", (36, 42, 54))
    muted_rgb = resolve_rgb(params, "muted_text_rgb", (91, 102, 120))
    method_labels = tuple(curve.method_label for curve in dataset.panels[0].curves)
    legend_bboxes = _draw_legend(
        draw,
        method_labels=method_labels,
        colors=tuple(curve.color_rgb for curve in dataset.panels[0].curves),
        params=params,
        origin=(float(canvas_width) - 560.0, 38.0),
    )

    plot_bbox = [
        float(margin_left),
        float(title_band) + float(layout_jitter_meta.get("dy_px", 0)),
        float(canvas_width - margin_right),
        float(canvas_height - margin_bottom),
    ]
    panel_boxes = _panel_layout(
        tuple(plot_bbox), len(dataset.panels), gap=float(panel_gap)
    )

    entities: List[Dict[str, Any]] = []
    panel_bboxes: Dict[str, List[float]] = {}
    panel_plot_bboxes: Dict[str, List[float]] = {}
    point_bboxes: Dict[str, List[float]] = {}
    for panel, panel_bbox in zip(dataset.panels, panel_boxes):
        rendered_entities, rendered_points, _extra_bboxes, panel_plot_bbox = (
            _draw_panel(
                draw,
                panel=panel,
                panel_bbox=panel_bbox,
                dataset=dataset,
                params=params,
            )
        )
        panel_bboxes[str(panel.panel_label)] = _bbox(panel_bbox)
        panel_plot_bboxes[str(panel.panel_label)] = list(panel_plot_bbox)
        point_bboxes.update(rendered_points)
        entities.extend(rendered_entities)
        entities.append(
            {
                "entity_id": f"scientific_subplot_{str(panel.panel_label)}",
                "entity_type": "scientific_subplot_panel",
                "bbox_px": _bbox(panel_bbox),
                "attrs": {
                    "panel_label": str(panel.panel_label),
                    "method_count": int(len(panel.curves)),
                    "x_tick_count": int(len(dataset.x_values)),
                },
            }
        )

    intersection_bboxes: Dict[str, List[float]] = {}
    for intersection in dataset.intersections:
        panel_plot_bbox = panel_plot_bboxes.get(str(intersection.panel_label))
        if panel_plot_bbox is None:
            continue
        cx, cy = _scale_point(
            x_value=float(intersection.x_value),
            y_value=float(intersection.y_value),
            x_values=dataset.x_values,
            y_min=dataset.y_min,
            y_max=dataset.y_max,
            plot_bbox=tuple(float(value) for value in panel_plot_bbox),
        )
        radius = 7.0
        box = _bbox(
            [
                float(cx) - radius,
                float(cy) - radius,
                float(cx) + radius,
                float(cy) + radius,
            ]
        )
        intersection_bboxes[str(intersection.intersection_id)] = list(box)
        entities.append(
            {
                "entity_id": str(intersection.intersection_id),
                "entity_type": "scientific_curve_intersection",
                "bbox_px": list(box),
                "attrs": {
                    "panel_label": str(intersection.panel_label),
                    "method_a_label": str(intersection.method_a_label),
                    "method_b_label": str(intersection.method_b_label),
                    "x_value": round(float(intersection.x_value), 3),
                    "y_value": round(float(intersection.y_value), 3),
                    "center_px": [round(float(cx), 3), round(float(cy), 3)],
                },
            }
        )

    threshold_crossing_bboxes: Dict[str, List[float]] = {}
    for crossing in dataset.threshold_crossings:
        panel_plot_bbox = panel_plot_bboxes.get(str(crossing.panel_label))
        if panel_plot_bbox is None:
            continue
        cx, cy = _scale_point(
            x_value=float(crossing.x_value),
            y_value=float(crossing.y_value),
            x_values=dataset.x_values,
            y_min=dataset.y_min,
            y_max=dataset.y_max,
            plot_bbox=tuple(float(value) for value in panel_plot_bbox),
        )
        radius = 7.0
        box = _bbox(
            [
                float(cx) - radius,
                float(cy) - radius,
                float(cx) + radius,
                float(cy) + radius,
            ]
        )
        threshold_crossing_bboxes[str(crossing.crossing_id)] = list(box)
        entities.append(
            {
                "entity_id": str(crossing.crossing_id),
                "entity_type": "scientific_threshold_crossing",
                "bbox_px": list(box),
                "attrs": {
                    "panel_label": str(crossing.panel_label),
                    "method_label": str(crossing.method_label),
                    "direction": str(crossing.direction),
                    "x_value": round(float(crossing.x_value), 3),
                    "y_value": round(float(crossing.y_value), 3),
                    "center_px": [round(float(cx), 3), round(float(cy), 3)],
                },
            }
        )

    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedCurvePanels(
        image=image,
        entities=tuple(entities),
        plot_bbox_px=_bbox(plot_bbox),
        panel_bboxes=dict(panel_bboxes),
        panel_plot_bboxes=dict(panel_plot_bboxes),
        point_bboxes=dict(point_bboxes),
        intersection_bboxes=dict(intersection_bboxes),
        threshold_crossing_bboxes=dict(threshold_crossing_bboxes),
        legend_bboxes=dict(legend_bboxes),
        render_meta={
            "background_style": dict(background_meta),
            "information_scene_style": dict(information_style_meta),
            "post_image_noise": dict(post_noise_meta),
            "layout_jitter": dict(layout_jitter_meta),
            "x_values": list(dataset.x_values),
            "y_range": [int(dataset.y_min), int(dataset.y_max)],
            "panel_bboxes_px": dict(panel_bboxes),
            "panel_plot_bboxes_px": dict(panel_plot_bboxes),
            "threshold_crossing_bboxes_px": dict(threshold_crossing_bboxes),
        },
    )
