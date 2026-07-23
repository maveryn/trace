"""Rendering helpers for physics motion-graph tasks."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.diagram_style import (
    prepare_physics_diagram_style_and_background,
)
from trace_tasks.tasks.physics.shared.option_cards import draw_lettered_option_cards
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.bbox_projection import bbox_union_many
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.drawing import draw_centered_text
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.render_variation import resolve_layout_jitter, resolve_render_int
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .state import (
    AverageSpeedGraphSpec,
    IntervalGraphSpec,
    MotionGraphRenderDefaults,
    OPTION_LETTERS,
    RenderedMotionGraph,
    SCENE_ID,
    STATE_LABELS,
    StateGraphSpec,
)


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(
    scene_id=SCENE_ID,
    apply_prob=0.5,
)


def _bbox_from_points(
    points: Sequence[Tuple[float, float]],
    padding: float = 0.0,
) -> List[float]:
    xs = [float(point[0]) for point in points]
    ys = [float(point[1]) for point in points]
    return [
        round(min(xs) - float(padding), 3),
        round(min(ys) - float(padding), 3),
        round(max(xs) + float(padding), 3),
        round(max(ys) + float(padding), 3),
    ]


def _text_bbox(
    draw: ImageDraw.ImageDraw,
    text: str,
    center: Tuple[float, float],
    font: Any,
    padding: float = 5.0,
) -> List[float]:
    bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=0)
    width = float(bbox[2] - bbox[0])
    height = float(bbox[3] - bbox[1])
    return [
        round(float(center[0]) - width / 2.0 - float(padding), 3),
        round(float(center[1]) - height / 2.0 - float(padding), 3),
        round(float(center[0]) + width / 2.0 + float(padding), 3),
        round(float(center[1]) + height / 2.0 + float(padding), 3),
    ]


def _resolve_render_defaults(
    *,
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    fallback: MotionGraphRenderDefaults,
    instance_seed: int,
    namespace: str,
    include_options: bool,
) -> Dict[str, int]:
    """Resolve pixel render parameters while preserving graph geometry invariants."""

    keys = [
        "plot_left_px",
        "plot_top_px",
        "plot_width_px",
        "plot_height_px",
        "axis_width_px",
        "curve_width_px",
        "grid_line_width_px",
        "bold_grid_line_width_px",
        "label_font_size_px",
        "tick_font_size_px",
        "option_font_size_px",
        "title_font_size_px",
        "label_stroke_width_px",
        "point_radius_px",
        "y_min",
        "y_max",
        "t_min",
        "t_max",
    ]
    if bool(include_options):
        keys.extend(
            [
                "option_panel_top_px",
                "option_cell_left_px",
                "option_cell_width_px",
                "option_cell_height_px",
                "option_cell_gap_x_px",
                "option_cell_gap_y_px",
            ]
        )
    resolved = {
        key: resolve_render_int(
            params,
            defaults,
            key,
            int(getattr(fallback, key)),
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
        for key in keys
    }
    if not bool(include_options):
        for key in (
            "option_panel_top_px",
            "option_cell_left_px",
            "option_cell_width_px",
            "option_cell_height_px",
        ):
            resolved[key] = int(group_default(defaults, key, int(getattr(fallback, key))))
    return resolved


def _resolve_state_layout_placement(
    *,
    render_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    canvas_width: int,
    canvas_height: int,
    namespace: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    content_left = 48.0
    content_top = 40.0
    content_right = max(
        float(render_defaults["plot_left_px"]) + float(render_defaults["plot_width_px"]) + 80.0,
        float(render_defaults["option_cell_left_px"])
        + (float(render_defaults["option_cell_width_px"]) * 2.0)
        + float(render_defaults["option_cell_gap_x_px"])
        + 22.0,
    )
    content_bottom = (
        float(render_defaults["option_panel_top_px"])
        + (float(render_defaults["option_cell_height_px"]) * 2.0)
        + float(render_defaults["option_cell_gap_y_px"])
        + 20.0
    )
    return _resolve_layout_placement(
        render_defaults=render_defaults,
        params=params,
        instance_seed=int(instance_seed),
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        namespace=str(namespace),
        content_bbox=(content_left, content_top, content_right, content_bottom),
        mode="whole_motion_graph_offset",
        adjust_x_keys=("plot_left_px", "option_cell_left_px"),
        adjust_y_keys=("plot_top_px", "option_panel_top_px"),
    )


def _resolve_interval_layout_placement(
    *,
    render_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    canvas_width: int,
    canvas_height: int,
    namespace: str,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    content_left = 48.0
    content_top = 40.0
    content_right = float(render_defaults["plot_left_px"]) + float(render_defaults["plot_width_px"]) + 86.0
    content_bottom = float(render_defaults["plot_top_px"]) + float(render_defaults["plot_height_px"]) + 72.0
    return _resolve_layout_placement(
        render_defaults=render_defaults,
        params=params,
        instance_seed=int(instance_seed),
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        namespace=str(namespace),
        content_bbox=(content_left, content_top, content_right, content_bottom),
        mode="whole_motion_graph_interval_offset",
        adjust_x_keys=("plot_left_px",),
        adjust_y_keys=("plot_top_px",),
    )


def _resolve_layout_placement(
    *,
    render_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    canvas_width: int,
    canvas_height: int,
    namespace: str,
    content_bbox: Tuple[float, float, float, float],
    mode: str,
    adjust_x_keys: Sequence[str],
    adjust_y_keys: Sequence[str],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply whole-diagram jitter and project the same offset into witnesses."""

    content_left, content_top, content_right, content_bottom = [
        float(value) for value in content_bbox
    ]
    base_bbox = [
        round(content_left, 3),
        round(content_top, 3),
        round(content_right, 3),
        round(content_bottom, 3),
    ]
    jitter = resolve_layout_jitter(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.layout",
    )
    min_margin = int(jitter.get("min_margin_px", 18))
    requested_dx = int(jitter.get("requested_dx_px", 0))
    requested_dy = int(jitter.get("requested_dy_px", 0))
    min_dx = int(math.ceil(float(min_margin) - float(content_left)))
    max_dx = int(math.floor(float(canvas_width) - float(min_margin) - float(content_right)))
    min_dy = int(math.ceil(float(min_margin) - float(content_top)))
    max_dy = int(math.floor(float(canvas_height) - float(min_margin) - float(content_bottom)))
    if int(min_dx) > int(max_dx):
        min_dx = 0
        max_dx = 0
    if int(min_dy) > int(max_dy):
        min_dy = 0
        max_dy = 0
    if not bool(jitter.get("enabled", False)):
        requested_dx = 0
        requested_dy = 0
    dx = max(int(min_dx), min(int(max_dx), int(requested_dx)))
    dy = max(int(min_dy), min(int(max_dy), int(requested_dy)))

    adjusted = dict(render_defaults)
    for key in adjust_x_keys:
        adjusted[str(key)] = int(adjusted[str(key)]) + int(dx)
    for key in adjust_y_keys:
        adjusted[str(key)] = int(adjusted[str(key)]) + int(dy)
    final_bbox = [
        round(float(content_left) + float(dx), 3),
        round(float(content_top) + float(dy), 3),
        round(float(content_right) + float(dx), 3),
        round(float(content_bottom) + float(dy), 3),
    ]
    placement = dict(jitter)
    placement.update(
        {
            "mode": str(mode),
            "content_bbox_px": base_bbox,
            "final_content_bbox_px": final_bbox,
            "canvas_size_px": [int(canvas_width), int(canvas_height)],
            "available_offset_x_px": [int(min_dx), int(max_dx)],
            "available_offset_y_px": [int(min_dy), int(max_dy)],
            "sampled_offset_px": [int(requested_dx), int(requested_dy)],
            "final_offset_px": [int(dx), int(dy)],
            "dx_px": int(dx),
            "dy_px": int(dy),
        }
    )
    return adjusted, placement


def _plot_helpers(render_defaults: Mapping[str, Any]):
    plot_left = float(render_defaults["plot_left_px"])
    plot_top = float(render_defaults["plot_top_px"])
    plot_width = float(render_defaults["plot_width_px"])
    plot_height = float(render_defaults["plot_height_px"])
    plot_right = plot_left + plot_width
    plot_bottom = plot_top + plot_height
    y_min = int(render_defaults["y_min"])
    y_max = int(render_defaults["y_max"])
    t_min = int(render_defaults["t_min"])
    t_max = int(render_defaults["t_max"])

    def x_px(t_value: float) -> float:
        return float(plot_left + ((float(t_value) - t_min) / max(1.0, t_max - t_min)) * plot_width)

    def y_px(y_value: float) -> float:
        return float(plot_bottom - ((float(y_value) - y_min) / max(1.0, y_max - y_min)) * plot_height)

    return (
        plot_left,
        plot_top,
        plot_right,
        plot_bottom,
        y_min,
        y_max,
        t_min,
        t_max,
        x_px,
        y_px,
    )


def _draw_axes_and_grid(
    *,
    draw: ImageDraw.ImageDraw,
    spec: StateGraphSpec | IntervalGraphSpec | AverageSpeedGraphSpec,
    render_defaults: Mapping[str, Any],
    style: Any,
    tick_font: Any,
    label_font: Any,
    title_font: Any,
    title_y: float,
    collect_tick_bboxes: bool,
) -> tuple[List[List[float]], Dict[str, Any]]:
    """Draw the coordinate frame and return projection helpers for annotations."""

    (
        plot_left,
        plot_top,
        plot_right,
        plot_bottom,
        y_min,
        y_max,
        t_min,
        t_max,
        x_px,
        y_px,
    ) = _plot_helpers(render_defaults)
    label_rgb = tuple(int(v) for v in style.label_rgb)
    axis_rgb = tuple(int(v) for v in style.axis_rgb)
    tick_label_bboxes: List[List[float]] = []
    grid_width = int(
        render_defaults["bold_grid_line_width_px"]
        if str(spec.scene_style) == "bold_grid"
        else render_defaults["grid_line_width_px"]
    )
    for t_value in range(t_min, t_max + 1):
        x = x_px(float(t_value))
        draw.line(
            [(x, plot_top), (x, plot_bottom)],
            fill=tuple(int(v) for v in style.grid_minor_rgb),
            width=grid_width,
        )
        center = (x, plot_bottom + 23)
        draw_centered_text(
            draw,
            text=str(t_value),
            center=center,
            font=tick_font,
            fill=label_rgb,
            stroke_fill=resolve_text_stroke_fill(label_rgb),
            stroke_width=1,
        )
        if bool(collect_tick_bboxes):
            tick_label_bboxes.append(_text_bbox(draw, str(t_value), center, tick_font, padding=4.0))
    for y_value in range(y_min, y_max + 1):
        y = y_px(float(y_value))
        line_width = int(render_defaults["bold_grid_line_width_px"] if y_value == 0 else grid_width)
        line_fill = tuple(int(v) for v in (style.axis_rgb if y_value == 0 else style.grid_minor_rgb))
        draw.line([(plot_left, y), (plot_right, y)], fill=line_fill, width=line_width)
        if bool(collect_tick_bboxes) or y_value % 2 == 0:
            center = (plot_left - 28, y)
            draw_centered_text(
                draw,
                text=str(y_value),
                center=center,
                font=tick_font,
                fill=label_rgb,
                stroke_fill=resolve_text_stroke_fill(label_rgb),
                stroke_width=1,
            )
            if bool(collect_tick_bboxes):
                tick_label_bboxes.append(_text_bbox(draw, str(y_value), center, tick_font, padding=4.0))

    zero_y = y_px(0.0)
    x_axis_y = zero_y if bool(collect_tick_bboxes) else plot_bottom
    draw.line(
        [(plot_left, x_axis_y), (plot_right + 14, x_axis_y)],
        fill=axis_rgb,
        width=int(render_defaults["axis_width_px"]),
    )
    draw.line(
        [(plot_left, plot_top - 14), (plot_left, plot_bottom)],
        fill=axis_rgb,
        width=int(render_defaults["axis_width_px"]),
    )
    draw.polygon(
        [(plot_right + 14, x_axis_y), (plot_right - 5, x_axis_y - 9), (plot_right - 5, x_axis_y + 9)],
        fill=axis_rgb,
    )
    draw.polygon(
        [(plot_left, plot_top - 14), (plot_left - 9, plot_top + 5), (plot_left + 9, plot_top + 5)],
        fill=axis_rgb,
    )
    x_label_center = (plot_right + 52, plot_bottom + 18)
    y_label_center = (plot_left + 8, plot_top - 42)
    draw_centered_text(
        draw,
        text="t (s)",
        center=x_label_center,
        font=label_font,
        fill=label_rgb,
        stroke_fill=resolve_text_stroke_fill(label_rgb),
        stroke_width=1,
    )
    draw_centered_text(
        draw,
        text=str(spec.y_axis_label),
        center=y_label_center,
        font=label_font,
        fill=label_rgb,
        stroke_fill=resolve_text_stroke_fill(label_rgb),
        stroke_width=1,
    )
    if bool(collect_tick_bboxes):
        tick_label_bboxes.append(_text_bbox(draw, "t (s)", x_label_center, label_font, padding=5.0))
        tick_label_bboxes.append(_text_bbox(draw, str(spec.y_axis_label), y_label_center, label_font, padding=5.0))
    draw_centered_text(
        draw,
        text=str(spec.title),
        center=((plot_left + plot_right) * 0.5, float(title_y)),
        font=title_font,
        fill=label_rgb,
        stroke_fill=resolve_text_stroke_fill(label_rgb),
        stroke_width=1,
    )
    return tick_label_bboxes, {
        "plot_left": float(plot_left),
        "plot_top": float(plot_top),
        "plot_right": float(plot_right),
        "plot_bottom": float(plot_bottom),
        "zero_y": float(zero_y),
        "x_px": x_px,
        "y_px": y_px,
    }


def _render_state_graph(
    *,
    image: Image.Image,
    spec: StateGraphSpec,
    render_defaults: Mapping[str, Any],
    font_family: str,
    style: Any,
) -> RenderedMotionGraph:
    """Render state-choice graph/options and keyed option/interval witnesses."""

    draw = ImageDraw.Draw(image)
    width, _height = image.size
    title_font = load_font(int(render_defaults["title_font_size_px"]), bold=True, font_family=font_family)
    label_font = load_font(int(render_defaults["label_font_size_px"]), bold=True, font_family=font_family)
    tick_font = load_font(int(render_defaults["tick_font_size_px"]), bold=False, font_family=font_family)
    option_font = load_font(int(render_defaults["option_font_size_px"]), bold=True, font_family=font_family)
    plot_left = float(render_defaults["plot_left_px"])
    plot_top = float(render_defaults["plot_top_px"])
    plot_right = plot_left + float(render_defaults["plot_width_px"])
    plot_bottom = plot_top + float(render_defaults["plot_height_px"])
    panel_bbox = [48, 40, width - 48, int(render_defaults["option_panel_top_px"]) - 24]
    draw.rounded_rectangle(
        panel_bbox,
        radius=18,
        fill=tuple(int(v) for v in style.panel_fill_rgb),
        outline=tuple(int(v) for v in style.panel_border_rgb),
        width=3,
    )
    plot_fill = tuple(
        int(v)
        for v in (
            style.panel_alt_fill_rgb
            if str(spec.scene_style) == "paper_grid"
            else style.panel_fill_rgb
        )
    )
    draw.rounded_rectangle(
        [plot_left - 18, plot_top - 22, plot_right + 28, plot_bottom + 42],
        radius=14,
        fill=plot_fill,
        outline=tuple(int(v) for v in style.panel_border_rgb),
        width=2,
    )
    _tick_boxes, plot = _draw_axes_and_grid(
        draw=draw,
        spec=spec,
        render_defaults=render_defaults,
        style=style,
        tick_font=tick_font,
        label_font=label_font,
        title_font=title_font,
        title_y=float(panel_bbox[1] + 28),
        collect_tick_bboxes=False,
    )
    x_px = plot["x_px"]
    y_px = plot["y_px"]
    points = [(x_px(t), y_px(y)) for t, y in zip(spec.t_values, spec.y_values)]
    seg_start = points[spec.target_segment_index]
    seg_end = points[spec.target_segment_index + 1]
    band_left = x_px(spec.t_values[spec.target_segment_index])
    band_right = x_px(spec.t_values[spec.target_segment_index + 1])
    accent = tuple(int(v) for v in style.accent_rgb)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [band_left, plot_top, band_right, plot_bottom],
        fill=(accent[0], accent[1], accent[2], 46),
        outline=(accent[0], accent[1], accent[2], 180),
        width=3,
    )
    image.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert(image.mode))
    draw = ImageDraw.Draw(image)

    curve_rgb = tuple(int(v) for v in style.stroke_rgb)
    draw.line(points, fill=curve_rgb, width=int(render_defaults["curve_width_px"]), joint="curve")
    highlight_rgb = tuple(int(v) for v in style.accent_rgb)
    draw.line(
        [seg_start, seg_end],
        fill=highlight_rgb,
        width=int(render_defaults["curve_width_px"]) + 3,
    )
    for point in points:
        radius = float(render_defaults["point_radius_px"])
        draw.ellipse(
            [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
            fill=curve_rgb,
            outline=tuple(int(v) for v in style.panel_fill_rgb),
            width=2,
        )
    bracket_y = plot_top + 18.0
    draw.line([(band_left, bracket_y), (band_right, bracket_y)], fill=highlight_rgb, width=4)
    draw.line([(band_left, bracket_y - 10), (band_left, bracket_y + 10)], fill=highlight_rgb, width=4)
    draw.line([(band_right, bracket_y - 10), (band_right, bracket_y + 10)], fill=highlight_rgb, width=4)
    draw_centered_text(
        draw,
        text="marked interval",
        center=((band_left + band_right) / 2.0, bracket_y + 24.0),
        font=tick_font,
        fill=highlight_rgb,
        stroke_fill=resolve_text_stroke_fill(highlight_rgb),
        stroke_width=1,
    )
    query_region_bbox = [
        round(float(band_left), 3),
        round(float(plot_top), 3),
        round(float(band_right), 3),
        round(float(plot_bottom), 3),
    ]
    curve_segment_bbox = _bbox_from_points(
        [seg_start, seg_end],
        padding=float(render_defaults["curve_width_px"]) + 6.0,
    )
    curve_segment = [
        [round(float(seg_start[0]), 3), round(float(seg_start[1]), 3)],
        [round(float(seg_end[0]), 3), round(float(seg_end[1]), 3)],
    ]
    option_cards = draw_lettered_option_cards(
        draw,
        options=[
            (str(letter), STATE_LABELS[str(spec.option_map[str(letter)])])
            for letter in OPTION_LETTERS
        ],
        option_left=float(render_defaults["option_cell_left_px"]),
        option_top=float(render_defaults["option_panel_top_px"]),
        card_width=float(render_defaults["option_cell_width_px"]),
        card_height=float(render_defaults["option_cell_height_px"]),
        card_gap_x=float(render_defaults["option_cell_gap_x_px"]),
        card_gap_y=float(render_defaults["option_cell_gap_y_px"]),
        columns=2,
        option_font=option_font,
        letter_font=option_font,
        text_rgb=tuple(int(v) for v in style.label_rgb),
        card_fill_rgb=tuple(int(v) for v in style.panel_alt_fill_rgb),
        card_outline_rgb=tuple(int(v) for v in style.panel_border_rgb),
        label_fill_rgb=tuple(int(v) for v in style.label_fill_rgb),
        label_outline_rgb=tuple(int(v) for v in style.label_border_rgb),
        label_text_rgb=tuple(int(v) for v in style.label_rgb),
        label_stroke_width_px=int(render_defaults["label_stroke_width_px"]),
        text_stroke_width_px=1,
        panel_fill_rgb=tuple(int(v) for v in style.panel_fill_rgb),
        panel_outline_rgb=tuple(int(v) for v in style.panel_border_rgb),
        panel_padding_px=20.0,
        panel_radius_px=16.0,
        panel_outline_width_px=3,
    )
    option_bboxes = option_cards.option_bboxes
    option_letter_bboxes = option_cards.option_letter_bboxes
    option_text_bboxes = option_cards.option_text_bboxes
    scene_entities: List[Dict[str, Any]] = [
        {
            "entity_id": "query_region",
            "entity_type": "marked_time_interval",
            "bbox_px": list(query_region_bbox),
            "meta": {
                "t_start": int(spec.t_values[spec.target_segment_index]),
                "t_end": int(spec.t_values[spec.target_segment_index + 1]),
            },
        },
        {
            "entity_id": "curve_segment",
            "entity_type": "queried_graph_segment",
            "bbox_px": list(curve_segment_bbox),
            "meta": {
                "y_start": int(spec.y_values[spec.target_segment_index]),
                "y_end": int(spec.y_values[spec.target_segment_index + 1]),
                "motion_state": str(spec.motion_state),
            },
        },
    ]
    for letter in OPTION_LETTERS:
        scene_entities.append(
            {
                "entity_id": f"option_{str(letter)}",
                "entity_type": "motion_state_option",
                "bbox_px": list(option_bboxes[str(letter)]),
                "meta": {
                    "option_letter": str(letter),
                    "motion_state": str(spec.option_map[str(letter)]),
                    "option_text": STATE_LABELS[str(spec.option_map[str(letter)])],
                    "is_correct": str(letter) == str(spec.correct_option_letter),
                },
            }
        )
    annotation_bbox_map = {
        "query_region": list(query_region_bbox),
        "curve_segment": list(curve_segment_bbox),
    }
    render_map = {
        "plot_bbox_px": [
            round(float(plot_left), 3),
            round(float(plot_top), 3),
            round(float(plot_right), 3),
            round(float(plot_bottom), 3),
        ],
        "graph_kind": str(spec.graph_kind),
        "operation": str(spec.operation),
        "motion_state": str(spec.motion_state),
        "correct_option_letter": str(spec.correct_option_letter),
        "option_map": dict(spec.option_map),
        "option_text_map": {
            str(letter): STATE_LABELS[str(state)]
            for letter, state in spec.option_map.items()
        },
        "option_bboxes_px": {str(letter): list(bbox) for letter, bbox in option_bboxes.items()},
        "option_letter_bboxes_px": {
            str(letter): list(bbox)
            for letter, bbox in option_letter_bboxes.items()
        },
        "option_text_bboxes_px": {
            str(letter): list(bbox)
            for letter, bbox in option_text_bboxes.items()
        },
        "t_values": [int(value) for value in spec.t_values],
        "y_values": [int(value) for value in spec.y_values],
        "points_px": [[round(float(x), 3), round(float(y), 3)] for x, y in points],
        "target_segment_index": int(spec.target_segment_index),
        "query_region_bbox_px": list(query_region_bbox),
        "curve_segment_bbox_px": list(curve_segment_bbox),
        "curve_segment_px": list(curve_segment),
        "annotation_keyed_bboxes_px": dict(annotation_bbox_map),
    }
    return RenderedMotionGraph(
        image=image,
        annotation_bbox_map={str(key): list(value) for key, value in annotation_bbox_map.items()},
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
        font_family=str(font_family),
    )


def _render_interval_graph(
    *,
    image: Image.Image,
    spec: IntervalGraphSpec,
    render_defaults: Mapping[str, Any],
    font_family: str,
    style: Any,
) -> RenderedMotionGraph:
    """Render interval-displacement graph and keyed graph-region witnesses."""

    draw = ImageDraw.Draw(image)
    width, height = image.size
    plot_left = float(render_defaults["plot_left_px"])
    plot_top = float(render_defaults["plot_top_px"])
    plot_right = plot_left + float(render_defaults["plot_width_px"])
    plot_bottom = plot_top + float(render_defaults["plot_height_px"])
    title_font = load_font(int(render_defaults["title_font_size_px"]), bold=True, font_family=font_family)
    label_font = load_font(int(render_defaults["label_font_size_px"]), bold=True, font_family=font_family)
    tick_font = load_font(int(render_defaults["tick_font_size_px"]), bold=False, font_family=font_family)
    small_font = load_font(
        max(14, int(render_defaults["tick_font_size_px"]) - 1),
        bold=True,
        font_family=font_family,
    )
    panel_bbox = [48, 40, width - 48, min(height - 42, int(plot_bottom + 64))]
    draw.rounded_rectangle(
        panel_bbox,
        radius=18,
        fill=tuple(int(v) for v in style.panel_fill_rgb),
        outline=tuple(int(v) for v in style.panel_border_rgb),
        width=3,
    )
    plot_fill = tuple(
        int(v)
        for v in (
            style.panel_alt_fill_rgb
            if str(spec.scene_style) == "paper_grid"
            else style.panel_fill_rgb
        )
    )
    draw.rounded_rectangle(
        [plot_left - 18, plot_top - 22, plot_right + 30, plot_bottom + 42],
        radius=14,
        fill=plot_fill,
        outline=tuple(int(v) for v in style.panel_border_rgb),
        width=2,
    )
    tick_boxes, plot = _draw_axes_and_grid(
        draw=draw,
        spec=spec,
        render_defaults=render_defaults,
        style=style,
        tick_font=tick_font,
        label_font=label_font,
        title_font=title_font,
        title_y=float(panel_bbox[1] + 28),
        collect_tick_bboxes=True,
    )
    x_px = plot["x_px"]
    y_px = plot["y_px"]
    zero_y = float(plot["zero_y"])
    points = [(x_px(t), y_px(v)) for t, v in zip(spec.t_values, spec.velocity_values)]
    start_index = int(spec.t_start) - int(render_defaults["t_min"])
    end_index = int(spec.t_end) - int(render_defaults["t_min"])
    interval_points = points[start_index : end_index + 1]
    band_left = x_px(spec.t_start)
    band_right = x_px(spec.t_end)
    accent = tuple(int(v) for v in style.accent_rgb)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [band_left, plot_top, band_right, plot_bottom],
        fill=(accent[0], accent[1], accent[2], 35),
        outline=(accent[0], accent[1], accent[2], 155),
        width=3,
    )
    area_polygon = [(band_left, zero_y), *interval_points, (band_right, zero_y)]
    overlay_draw.polygon(area_polygon, fill=(accent[0], accent[1], accent[2], 70))
    image.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert(image.mode))
    draw = ImageDraw.Draw(image)

    curve_rgb = tuple(int(v) for v in style.stroke_rgb)
    draw.line(points, fill=curve_rgb, width=int(render_defaults["curve_width_px"]), joint="curve")
    highlight_rgb = tuple(int(v) for v in style.accent_rgb)
    draw.line(
        interval_points,
        fill=highlight_rgb,
        width=int(render_defaults["curve_width_px"]) + 3,
        joint="curve",
    )
    for point in points:
        radius = float(render_defaults["point_radius_px"])
        draw.ellipse(
            [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
            fill=curve_rgb,
            outline=tuple(int(v) for v in style.panel_fill_rgb),
            width=2,
        )
    bracket_y = plot_top + 18.0
    draw.line([(band_left, bracket_y), (band_right, bracket_y)], fill=highlight_rgb, width=4)
    draw.line([(band_left, bracket_y - 10), (band_left, bracket_y + 10)], fill=highlight_rgb, width=4)
    draw.line([(band_right, bracket_y - 10), (band_right, bracket_y + 10)], fill=highlight_rgb, width=4)
    draw_centered_text(
        draw,
        text="marked interval",
        center=((band_left + band_right) / 2.0, bracket_y + 24.0),
        font=small_font,
        fill=highlight_rgb,
        stroke_fill=resolve_text_stroke_fill(highlight_rgb),
        stroke_width=1,
    )
    marked_interval_bbox = [
        round(float(band_left), 3),
        round(float(plot_top), 3),
        round(float(band_right), 3),
        round(float(plot_bottom), 3),
    ]
    velocity_segment_bbox = _bbox_from_points(
        interval_points,
        padding=float(render_defaults["curve_width_px"]) + 8.0,
    )
    velocity_segment = [
        [round(float(interval_points[0][0]), 3), round(float(interval_points[0][1]), 3)],
        [round(float(interval_points[-1][0]), 3), round(float(interval_points[-1][1]), 3)],
    ]
    axis_scale_bbox = bbox_union_many(
        [plot_left - 58.0, plot_top - 50.0, plot_left + 10.0, plot_bottom + 34.0],
        [plot_left - 8.0, plot_bottom - 8.0, plot_right + 64.0, plot_bottom + 40.0],
        *tick_boxes,
        padding=2.0,
    )
    annotation_bbox_map = {
        "marked_interval": list(marked_interval_bbox),
        "velocity_segment": list(velocity_segment_bbox),
        "axis_scale": list(axis_scale_bbox),
    }
    scene_entities = [
        {
            "entity_id": "marked_interval",
            "entity_type": "marked_time_interval",
            "bbox_px": list(marked_interval_bbox),
            "meta": {"t_start": int(spec.t_start), "t_end": int(spec.t_end)},
        },
        {
            "entity_id": "velocity_segment",
            "entity_type": "queried_velocity_segment",
            "bbox_px": list(velocity_segment_bbox),
            "meta": {"v_start": int(spec.v_start), "v_end": int(spec.v_end)},
        },
        {
            "entity_id": "axis_scale",
            "entity_type": "axis_scale",
            "bbox_px": list(axis_scale_bbox),
            "meta": {"t_unit": "s", "velocity_unit": "m/s"},
        },
    ]
    render_map = {
        "plot_bbox_px": [
            round(float(plot_left), 3),
            round(float(plot_top), 3),
            round(float(plot_right), 3),
            round(float(plot_bottom), 3),
        ],
        "graph_kind": str(spec.graph_kind),
        "segment_mode": str(spec.segment_mode),
        "t_values": [int(value) for value in spec.t_values],
        "velocity_values": [int(value) for value in spec.velocity_values],
        "points_px": [[round(float(x), 3), round(float(y), 3)] for x, y in points],
        "t_start": int(spec.t_start),
        "t_end": int(spec.t_end),
        "delta_t_s": int(spec.t_end) - int(spec.t_start),
        "v_start_m_s": int(spec.v_start),
        "v_end_m_s": int(spec.v_end),
        "displacement_m": int(spec.displacement_m),
        "marked_interval_bbox_px": list(marked_interval_bbox),
        "velocity_segment_bbox_px": list(velocity_segment_bbox),
        "velocity_segment_px": list(velocity_segment),
        "axis_scale_bbox_px": list(axis_scale_bbox),
        "annotation_keyed_bboxes_px": dict(annotation_bbox_map),
    }
    return RenderedMotionGraph(
        image=image,
        annotation_bbox_map={str(key): list(value) for key, value in annotation_bbox_map.items()},
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
        font_family=str(font_family),
    )


def _render_average_speed_graph(
    *,
    image: Image.Image,
    spec: AverageSpeedGraphSpec,
    render_defaults: Mapping[str, Any],
    font_family: str,
    style: Any,
) -> RenderedMotionGraph:
    """Render a distance-time graph and keyed rate-of-change witnesses."""

    draw = ImageDraw.Draw(image)
    width, height = image.size
    plot_left = float(render_defaults["plot_left_px"])
    plot_top = float(render_defaults["plot_top_px"])
    plot_right = plot_left + float(render_defaults["plot_width_px"])
    plot_bottom = plot_top + float(render_defaults["plot_height_px"])
    title_font = load_font(int(render_defaults["title_font_size_px"]), bold=True, font_family=font_family)
    label_font = load_font(int(render_defaults["label_font_size_px"]), bold=True, font_family=font_family)
    tick_font = load_font(int(render_defaults["tick_font_size_px"]), bold=False, font_family=font_family)
    small_font = load_font(
        max(14, int(render_defaults["tick_font_size_px"]) - 1),
        bold=True,
        font_family=font_family,
    )
    panel_bbox = [48, 40, width - 48, min(height - 42, int(plot_bottom + 64))]
    draw.rounded_rectangle(
        panel_bbox,
        radius=18,
        fill=tuple(int(v) for v in style.panel_fill_rgb),
        outline=tuple(int(v) for v in style.panel_border_rgb),
        width=3,
    )
    plot_fill = tuple(
        int(v)
        for v in (
            style.panel_alt_fill_rgb
            if str(spec.scene_style) == "paper_grid"
            else style.panel_fill_rgb
        )
    )
    draw.rounded_rectangle(
        [plot_left - 18, plot_top - 22, plot_right + 30, plot_bottom + 42],
        radius=14,
        fill=plot_fill,
        outline=tuple(int(v) for v in style.panel_border_rgb),
        width=2,
    )
    tick_boxes, plot = _draw_axes_and_grid(
        draw=draw,
        spec=spec,
        render_defaults=render_defaults,
        style=style,
        tick_font=tick_font,
        label_font=label_font,
        title_font=title_font,
        title_y=float(panel_bbox[1] + 28),
        collect_tick_bboxes=True,
    )
    x_px = plot["x_px"]
    y_px = plot["y_px"]
    points = [(x_px(t), y_px(d)) for t, d in zip(spec.t_values, spec.distance_values)]
    start_index = int(spec.t_start) - int(render_defaults["t_min"])
    end_index = int(spec.t_end) - int(render_defaults["t_min"])
    interval_points = points[start_index : end_index + 1]
    band_left = x_px(spec.t_start)
    band_right = x_px(spec.t_end)
    accent = tuple(int(v) for v in style.accent_rgb)
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.rectangle(
        [band_left, plot_top, band_right, plot_bottom],
        fill=(accent[0], accent[1], accent[2], 35),
        outline=(accent[0], accent[1], accent[2], 155),
        width=3,
    )
    image.paste(Image.alpha_composite(image.convert("RGBA"), overlay).convert(image.mode))
    draw = ImageDraw.Draw(image)

    curve_rgb = tuple(int(v) for v in style.stroke_rgb)
    draw.line(points, fill=curve_rgb, width=int(render_defaults["curve_width_px"]), joint="curve")
    highlight_rgb = tuple(int(v) for v in style.accent_rgb)
    draw.line(
        interval_points,
        fill=highlight_rgb,
        width=int(render_defaults["curve_width_px"]) + 3,
        joint="curve",
    )
    for point in points:
        radius = float(render_defaults["point_radius_px"])
        draw.ellipse(
            [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
            fill=curve_rgb,
            outline=tuple(int(v) for v in style.panel_fill_rgb),
            width=2,
        )
    bracket_y = plot_top + 18.0
    draw.line([(band_left, bracket_y), (band_right, bracket_y)], fill=highlight_rgb, width=4)
    draw.line([(band_left, bracket_y - 10), (band_left, bracket_y + 10)], fill=highlight_rgb, width=4)
    draw.line([(band_right, bracket_y - 10), (band_right, bracket_y + 10)], fill=highlight_rgb, width=4)
    draw_centered_text(
        draw,
        text="marked interval",
        center=((band_left + band_right) / 2.0, bracket_y + 24.0),
        font=small_font,
        fill=highlight_rgb,
        stroke_fill=resolve_text_stroke_fill(highlight_rgb),
        stroke_width=1,
    )
    marked_interval_bbox = [
        round(float(band_left), 3),
        round(float(plot_top), 3),
        round(float(band_right), 3),
        round(float(plot_bottom), 3),
    ]
    distance_segment_bbox = _bbox_from_points(
        interval_points,
        padding=float(render_defaults["curve_width_px"]) + 8.0,
    )
    distance_segment = [
        [round(float(interval_points[0][0]), 3), round(float(interval_points[0][1]), 3)],
        [round(float(interval_points[-1][0]), 3), round(float(interval_points[-1][1]), 3)],
    ]
    axis_scale_bbox = bbox_union_many(
        [plot_left - 58.0, plot_top - 50.0, plot_left + 10.0, plot_bottom + 34.0],
        [plot_left - 8.0, plot_bottom - 8.0, plot_right + 64.0, plot_bottom + 40.0],
        *tick_boxes,
        padding=2.0,
    )
    annotation_bbox_map = {
        "marked_interval": list(marked_interval_bbox),
        "distance_segment": list(distance_segment_bbox),
        "axis_scale": list(axis_scale_bbox),
    }
    scene_entities = [
        {
            "entity_id": "marked_interval",
            "entity_type": "marked_time_interval",
            "bbox_px": list(marked_interval_bbox),
            "meta": {"t_start": int(spec.t_start), "t_end": int(spec.t_end)},
        },
        {
            "entity_id": "distance_segment",
            "entity_type": "queried_distance_segment",
            "bbox_px": list(distance_segment_bbox),
            "meta": {"d_start": int(spec.d_start), "d_end": int(spec.d_end)},
        },
        {
            "entity_id": "axis_scale",
            "entity_type": "axis_scale",
            "bbox_px": list(axis_scale_bbox),
            "meta": {"t_unit": "s", "distance_unit": "m"},
        },
    ]
    render_map = {
        "plot_bbox_px": [
            round(float(plot_left), 3),
            round(float(plot_top), 3),
            round(float(plot_right), 3),
            round(float(plot_bottom), 3),
        ],
        "graph_kind": str(spec.graph_kind),
        "t_values": [int(value) for value in spec.t_values],
        "distance_values": [int(value) for value in spec.distance_values],
        "points_px": [[round(float(x), 3), round(float(y), 3)] for x, y in points],
        "t_start": int(spec.t_start),
        "t_end": int(spec.t_end),
        "delta_t_s": int(spec.t_end) - int(spec.t_start),
        "d_start_m": int(spec.d_start),
        "d_end_m": int(spec.d_end),
        "delta_d_m": int(spec.d_end) - int(spec.d_start),
        "average_speed_m_s": int(spec.average_speed_m_s),
        "marked_interval_bbox_px": list(marked_interval_bbox),
        "distance_segment_bbox_px": list(distance_segment_bbox),
        "distance_segment_px": list(distance_segment),
        "axis_scale_bbox_px": list(axis_scale_bbox),
        "annotation_keyed_bboxes_px": dict(annotation_bbox_map),
    }
    return RenderedMotionGraph(
        image=image,
        annotation_bbox_map={str(key): list(value) for key, value in annotation_bbox_map.items()},
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
        font_family=str(font_family),
    )


def render_state_choice_graph(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    spec: StateGraphSpec,
    render_defaults: Mapping[str, Any],
    fallback: MotionGraphRenderDefaults,
    namespace: str,
) -> RenderedMotionGraph:
    """Prepare and draw a state-choice motion graph."""

    canvas_width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", fallback.canvas_width)))
    canvas_height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", fallback.canvas_height)))
    resolved_defaults = _resolve_render_defaults(
        params=params,
        defaults=render_defaults,
        fallback=fallback,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        include_options=True,
    )
    resolved_defaults, layout_meta = _resolve_state_layout_placement(
        render_defaults=resolved_defaults,
        params=params,
        instance_seed=int(instance_seed),
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        namespace=str(namespace),
    )
    background, background_meta, diagram_style, diagram_style_meta = (
        prepare_physics_diagram_style_and_background(
            instance_seed=int(instance_seed),
            params=params,
            scene_id=SCENE_ID,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            require_grid=True,
            style_profile="graph_paper",
        )
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font",
        params=params,
    )
    rendered = _render_state_graph(
        image=background,
        spec=spec,
        render_defaults=resolved_defaults,
        font_family=str(font_family),
        style=diagram_style,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    render_map = dict(rendered.render_map)
    render_map["technical_diagram_style"] = dict(diagram_style_meta)
    render_map["background_style"] = dict(background_meta)
    render_map["layout_placement"] = dict(layout_meta)
    render_map["post_image_noise"] = dict(post_noise_meta)
    return RenderedMotionGraph(
        image=image,
        annotation_bbox_map=dict(rendered.annotation_bbox_map),
        scene_entities=list(rendered.scene_entities),
        render_map=render_map,
        font_family=str(font_family),
    )


def render_interval_graph(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    spec: IntervalGraphSpec,
    render_defaults: Mapping[str, Any],
    fallback: MotionGraphRenderDefaults,
    namespace: str,
) -> RenderedMotionGraph:
    """Prepare and draw an interval-displacement motion graph."""

    canvas_width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", fallback.canvas_width)))
    canvas_height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", fallback.canvas_height)))
    resolved_defaults = _resolve_render_defaults(
        params=params,
        defaults=render_defaults,
        fallback=fallback,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        include_options=False,
    )
    resolved_defaults, layout_meta = _resolve_interval_layout_placement(
        render_defaults=resolved_defaults,
        params=params,
        instance_seed=int(instance_seed),
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        namespace=str(namespace),
    )
    background, background_meta, diagram_style, diagram_style_meta = (
        prepare_physics_diagram_style_and_background(
            instance_seed=int(instance_seed),
            params=params,
            scene_id=SCENE_ID,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            require_grid=True,
            style_profile="graph_paper",
        )
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font",
        params=params,
    )
    rendered = _render_interval_graph(
        image=background,
        spec=spec,
        render_defaults=resolved_defaults,
        font_family=str(font_family),
        style=diagram_style,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    render_map = dict(rendered.render_map)
    render_map["technical_diagram_style"] = dict(diagram_style_meta)
    render_map["background_style"] = dict(background_meta)
    render_map["layout_placement"] = dict(layout_meta)
    render_map["post_image_noise"] = dict(post_noise_meta)
    return RenderedMotionGraph(
        image=image,
        annotation_bbox_map=dict(rendered.annotation_bbox_map),
        scene_entities=list(rendered.scene_entities),
        render_map=render_map,
        font_family=str(font_family),
    )


def render_average_speed_graph(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    spec: AverageSpeedGraphSpec,
    render_defaults: Mapping[str, Any],
    fallback: MotionGraphRenderDefaults,
    namespace: str,
) -> RenderedMotionGraph:
    """Prepare and draw an average-speed distance-time graph."""

    canvas_width = int(params.get("canvas_width", group_default(render_defaults, "canvas_width", fallback.canvas_width)))
    canvas_height = int(params.get("canvas_height", group_default(render_defaults, "canvas_height", fallback.canvas_height)))
    resolved_defaults = _resolve_render_defaults(
        params=params,
        defaults=render_defaults,
        fallback=fallback,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        include_options=False,
    )
    resolved_defaults, layout_meta = _resolve_interval_layout_placement(
        render_defaults=resolved_defaults,
        params=params,
        instance_seed=int(instance_seed),
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        namespace=str(namespace),
    )
    background, background_meta, diagram_style, diagram_style_meta = (
        prepare_physics_diagram_style_and_background(
            instance_seed=int(instance_seed),
            params=params,
            scene_id=SCENE_ID,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            require_grid=True,
            style_profile="graph_paper",
        )
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font",
        params=params,
    )
    rendered = _render_average_speed_graph(
        image=background,
        spec=spec,
        render_defaults=resolved_defaults,
        font_family=str(font_family),
        style=diagram_style,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    render_map = dict(rendered.render_map)
    render_map["technical_diagram_style"] = dict(diagram_style_meta)
    render_map["background_style"] = dict(background_meta)
    render_map["layout_placement"] = dict(layout_meta)
    render_map["post_image_noise"] = dict(post_noise_meta)
    return RenderedMotionGraph(
        image=image,
        annotation_bbox_map=dict(rendered.annotation_bbox_map),
        scene_entities=list(rendered.scene_entities),
        render_map=render_map,
        font_family=str(font_family),
    )


__all__ = [
    "render_average_speed_graph",
    "render_interval_graph",
    "render_state_choice_graph",
]
