"""Rendering primitives for error-interval charts."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.charts.shared.cartesian.geometry import round_bbox, round_point
from trace_tasks.tasks.charts.error_interval.shared.defaults import SCENE_NAMESPACE, _RENDER_DEFAULTS
from trace_tasks.tasks.charts.error_interval.shared.state import BBox, Point, RGB, Segment, _Dataset, _IntervalItem, _Rendered, _RenderParams
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.drawing import draw_dashed_line, draw_rounded_rect
from trace_tasks.tasks.shared.font_assets import font_asset_version, sample_font_family
from trace_tasks.tasks.shared.render_variation import apply_layout_jitter_to_margins, resolve_render_int, resolve_render_rgb
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font


def _bbox(values: Sequence[float]) -> BBox:
    return round_bbox(values)

def _point(x: float, y: float) -> Point:
    return round_point(float(x), float(y))

def _render_int(params: Mapping[str, Any], key: str, fallback: int, *, instance_seed: int | None = None) -> int:
    return int(
        resolve_render_int(
            params,
            _RENDER_DEFAULTS,
            str(key),
            int(fallback),
            instance_seed=instance_seed,
            namespace=SCENE_NAMESPACE,
        )
    )


def _render_float(params: Mapping[str, Any], key: str, fallback: float) -> float:
    return float(params.get(str(key), group_default(_RENDER_DEFAULTS, str(key), float(fallback))))


def _resolve_render_params(params: Mapping[str, Any], *, instance_seed: int) -> _RenderParams:
    """Resolve all style, margin, font, and axis parameters for one render."""

    def rint(key: str, fallback: int) -> int:
        return _render_int(params, str(key), int(fallback), instance_seed=int(instance_seed))

    outer_margin = _render_int(params, "outer_margin_px", 54)
    margin_left, margin_right, margin_top, margin_bottom, layout_jitter_meta = apply_layout_jitter_to_margins(
        left_px=int(outer_margin),
        right_px=int(outer_margin),
        top_px=int(outer_margin),
        bottom_px=int(outer_margin),
        params=params,
        defaults=_RENDER_DEFAULTS,
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.layout",
    )
    return _RenderParams(
        canvas_width=rint("canvas_width", 1320),
        canvas_height=rint("canvas_height", 900),
        outer_margin_px=int(outer_margin),
        outer_margin_left_px=int(margin_left),
        outer_margin_right_px=int(margin_right),
        outer_margin_top_px=int(margin_top),
        outer_margin_bottom_px=int(margin_bottom),
        title_band_height_px=rint("title_band_height_px", 72),
        label_band_px=rint("label_band_px", 154),
        plot_padding_px=rint("plot_padding_px", 32),
        panel_corner_radius_px=rint("panel_corner_radius_px", 10),
        panel_outline_width_px=rint("panel_outline_width_px", 2),
        axis_line_width_px=rint("axis_line_width_px", 2),
        grid_line_width_px=rint("grid_line_width_px", 1),
        interval_line_width_px=rint("interval_line_width_px", 5),
        cap_length_px=rint("cap_length_px", 20),
        point_radius_px=rint("point_radius_px", 7),
        bar_width_fraction=_render_float(params, "bar_width_fraction", 0.58),
        title_font_size_px=rint("title_font_size_px", 30),
        label_font_size_px=rint("label_font_size_px", 20),
        tick_font_size_px=rint("tick_font_size_px", 17),
        value_font_size_px=rint("value_font_size_px", 15),
        axis_min=rint("axis_min", 0),
        axis_max=rint("axis_max", 100),
        tick_step=max(1, rint("tick_step", 20)),
        text_rgb=resolve_render_rgb(params, _RENDER_DEFAULTS, "text_rgb", [38, 41, 48], instance_seed=int(instance_seed), namespace="charts.error_interval"),
        muted_text_rgb=resolve_render_rgb(params, _RENDER_DEFAULTS, "muted_text_rgb", [88, 96, 112], instance_seed=int(instance_seed), namespace="charts.error_interval"),
        text_stroke_rgb=resolve_render_rgb(params, _RENDER_DEFAULTS, "text_stroke_rgb", [255, 255, 255], instance_seed=int(instance_seed), namespace="charts.error_interval"),
        panel_fill_rgb=resolve_render_rgb(params, _RENDER_DEFAULTS, "panel_fill_rgb", [255, 255, 255], instance_seed=int(instance_seed), namespace="charts.error_interval"),
        panel_outline_rgb=resolve_render_rgb(params, _RENDER_DEFAULTS, "panel_outline_rgb", [194, 202, 214], instance_seed=int(instance_seed), namespace="charts.error_interval"),
        axis_rgb=resolve_render_rgb(params, _RENDER_DEFAULTS, "axis_rgb", [64, 68, 76], instance_seed=int(instance_seed), namespace="charts.error_interval"),
        grid_rgb=resolve_render_rgb(params, _RENDER_DEFAULTS, "grid_rgb", [224, 227, 232], instance_seed=int(instance_seed), namespace="charts.error_interval"),
        reference_rgb=resolve_render_rgb(params, _RENDER_DEFAULTS, "reference_rgb", [76, 84, 94], instance_seed=int(instance_seed), namespace="charts.error_interval"),
        interval_outline_rgb=resolve_render_rgb(params, _RENDER_DEFAULTS, "interval_outline_rgb", [34, 40, 48], instance_seed=int(instance_seed), namespace="charts.error_interval"),
        font_family=sample_font_family(
            role="readout",
            instance_seed=int(instance_seed),
            namespace=f"{SCENE_NAMESPACE}.chart_font",
            params=params,
            exclude_tags=("display",),
            explicit_key="chart_font_family",
            weights_key="chart_font_family_weights",
        ),
        layout_jitter_meta=dict(layout_jitter_meta),
    )


def _draw_text(
    draw: ImageDraw.ImageDraw,
    *,
    xy: Tuple[float, float],
    text: str,
    font,
    fill: RGB,
    stroke_fill: RGB,
    stroke_width: int = 1,
    anchor: str = "la",
) -> BBox:
    """Draw readable chart text and return its pixel bbox for render metadata."""

    try:
        draw_text_traced(draw,
            (float(xy[0]), float(xy[1])),
            str(text),
            font=font,
            fill=tuple(fill),
            stroke_width=max(0, int(stroke_width)),
            stroke_fill=tuple(stroke_fill),
            anchor=str(anchor),
         role="readout", required=False,)
        bbox = draw.textbbox(
            (float(xy[0]), float(xy[1])),
            str(text),
            font=font,
            stroke_width=max(0, int(stroke_width)),
            anchor=str(anchor),
        )
        return _bbox(bbox)
    except Exception:
        draw_text_traced(draw,
            (float(xy[0]), float(xy[1])),
            str(text),
            font=font,
            fill=tuple(fill),
            stroke_width=max(0, int(stroke_width)),
            stroke_fill=tuple(stroke_fill),
         role="readout", required=False,)
        bbox = draw.textbbox((float(xy[0]), float(xy[1])), str(text), font=font, stroke_width=max(0, int(stroke_width)))
        return _bbox(bbox)


def _draw_axis_ticks_horizontal(
    draw: ImageDraw.ImageDraw,
    *,
    p: _RenderParams,
    plot_bbox: Sequence[float],
) -> None:
    left, top, right, bottom = [float(value) for value in plot_bbox]
    tick_font = load_font(p.tick_font_size_px, bold=False, font_family=p.font_family)

    def x_for(value: int) -> float:
        return float(left + ((int(value) - p.axis_min) / max(1, p.axis_max - p.axis_min)) * (right - left))

    for tick in range(int(p.axis_min), int(p.axis_max) + 1, int(p.tick_step)):
        x = x_for(int(tick))
        draw.line([(x, top), (x, bottom)], fill=tuple(p.grid_rgb), width=int(p.grid_line_width_px))
        draw.line([(x, bottom), (x, bottom + 7)], fill=tuple(p.axis_rgb), width=int(p.axis_line_width_px))
        _draw_text(draw, xy=(x, bottom + 12), text=str(tick), font=tick_font, fill=p.muted_text_rgb, stroke_fill=p.text_stroke_rgb, anchor="mt")
    draw.line([(left, bottom), (right, bottom)], fill=tuple(p.axis_rgb), width=int(p.axis_line_width_px))


def _draw_axis_ticks_vertical(
    draw: ImageDraw.ImageDraw,
    *,
    p: _RenderParams,
    plot_bbox: Sequence[float],
) -> None:
    left, top, right, bottom = [float(value) for value in plot_bbox]
    tick_font = load_font(p.tick_font_size_px, bold=False, font_family=p.font_family)

    def y_for(value: int) -> float:
        return float(bottom - ((int(value) - p.axis_min) / max(1, p.axis_max - p.axis_min)) * (bottom - top))

    for tick in range(int(p.axis_min), int(p.axis_max) + 1, int(p.tick_step)):
        y = y_for(int(tick))
        draw.line([(left, y), (right, y)], fill=tuple(p.grid_rgb), width=int(p.grid_line_width_px))
        draw.line([(left - 7, y), (left, y)], fill=tuple(p.axis_rgb), width=int(p.axis_line_width_px))
        _draw_text(draw, xy=(left - 12, y), text=str(tick), font=tick_font, fill=p.muted_text_rgb, stroke_fill=p.text_stroke_rgb, anchor="rm")
    draw.line([(left, top), (left, bottom)], fill=tuple(p.axis_rgb), width=int(p.axis_line_width_px))
    draw.line([(left, bottom), (right, bottom)], fill=tuple(p.axis_rgb), width=int(p.axis_line_width_px))


def _render_horizontal_forest(
    *,
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    dataset: _Dataset,
    p: _RenderParams,
    panel_bbox: Sequence[float],
) -> Tuple[BBox, Dict[str, BBox], Dict[str, BBox], Dict[str, List[float]], Dict[str, Segment], List[Dict[str, Any]]]:
    """Render the forest-plot variant and project category interval geometry."""

    left, top, right, bottom = [float(value) for value in panel_bbox]
    plot_left = left + p.label_band_px
    plot_right = right - p.plot_padding_px
    plot_top = top + p.title_band_height_px + 10
    plot_bottom = bottom - 54
    plot_bbox = _bbox([plot_left, plot_top, plot_right, plot_bottom])
    _draw_axis_ticks_horizontal(draw, p=p, plot_bbox=plot_bbox)

    def x_for(value: int) -> float:
        return float(plot_left + ((int(value) - p.axis_min) / max(1, p.axis_max - p.axis_min)) * (plot_right - plot_left))

    if dataset.reference_value is not None:
        ref_x = x_for(int(dataset.reference_value))
        draw_dashed_line(
            draw,
            start=(ref_x, plot_top),
            end=(ref_x, plot_bottom),
            fill=p.reference_rgb,
            width=2,
            dash_px=10,
            gap_px=7,
        )
        ref_font = load_font(p.tick_font_size_px, bold=True, font_family=p.font_family)
        _draw_text(draw, xy=(ref_x + 6, plot_top - 8), text=f"ref {dataset.reference_value}", font=ref_font, fill=p.reference_rgb, stroke_fill=p.text_stroke_rgb, anchor="lb")

    row_h = float((plot_bottom - plot_top) / max(1, len(dataset.items)))
    label_font = load_font(p.label_font_size_px, bold=True, font_family=p.font_family)
    value_font = load_font(p.value_font_size_px, bold=False, font_family=p.font_family)
    item_bboxes: Dict[str, BBox] = {}
    interval_bboxes: Dict[str, BBox] = {}
    interval_center_points: Dict[str, List[float]] = {}
    interval_segments: Dict[str, Segment] = {}
    entities: List[Dict[str, Any]] = []

    for index, item in enumerate(dataset.items):
        y = float(plot_top + (index + 0.5) * row_h)
        row_top = float(plot_top + index * row_h + 4)
        row_bottom = float(plot_top + (index + 1) * row_h - 4)
        _draw_text(draw, xy=(left + 26, y), text=item.label, font=label_font, fill=p.text_rgb, stroke_fill=p.text_stroke_rgb, anchor="lm")
        x0 = x_for(item.lower)
        xm = x_for(item.midpoint)
        x1 = x_for(item.upper)
        cap = float(p.cap_length_px) * 0.5
        cap_width = max(2, int(p.interval_line_width_px))
        draw.line([(x0, y), (x1, y)], fill=tuple(p.interval_outline_rgb), width=int(p.interval_line_width_px + 2))
        draw.line([(x0, y), (x1, y)], fill=tuple(item.color_rgb), width=int(p.interval_line_width_px))
        draw.line([(x0, y - cap), (x0, y + cap)], fill=tuple(item.color_rgb), width=cap_width)
        draw.line([(x1, y - cap), (x1, y + cap)], fill=tuple(item.color_rgb), width=cap_width)
        r = float(p.point_radius_px)
        draw.ellipse((xm - r, y - r, xm + r, y + r), fill=tuple(item.color_rgb), outline=tuple(p.interval_outline_rgb), width=2)
        _draw_text(draw, xy=(x0, y - cap - 3), text=str(item.lower), font=value_font, fill=p.text_rgb, stroke_fill=p.text_stroke_rgb, anchor="mb")
        _draw_text(draw, xy=(x1, y + cap + 3), text=str(item.upper), font=value_font, fill=p.text_rgb, stroke_fill=p.text_stroke_rgb, anchor="mt")
        mark_pad = max(4.0, float(p.point_radius_px) + 3.0)
        interval_bbox = _bbox([
            min(x0, x1) - mark_pad,
            y - cap - mark_pad,
            max(x0, x1) + mark_pad,
            y + cap + mark_pad,
        ])
        item_bbox = _bbox([left + 12, row_top, right - 12, row_bottom])
        interval_bboxes[item.item_id] = interval_bbox
        interval_center_points[item.item_id] = _bbox([xm, y, xm, y])[:2]
        interval_segments[item.item_id] = [_point(x0, y), _point(x1, y)]
        item_bboxes[item.item_id] = item_bbox
        entities.append(_entity_record(item, interval_bbox=interval_bbox, item_bbox=item_bbox, interval_center_point=interval_center_points[item.item_id], interval_segment=interval_segments[item.item_id]))
    if dataset.reference_value is not None:
        ref_x = x_for(int(dataset.reference_value))
        draw_dashed_line(
            draw,
            start=(ref_x, plot_top),
            end=(ref_x, plot_bottom),
            fill=p.reference_rgb,
            width=2,
            dash_px=10,
            gap_px=7,
        )
    return plot_bbox, item_bboxes, interval_bboxes, interval_center_points, interval_segments, entities


def _render_vertical_dot_whisker(
    *,
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    dataset: _Dataset,
    p: _RenderParams,
    panel_bbox: Sequence[float],
) -> Tuple[BBox, Dict[str, BBox], Dict[str, BBox], Dict[str, List[float]], Dict[str, Segment], List[Dict[str, Any]]]:
    """Render the vertical dot-whisker variant and project interval geometry."""

    del image
    left, top, right, bottom = [float(value) for value in panel_bbox]
    plot_left = left + 82
    plot_right = right - 42
    plot_top = top + p.title_band_height_px + 10
    plot_bottom = bottom - 92
    plot_bbox = _bbox([plot_left, plot_top, plot_right, plot_bottom])
    _draw_axis_ticks_vertical(draw, p=p, plot_bbox=plot_bbox)

    def y_for(value: int) -> float:
        return float(plot_bottom - ((int(value) - p.axis_min) / max(1, p.axis_max - p.axis_min)) * (plot_bottom - plot_top))

    if dataset.reference_value is not None:
        ref_y = y_for(int(dataset.reference_value))
        draw_dashed_line(draw, start=(plot_left, ref_y), end=(plot_right, ref_y), fill=p.reference_rgb, width=2, dash_px=11, gap_px=7)
        ref_font = load_font(p.tick_font_size_px, bold=True, font_family=p.font_family)
        _draw_text(draw, xy=(plot_right - 4, ref_y - 5), text=f"ref {dataset.reference_value}", font=ref_font, fill=p.reference_rgb, stroke_fill=p.text_stroke_rgb, anchor="rb")

    slot_w = float((plot_right - plot_left) / max(1, len(dataset.items)))
    label_font = load_font(p.label_font_size_px, bold=True, font_family=p.font_family)
    value_font = load_font(p.value_font_size_px, bold=False, font_family=p.font_family)
    item_bboxes: Dict[str, BBox] = {}
    interval_bboxes: Dict[str, BBox] = {}
    interval_center_points: Dict[str, List[float]] = {}
    interval_segments: Dict[str, Segment] = {}
    entities: List[Dict[str, Any]] = []

    for index, item in enumerate(dataset.items):
        x = float(plot_left + (index + 0.5) * slot_w)
        y0 = y_for(item.lower)
        ym = y_for(item.midpoint)
        y1 = y_for(item.upper)
        cap = min(float(p.cap_length_px), slot_w * 0.34)
        cap_width = max(2, int(p.interval_line_width_px))
        draw.line([(x, min(y0, y1)), (x, max(y0, y1))], fill=tuple(p.interval_outline_rgb), width=int(p.interval_line_width_px + 2))
        draw.line([(x, min(y0, y1)), (x, max(y0, y1))], fill=tuple(item.color_rgb), width=int(p.interval_line_width_px))
        draw.line([(x - cap, y0), (x + cap, y0)], fill=tuple(item.color_rgb), width=cap_width)
        draw.line([(x - cap, y1), (x + cap, y1)], fill=tuple(item.color_rgb), width=cap_width)
        r = float(p.point_radius_px)
        draw.ellipse((x - r, ym - r, x + r, ym + r), fill=tuple(item.color_rgb), outline=tuple(p.interval_outline_rgb), width=2)
        _draw_text(draw, xy=(x - cap - 3, y0), text=str(item.lower), font=value_font, fill=p.text_rgb, stroke_fill=p.text_stroke_rgb, anchor="rm")
        _draw_text(draw, xy=(x + cap + 3, y1), text=str(item.upper), font=value_font, fill=p.text_rgb, stroke_fill=p.text_stroke_rgb, anchor="lm")
        _draw_text(draw, xy=(x, plot_bottom + 19), text=item.label, font=label_font, fill=p.text_rgb, stroke_fill=p.text_stroke_rgb, anchor="mt")
        mark_pad = max(4.0, float(p.point_radius_px) + 3.0)
        interval_bbox = _bbox([
            x - cap - mark_pad,
            min(y0, y1) - mark_pad,
            x + cap + mark_pad,
            max(y0, y1) + mark_pad,
        ])
        item_bbox = _bbox([x - slot_w * 0.45, plot_top, x + slot_w * 0.45, plot_bottom + 48])
        interval_bboxes[item.item_id] = interval_bbox
        interval_center_points[item.item_id] = _bbox([x, ym, x, ym])[:2]
        interval_segments[item.item_id] = [_point(x, y0), _point(x, y1)]
        item_bboxes[item.item_id] = item_bbox
        entities.append(_entity_record(item, interval_bbox=interval_bbox, item_bbox=item_bbox, interval_center_point=interval_center_points[item.item_id], interval_segment=interval_segments[item.item_id]))
    return plot_bbox, item_bboxes, interval_bboxes, interval_center_points, interval_segments, entities


def _render_bar_with_error(
    *,
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    dataset: _Dataset,
    p: _RenderParams,
    panel_bbox: Sequence[float],
) -> Tuple[BBox, Dict[str, BBox], Dict[str, BBox], Dict[str, List[float]], Dict[str, Segment], List[Dict[str, Any]]]:
    """Render the bar-with-error variant and project interval geometry."""

    del image
    left, top, right, bottom = [float(value) for value in panel_bbox]
    plot_left = left + 82
    plot_right = right - 42
    plot_top = top + p.title_band_height_px + 10
    plot_bottom = bottom - 92
    plot_bbox = _bbox([plot_left, plot_top, plot_right, plot_bottom])
    _draw_axis_ticks_vertical(draw, p=p, plot_bbox=plot_bbox)

    def y_for(value: int) -> float:
        return float(plot_bottom - ((int(value) - p.axis_min) / max(1, p.axis_max - p.axis_min)) * (plot_bottom - plot_top))

    if dataset.reference_value is not None:
        ref_y = y_for(int(dataset.reference_value))
        draw_dashed_line(draw, start=(plot_left, ref_y), end=(plot_right, ref_y), fill=p.reference_rgb, width=2, dash_px=11, gap_px=7)
        ref_font = load_font(p.tick_font_size_px, bold=True, font_family=p.font_family)
        _draw_text(draw, xy=(plot_right - 4, ref_y - 5), text=f"ref {dataset.reference_value}", font=ref_font, fill=p.reference_rgb, stroke_fill=p.text_stroke_rgb, anchor="rb")

    slot_w = float((plot_right - plot_left) / max(1, len(dataset.items)))
    bar_w = max(18.0, float(slot_w) * float(p.bar_width_fraction))
    label_font = load_font(p.label_font_size_px, bold=True, font_family=p.font_family)
    value_font = load_font(p.value_font_size_px, bold=False, font_family=p.font_family)
    item_bboxes: Dict[str, BBox] = {}
    interval_bboxes: Dict[str, BBox] = {}
    interval_center_points: Dict[str, List[float]] = {}
    interval_segments: Dict[str, Segment] = {}
    entities: List[Dict[str, Any]] = []

    for index, item in enumerate(dataset.items):
        x = float(plot_left + (index + 0.5) * slot_w)
        y_lower = y_for(item.lower)
        y_mid = y_for(item.midpoint)
        y_upper = y_for(item.upper)
        baseline_y = y_for(0)
        bar_bbox = (x - bar_w * 0.5, min(y_mid, baseline_y), x + bar_w * 0.5, max(y_mid, baseline_y))
        draw.rounded_rectangle(bar_bbox, radius=5, fill=tuple(item.color_rgb), outline=tuple(p.interval_outline_rgb), width=2)
        cap = min(float(p.cap_length_px), slot_w * 0.36)
        cap_width = max(2, int(p.interval_line_width_px))
        draw.line([(x, min(y_lower, y_upper)), (x, max(y_lower, y_upper))], fill=tuple(p.interval_outline_rgb), width=int(p.interval_line_width_px + 2))
        draw.line([(x, min(y_lower, y_upper)), (x, max(y_lower, y_upper))], fill=tuple(item.color_rgb), width=int(p.interval_line_width_px))
        draw.line([(x - cap, y_lower), (x + cap, y_lower)], fill=tuple(item.color_rgb), width=cap_width)
        draw.line([(x - cap, y_upper), (x + cap, y_upper)], fill=tuple(item.color_rgb), width=cap_width)
        draw.ellipse((x - 4, y_mid - 4, x + 4, y_mid + 4), fill=tuple(p.text_stroke_rgb), outline=tuple(p.interval_outline_rgb), width=1)
        _draw_text(draw, xy=(x - cap - 3, y_lower), text=str(item.lower), font=value_font, fill=p.text_rgb, stroke_fill=p.text_stroke_rgb, anchor="rm")
        _draw_text(draw, xy=(x + cap + 3, y_upper), text=str(item.upper), font=value_font, fill=p.text_rgb, stroke_fill=p.text_stroke_rgb, anchor="lm")
        _draw_text(draw, xy=(x, plot_bottom + 19), text=item.label, font=label_font, fill=p.text_rgb, stroke_fill=p.text_stroke_rgb, anchor="mt")
        mark_half_width = max(float(cap) + 6.0, (float(bar_w) * 0.5) + 4.0)
        interval_bbox = _bbox([
            x - mark_half_width,
            min(float(y_lower), float(y_upper), float(bar_bbox[1])) - 6.0,
            x + mark_half_width,
            max(float(y_lower), float(y_upper), float(bar_bbox[3])) + 6.0,
        ])
        item_bbox = _bbox([x - slot_w * 0.45, plot_top, x + slot_w * 0.45, plot_bottom + 48])
        interval_bboxes[item.item_id] = interval_bbox
        interval_center_points[item.item_id] = _bbox([x, y_mid, x, y_mid])[:2]
        interval_segments[item.item_id] = [_point(x, y_lower), _point(x, y_upper)]
        item_bboxes[item.item_id] = item_bbox
        entities.append(_entity_record(item, interval_bbox=interval_bbox, item_bbox=item_bbox, interval_center_point=interval_center_points[item.item_id], interval_segment=interval_segments[item.item_id]))
    return plot_bbox, item_bboxes, interval_bboxes, interval_center_points, interval_segments, entities


def _entity_record(
    item: _IntervalItem,
    *,
    interval_bbox: Sequence[float],
    item_bbox: Sequence[float],
    interval_center_point: Sequence[float],
    interval_segment: Sequence[Sequence[float]],
) -> Dict[str, Any]:
    return {
        "entity_id": str(item.item_id),
        "entity_type": "error_interval_item",
        "label": str(item.label),
        "lower": int(item.lower),
        "midpoint": int(item.midpoint),
        "upper": int(item.upper),
        "interval_width": int(item.upper) - int(item.lower),
        "bbox_px": list(item_bbox),
        "interval_bbox_px": list(interval_bbox),
        "interval_center_point_px": [round(float(interval_center_point[0]), 3), round(float(interval_center_point[1]), 3)],
        "interval_segment_px": [[round(float(point[0]), 3), round(float(point[1]), 3)] for point in interval_segment],
    }


def _render_chart(
    *,
    background: Image.Image,
    dataset: _Dataset,
    params: Mapping[str, Any],
    instance_seed: int,
    render_params: _RenderParams | None = None,
) -> _Rendered:
    """Render the selected scene variant and return projected interval geometry."""

    p = render_params or _resolve_render_params(params, instance_seed=int(instance_seed))
    image = background.convert("RGB")
    if image.size != (int(p.canvas_width), int(p.canvas_height)):
        image = image.resize((int(p.canvas_width), int(p.canvas_height)))
    draw = ImageDraw.Draw(image)

    panel_bbox = _bbox([
        float(p.outer_margin_left_px),
        float(p.outer_margin_top_px),
        float(p.canvas_width - p.outer_margin_right_px),
        float(p.canvas_height - p.outer_margin_bottom_px),
    ])
    draw_rounded_rect(
        draw,
        tuple(panel_bbox),
        radius=int(p.panel_corner_radius_px),
        fill=p.panel_fill_rgb,
        outline=p.panel_outline_rgb,
        width=int(p.panel_outline_width_px),
    )
    if dataset.scene_variant == "horizontal_forest":
        plot_bbox, item_bboxes, interval_bboxes, interval_center_points, interval_segments, entities = _render_horizontal_forest(
            image=image,
            draw=draw,
            dataset=dataset,
            p=p,
            panel_bbox=panel_bbox,
        )
    elif dataset.scene_variant == "vertical_dot_whisker":
        plot_bbox, item_bboxes, interval_bboxes, interval_center_points, interval_segments, entities = _render_vertical_dot_whisker(
            image=image,
            draw=draw,
            dataset=dataset,
            p=p,
            panel_bbox=panel_bbox,
        )
    elif dataset.scene_variant == "bar_with_error":
        plot_bbox, item_bboxes, interval_bboxes, interval_center_points, interval_segments, entities = _render_bar_with_error(
            image=image,
            draw=draw,
            dataset=dataset,
            p=p,
            panel_bbox=panel_bbox,
        )
    else:
        raise ValueError(f"unsupported scene variant: {dataset.scene_variant}")

    render_meta = {
        "scene_variant": str(dataset.scene_variant),
        "axis_min": int(p.axis_min),
        "axis_max": int(p.axis_max),
        "tick_step": int(p.tick_step),
        "reference_value": dataset.reference_value,
        "style": {
            "panel_fill_rgb": list(p.panel_fill_rgb),
            "panel_outline_rgb": list(p.panel_outline_rgb),
            "axis_rgb": list(p.axis_rgb),
            "grid_rgb": list(p.grid_rgb),
            "reference_rgb": list(p.reference_rgb),
        },
        "layout_jitter": dict(p.layout_jitter_meta),
        "font_assets": {
            "asset_version": font_asset_version(),
            "chart_font_family": str(p.font_family),
        },
    }
    return _Rendered(
        image=image,
        entities=tuple(entities),
        plot_bbox_px=list(plot_bbox),
        item_bboxes_px=dict(item_bboxes),
        interval_bboxes_px=dict(interval_bboxes),
        interval_center_points_px=dict(interval_center_points),
        interval_segments_px=dict(interval_segments),
        render_meta=render_meta,
    )

def render_error_interval_chart(
    background: Image.Image,
    *,
    dataset: _Dataset,
    params: Mapping[str, Any],
    instance_seed: int,
    render_params: _RenderParams | None = None,
) -> _Rendered:
    """Render an already sampled error-interval dataset."""

    return _render_chart(
        background=background,
        dataset=dataset,
        params=params,
        instance_seed=int(instance_seed),
        render_params=render_params,
    )


def resolve_error_interval_render_params(params: Mapping[str, Any], *, instance_seed: int) -> _RenderParams:
    """Resolve render parameters for callers that need canvas-level metadata."""

    return _resolve_render_params(params, instance_seed=int(instance_seed))


__all__ = [
    "render_error_interval_chart",
    "resolve_error_interval_render_params",
]
