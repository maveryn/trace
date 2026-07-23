"""Rendering primitives for survey-traverse diagrams."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

from PIL import ImageDraw

from trace_tasks.tasks.geometry.shared.diagram_style import (
    GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM,
    geometry_diagram_style_metadata,
    prepare_geometry_diagram_style_and_background,
)
from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_from_points, bbox_to_list, pad_bbox
from trace_tasks.tasks.geometry.shared.vector2d import point_to_list
from trace_tasks.tasks.shared.text_rendering import load_font

from .measurements import direction_endpoint
from .state import (
    BBox,
    DEGREE_SYMBOL,
    SCENE_ID,
    AreaOffsetCase,
    BearingTurnCase,
    ElevationLevelingCase,
    Point,
    RenderContext,
    RenderedAreaScene,
)

BEARING_BBOX_ANNOTATION_KEYS = ("turn_diagram", "field_note_region")
ELEVATION_BBOX_ANNOTATION_KEYS = ("station_profile", "field_note_region")
AREA_BBOX_ANNOTATION_KEYS = ("traverse_region", "field_note_region")


def _union_bboxes(bboxes: Sequence[BBox], *, width: int, height: int, pad: float = 0.0) -> BBox:
    """Return one clamped bbox covering the supplied visible regions."""

    x0 = min(float(bbox[0]) for bbox in bboxes)
    y0 = min(float(bbox[1]) for bbox in bboxes)
    x1 = max(float(bbox[2]) for bbox in bboxes)
    y1 = max(float(bbox[3]) for bbox in bboxes)
    return pad_bbox((x0, y0, x1, y1), float(pad), width=int(width), height=int(height))


def create_render_context(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
) -> tuple[RenderContext, Dict[str, Any]]:
    """Prepare one styled technical-diagram canvas for a survey-traverse sample."""

    width = int(render_defaults.get("canvas_width", 820))
    height = int(render_defaults.get("canvas_height", 580))
    background, background_meta, style, style_meta = prepare_geometry_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=width,
        canvas_height=height,
        allow_dark=True,
        style_profile=str(render_defaults.get("style_profile", GEOMETRY_STYLE_PROFILE_ANALYTICAL_DIAGRAM)),
    )
    draw = ImageDraw.Draw(background)
    line_width = int(render_defaults.get("line_width", max(2, int(style.axis_stroke_width_px))))
    label_size = int(render_defaults.get("label_font_size", 22))
    small_size = int(render_defaults.get("small_label_font_size", 18))
    tiny_size = int(render_defaults.get("tiny_label_font_size", 14))
    ctx = RenderContext(
        image=background,
        draw=draw,
        width=width,
        height=height,
        line_color=tuple(style.axis_rgb),
        secondary_color=tuple(style.grid_major_rgb),
        guide_color=tuple(style.grid_minor_rgb),
        label_color=tuple(style.label_rgb),
        label_stroke_color=tuple(style.label_stroke_rgb),
        panel_fill=tuple(style.panel_fill_rgb),
        panel_alt_fill=tuple(style.panel_alt_fill_rgb),
        accent_color=tuple(style.accent_rgb),
        secondary_accent_color=tuple(style.secondary_accent_rgb),
        line_width=line_width,
        label_stroke_width=0,
        font=load_font(label_size, bold=False, font_family=str(render_defaults.get("readout_font_family", "roboto"))),
        small_font=load_font(small_size, bold=False, font_family=str(render_defaults.get("readout_font_family", "roboto"))),
        tiny_font=load_font(tiny_size, bold=False, font_family=str(render_defaults.get("readout_font_family", "roboto"))),
        diagram_style_meta=dict(style_meta),
        background_meta=dict(background_meta),
    )
    render_meta = {
        "style": geometry_diagram_style_metadata(style),
        "background": dict(background_meta),
    }
    return ctx, render_meta


def _draw_label(
    ctx: RenderContext,
    xy: Point,
    text: str,
    *,
    font: Any | None = None,
    fill: tuple[int, int, int] | None = None,
    anchor: str = "mm",
) -> BBox:
    bbox = ctx.draw.textbbox(
        (float(xy[0]), float(xy[1])),
        str(text),
        font=font or ctx.font,
        anchor=str(anchor),
        stroke_width=int(ctx.label_stroke_width),
    )
    ctx.draw.text(
        (float(xy[0]), float(xy[1])),
        str(text),
        font=font or ctx.font,
        fill=fill or ctx.label_color,
        anchor=str(anchor),
        stroke_width=int(ctx.label_stroke_width),
        stroke_fill=ctx.label_stroke_color,
    )
    return (float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]))


def _draw_arrow(ctx: RenderContext, start: Point, end: Point, *, fill: tuple[int, int, int], width: int | None = None) -> None:
    line_width = int(width or ctx.line_width)
    ctx.draw.line([start, end], fill=fill, width=line_width)
    angle = math.atan2(float(end[1]) - float(start[1]), float(end[0]) - float(start[0]))
    arrow_len = 12.0 + line_width * 1.2
    for delta in (math.radians(150), math.radians(-150)):
        tip = (
            float(end[0]) + math.cos(angle + delta) * arrow_len,
            float(end[1]) + math.sin(angle + delta) * arrow_len,
        )
        ctx.draw.line([end, tip], fill=fill, width=line_width)


def _draw_station(ctx: RenderContext, point: Point, label: str, *, label_offset: Point = (0.0, -24.0)) -> BBox:
    radius = 5.0
    ctx.draw.ellipse(
        [point[0] - radius, point[1] - radius, point[0] + radius, point[1] + radius],
        fill=ctx.accent_color,
        outline=ctx.line_color,
        width=max(1, ctx.line_width - 1),
    )
    return _draw_label(
        ctx,
        (float(point[0]) + float(label_offset[0]), float(point[1]) + float(label_offset[1])),
        str(label),
        font=ctx.small_font,
    )


def _draw_dashed_line(ctx: RenderContext, start: Point, end: Point, *, fill: tuple[int, int, int], width: int, dash: float = 10.0, gap: float = 7.0) -> None:
    """Draw a dashed guide segment."""

    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    length = math.hypot(x1 - x0, y1 - y0)
    if length <= 0.0:
        return
    ux = (x1 - x0) / length
    uy = (y1 - y0) / length
    offset = 0.0
    while offset < length:
        segment_end = min(length, offset + float(dash))
        ctx.draw.line(
            [(x0 + ux * offset, y0 + uy * offset), (x0 + ux * segment_end, y0 + uy * segment_end)],
            fill=fill,
            width=int(width),
        )
        offset += float(dash) + float(gap)


def _draw_turn_arc(
    ctx: RenderContext,
    center: Point,
    *,
    start_bearing: int,
    turn_angle: int,
    turn_direction: str,
    radius: float,
    label: str,
) -> None:
    """Draw the non-reflex turn arc from the incoming course to the outgoing course."""

    signed_step = -1.0 if str(turn_direction) == "left" else 1.0
    samples = max(8, int(abs(int(turn_angle)) // 4) + 2)
    arc_points = [
        direction_endpoint(center, float(start_bearing) + signed_step * float(turn_angle) * idx / float(samples - 1), radius)
        for idx in range(samples)
    ]
    ctx.draw.line(arc_points, fill=ctx.secondary_accent_color, width=max(2, ctx.line_width - 1))
    mid_bearing = float(start_bearing) + signed_step * float(turn_angle) / 2.0
    label_point = direction_endpoint(center, mid_bearing, radius + 34.0)
    _draw_label(ctx, label_point, str(label), font=ctx.tiny_font, fill=ctx.secondary_accent_color)


def _draw_note_box(ctx: RenderContext, bbox: BBox, lines: Sequence[str]) -> Dict[str, BBox]:
    ctx.draw.rounded_rectangle(bbox, radius=8, fill=ctx.panel_alt_fill, outline=ctx.secondary_color, width=2)
    line_bboxes: Dict[str, BBox] = {}
    row_gap = max(20.0, (float(bbox[3]) - float(bbox[1]) - 36.0) / max(1, len(lines)))
    for row, text in enumerate(lines):
        y = float(bbox[1]) + 20.0 + row * row_gap
        line_bboxes[f"note_{row}"] = _draw_label(ctx, (float(bbox[0]) + 16.0, y), text, font=ctx.tiny_font, anchor="lm")
    return line_bboxes


def _draw_staff(ctx: RenderContext, base: Point, *, height: float = 86.0) -> BBox:
    x, y = base
    ctx.draw.line([(x, y), (x, y - height)], fill=ctx.line_color, width=max(2, ctx.line_width - 1))
    for step in range(0, 5):
        yy = y - step * (height / 4.0)
        ctx.draw.line([(x - 10.0, yy), (x + 10.0, yy)], fill=ctx.line_color, width=1)
    return pad_bbox((x - 10.0, y - height, x + 10.0, y), 4.0, width=ctx.width, height=ctx.height)


def render_closed_traverse_scene(ctx: RenderContext, case: BearingTurnCase, *, instance_seed: int) -> RenderedAreaScene:
    """Render a three-station traverse turn where the middle vertex binds the turn."""

    labels = case.station_labels
    panel = (64.0, 62.0, 530.0, 468.0)
    ctx.draw.rounded_rectangle(panel, radius=14, fill=ctx.panel_fill, outline=ctx.secondary_color, width=2)
    station_b = (316.0, 284.0)
    leg_length = 170.0
    incoming_bearing = (int(case.base_bearing) + 180) % 360
    station_a = direction_endpoint(station_b, incoming_bearing, leg_length)
    station_c = direction_endpoint(station_b, int(case.answer), leg_length)

    label_bboxes: Dict[str, BBox] = {}
    label_bboxes[labels[0]] = _draw_station(ctx, station_a, labels[0])
    label_bboxes[labels[1]] = _draw_station(ctx, station_b, labels[1], label_offset=(22.0, -18.0))
    label_bboxes[labels[2]] = _draw_station(ctx, station_c, labels[2])
    ctx.draw.line([station_a, station_b, station_c], fill=ctx.line_color, width=ctx.line_width)
    incoming_arrow_start = direction_endpoint(station_b, incoming_bearing, 74.0)
    incoming_arrow_end = direction_endpoint(station_b, incoming_bearing, 30.0)
    _draw_arrow(ctx, incoming_arrow_start, incoming_arrow_end, fill=ctx.line_color, width=max(2, ctx.line_width - 1))
    outgoing_arrow_start = direction_endpoint(station_b, int(case.answer), 28.0)
    outgoing_arrow_end = direction_endpoint(station_b, int(case.answer), 74.0)
    _draw_arrow(ctx, outgoing_arrow_start, outgoing_arrow_end, fill=ctx.line_color, width=max(2, ctx.line_width - 1))
    north_end = (station_b[0], station_b[1] - 92.0)
    _draw_arrow(ctx, station_b, north_end, fill=ctx.guide_color, width=max(2, ctx.line_width - 1))
    label_bboxes["north"] = _draw_label(ctx, (north_end[0], north_end[1] - 15), "N", font=ctx.small_font)
    incoming_guide_end = direction_endpoint(station_b, int(case.base_bearing), 90.0)
    _draw_dashed_line(ctx, station_b, incoming_guide_end, fill=ctx.guide_color, width=max(2, ctx.line_width - 1))
    known_mid = direction_endpoint(station_b, int(case.base_bearing), 112.0)
    target_mid = direction_endpoint(station_b, int(case.answer), 84.0)
    label_bboxes["known_bearing"] = _draw_label(ctx, known_mid, f"in {case.base_bearing}{DEGREE_SYMBOL}", font=ctx.tiny_font)
    label_bboxes["target"] = _draw_label(ctx, (target_mid[0] + 16.0, target_mid[1] - 4.0), "?", font=ctx.font, fill=ctx.accent_color)
    _draw_turn_arc(
        ctx,
        station_b,
        start_bearing=int(case.base_bearing),
        turn_angle=int(case.turn_angle),
        turn_direction=str(case.turn_direction),
        radius=58.0,
        label=f"{case.turn_direction} {case.turn_angle}{DEGREE_SYMBOL}",
    )
    note_bbox = (554.0, 90.0, float(ctx.width) - 70.0, 228.0)
    _draw_note_box(ctx, note_bbox, [f"At {labels[1]}", f"bearing in {case.base_bearing}{DEGREE_SYMBOL}", f"turn {case.turn_direction} {case.turn_angle}{DEGREE_SYMBOL}"])

    annotation_bboxes = {
        "turn_diagram": pad_bbox(panel, 4.0, width=ctx.width, height=ctx.height),
        "field_note_region": pad_bbox(note_bbox, 4.0, width=ctx.width, height=ctx.height),
    }
    return RenderedAreaScene(
        image=ctx.image,
        annotation_bboxes=dict(annotation_bboxes),
        annotation_roles=BEARING_BBOX_ANNOTATION_KEYS,
        scene_entities=(
            {"id": "station_a", "type": "survey_station", "label": labels[0], "point": point_to_list(station_a)},
            {"id": "station_b", "type": "survey_station", "label": labels[1], "point": point_to_list(station_b)},
            {"id": "station_c", "type": "survey_station", "label": labels[2], "point": point_to_list(station_c)},
        ),
        label_bboxes=dict(label_bboxes),
        render_map={
            "station_points": {
                labels[0]: point_to_list(station_a),
                labels[1]: point_to_list(station_b),
                labels[2]: point_to_list(station_c),
            },
            "annotation_bboxes": {key: bbox_to_list(value) for key, value in annotation_bboxes.items()},
            "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
            "field_note_bbox": bbox_to_list(note_bbox),
            "panel_bbox": bbox_to_list(panel),
        },
        witness={
            "known_bearing": int(case.base_bearing),
            "target_bearing": int(case.answer),
            "turn_angle": int(case.turn_angle),
            "turn_direction": str(case.turn_direction),
            "station_labels": list(labels),
        },
    )


def render_leveling_station_scene(ctx: RenderContext, case: ElevationLevelingCase, *, instance_seed: int) -> RenderedAreaScene:
    """Render a leveling field-note diagram with station and note-center witnesses."""

    labels = case.station_labels
    panel = (62.0, 92.0, 520.0, 452.0)
    ctx.draw.rounded_rectangle(panel, radius=14, fill=ctx.panel_fill, outline=ctx.secondary_color, width=2)
    reference_station = (164.0, 384.0)
    target_station = (420.0, 356.0)
    terrain_points = [
        (panel[0] + 18.0, 402.0),
        (reference_station[0], reference_station[1] + 8.0),
        (282.0, 372.0),
        (target_station[0], target_station[1] + 8.0),
        (panel[2] - 18.0, 388.0),
    ]
    ctx.draw.line(terrain_points, fill=ctx.guide_color, width=max(2, ctx.line_width - 1))
    label_bboxes: Dict[str, BBox] = {}
    label_bboxes[labels[0]] = _draw_station(ctx, reference_station, labels[0], label_offset=(-22.0, -18.0))
    label_bboxes[labels[1]] = _draw_station(ctx, target_station, labels[1], label_offset=(24.0, -18.0))
    _draw_staff(ctx, reference_station)
    _draw_staff(ctx, target_station)
    label_bboxes["backsight_label"] = _draw_label(
        ctx,
        (reference_station[0] + 42.0, reference_station[1] - 52.0),
        f"backsight {case.backsight}",
        font=ctx.tiny_font,
        anchor="lm",
    )
    label_bboxes["foresight_label"] = _draw_label(
        ctx,
        (target_station[0] + 42.0, target_station[1] - 52.0),
        f"foresight {case.foresight}",
        font=ctx.tiny_font,
        anchor="lm",
    )
    sight_left = (reference_station[0], reference_station[1] - 86.0)
    sight_right = (target_station[0], target_station[1] - 86.0)
    ctx.draw.line([sight_left, sight_right], fill=ctx.secondary_accent_color, width=ctx.line_width)
    measurement_mid = ((sight_left[0] + sight_right[0]) / 2.0, (sight_left[1] + sight_right[1]) / 2.0)
    label_bboxes["level_line"] = _draw_label(ctx, (measurement_mid[0], measurement_mid[1] - 16.0), "level line", font=ctx.tiny_font)
    label_bboxes["known_elevation"] = _draw_label(ctx, (reference_station[0] - 6.0, reference_station[1] + 32.0), f"{case.reference_elevation}", font=ctx.small_font)
    label_bboxes["target_unknown"] = _draw_label(ctx, (target_station[0] + 8.0, target_station[1] + 32.0), "?", font=ctx.font, fill=ctx.accent_color)
    note_bbox = (548.0, 70.0, float(ctx.width) - 70.0, 190.0)
    label_bboxes.update(
        _draw_note_box(
            ctx,
            note_bbox,
            [
                f"benchmark {labels[0]} = {case.reference_elevation}",
                f"backsight = {case.backsight}",
                f"foresight = {case.foresight}",
            ],
        )
    )
    annotation_bboxes = {
        "station_profile": pad_bbox(panel, 4.0, width=ctx.width, height=ctx.height),
        "field_note_region": pad_bbox(note_bbox, 4.0, width=ctx.width, height=ctx.height),
    }
    return RenderedAreaScene(
        image=ctx.image,
        annotation_bboxes=dict(annotation_bboxes),
        annotation_roles=ELEVATION_BBOX_ANNOTATION_KEYS,
        scene_entities=(
            {"id": "reference_station", "type": "survey_station", "label": labels[0], "point": point_to_list(reference_station)},
            {"id": "target_station", "type": "survey_station", "label": labels[1], "point": point_to_list(target_station)},
        ),
        label_bboxes=dict(label_bboxes),
        render_map={
            "station_points": {labels[0]: point_to_list(reference_station), labels[1]: point_to_list(target_station)},
            "annotation_bboxes": {key: bbox_to_list(value) for key, value in annotation_bboxes.items()},
            "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
            "field_note_bbox": bbox_to_list(note_bbox),
            "measurement_line_bbox": bbox_to_list(bbox_from_points((sight_left, sight_right), width=ctx.width, height=ctx.height, pad=6.0)),
            "panel_bbox": bbox_to_list(panel),
        },
        witness={
            "reference_elevation": int(case.reference_elevation),
            "backsight": int(case.backsight),
            "foresight": int(case.foresight),
            "height_of_instrument": int(case.height_of_instrument),
            "target_elevation": int(case.answer),
            "station_labels": list(labels),
        },
    )


def render_offset_area_scene(ctx: RenderContext, case: AreaOffsetCase, *, instance_seed: int) -> RenderedAreaScene:
    """Render baseline-offset traverse data with shape, note, and baseline witnesses."""

    labels = case.station_labels
    chainages = tuple(int(value) for value in case.chainages)
    offsets = tuple(int(value) for value in case.offsets)
    panel = (56.0, 60.0, 524.0, 500.0)
    ctx.draw.rounded_rectangle(panel, radius=14, fill=ctx.panel_fill, outline=ctx.secondary_color, width=2)
    x0, y0, x1, y1 = (96.0, 128.0, 476.0, 432.0)
    max_chain = max(chainages)
    max_offset = max(offsets)
    x_scale = (x1 - x0 - 28.0) / max(1, max_chain)
    y_scale = (y1 - y0 - 38.0) / max(1, max_offset)
    baseline_y = y1 - 24.0
    baseline_points = [(x0 + 14.0 + chain * x_scale, baseline_y) for chain in chainages]
    offset_points = [(base[0], baseline_y - offset * y_scale) for base, offset in zip(baseline_points, offsets)]
    shape_points = [baseline_points[0], baseline_points[-1]] + list(reversed(offset_points))
    ctx.draw.line([baseline_points[0], baseline_points[-1]], fill=ctx.line_color, width=ctx.line_width)
    ctx.draw.polygon(shape_points, fill=None, outline=ctx.accent_color)
    ctx.draw.line(offset_points, fill=ctx.accent_color, width=ctx.line_width + 1)
    label_bboxes: Dict[str, BBox] = {}
    for label, chain, offset, base, top in zip(labels, chainages, offsets, baseline_points, offset_points):
        ctx.draw.line([base, top], fill=ctx.secondary_accent_color, width=max(2, ctx.line_width - 1))
        label_bboxes[label] = _draw_station(ctx, top, str(label), label_offset=(0.0, -18.0))
        label_bboxes[f"{label}_offset"] = _draw_label(ctx, (top[0] + 18.0, (top[1] + base[1]) / 2.0), str(offset), font=ctx.tiny_font)
        label_bboxes[f"{label}_chain"] = _draw_label(ctx, (base[0], base[1] + 20.0), str(chain), font=ctx.tiny_font)
    label_bboxes["baseline"] = _draw_label(ctx, ((baseline_points[0][0] + baseline_points[-1][0]) / 2.0, baseline_y + 42.0), "baseline chainage", font=ctx.tiny_font)
    note_bbox = (548.0, 83.0, float(ctx.width) - 68.0, 220.0)
    note_lines = [f"{label}: ch {chain}, off {offset}" for label, chain, offset in zip(labels, chainages, offsets)]
    label_bboxes.update(_draw_note_box(ctx, note_bbox, note_lines))
    _draw_label(ctx, (float(note_bbox[0]) + 16.0, float(note_bbox[3]) + 30.0), "Area = ?", font=ctx.small_font, anchor="lm", fill=ctx.accent_color)
    traverse_bbox = bbox_from_points(shape_points, width=ctx.width, height=ctx.height, pad=14.0)
    area_reference_bbox = bbox_from_points(baseline_points, width=ctx.width, height=ctx.height, pad=18.0)
    annotation_bboxes = {
        "traverse_region": _union_bboxes((traverse_bbox, area_reference_bbox), width=ctx.width, height=ctx.height, pad=0.0),
        "field_note_region": pad_bbox(note_bbox, 4.0, width=ctx.width, height=ctx.height),
    }
    return RenderedAreaScene(
        image=ctx.image,
        annotation_bboxes=dict(annotation_bboxes),
        annotation_roles=AREA_BBOX_ANNOTATION_KEYS,
        scene_entities=tuple(
            {
                "id": f"offset_{label}",
                "type": "survey_offset",
                "label": str(label),
                "chainage": int(chain),
                "offset": int(offset),
                "point": point_to_list(top),
            }
            for label, chain, offset, top in zip(labels, chainages, offsets, offset_points)
        ),
        label_bboxes=dict(label_bboxes),
        render_map={
            "offset_points": {label: point_to_list(point) for label, point in zip(labels, offset_points)},
            "baseline_points": {label: point_to_list(point) for label, point in zip(labels, baseline_points)},
            "chainages": [int(value) for value in chainages],
            "offsets": [int(value) for value in offsets],
            "label_bboxes": {key: bbox_to_list(value) for key, value in label_bboxes.items()},
            "field_note_bbox": bbox_to_list(note_bbox),
            "traverse_bbox": bbox_to_list(traverse_bbox),
            "area_reference_bbox": bbox_to_list(area_reference_bbox),
            "panel_bbox": bbox_to_list(panel),
        },
        witness={
            "chainages": [int(value) for value in chainages],
            "offsets": [int(value) for value in offsets],
            "station_labels": list(labels),
        },
    )


__all__ = [
    "AREA_BBOX_ANNOTATION_KEYS",
    "BEARING_BBOX_ANNOTATION_KEYS",
    "ELEVATION_BBOX_ANNOTATION_KEYS",
    "create_render_context",
    "render_closed_traverse_scene",
    "render_leveling_station_scene",
    "render_offset_area_scene",
]
