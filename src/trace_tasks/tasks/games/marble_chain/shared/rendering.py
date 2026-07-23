"""Rendering helpers for marble-chain game scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.tasks.games.shared.layout import (
    apply_games_layout_jitter_to_bbox,
    attach_games_unit_size_jitter,
    resolve_games_layout_jitter,
    resolve_games_unit_size_scale,
    scale_games_px,
)
from trace_tasks.tasks.games.shared.scene_style import (
    draw_panel_scene_chrome,
    make_panel_scene_background,
    resolve_game_panel_scene_style,
)
from trace_tasks.tasks.games.shared.text import draw_game_text_traced as draw_text_traced
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .defaults import DEFAULTS
from .rules import marble_entity_id
from .state import MarbleSample, RenderedMarbleScene
from .styles import COLOR_RGB, MARBLE_CHAIN_STYLE_RGB


def _int_default(params: Mapping[str, Any], render_defaults: Mapping[str, Any], key: str, fallback: int) -> int:
    if str(key) in params:
        return int(params[str(key)])
    return int(group_default(render_defaults, str(key), int(fallback)))


def _draw_text_center(
    draw: ImageDraw.ImageDraw,
    bbox: Tuple[float, float, float, float],
    text: str,
    *,
    font: Any,
    fill: Tuple[int, int, int],
    stroke_width: int = 1,
) -> None:
    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=int(stroke_width))
    width = float(text_bbox[2] - text_bbox[0])
    height = float(text_bbox[3] - text_bbox[1])
    x0, y0, x1, y1 = bbox
    origin = (
        float(x0 + ((x1 - x0) - width) / 2.0 - float(text_bbox[0])),
        float(y0 + ((y1 - y0) - height) / 2.0 - float(text_bbox[1])),
    )
    draw_text_traced(
        draw,
        origin,
        str(text),
        font=font,
        fill=tuple(int(value) for value in fill),
        stroke_width=int(stroke_width),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(fill)),
        role="readout",
        required=False,
    )


def _track_point(t: float, *, variant: str, bbox: Tuple[float, float, float, float]) -> Tuple[float, float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0 + 18.0
    base_radius = min(x1 - x0, y1 - y0) * 0.40
    if str(variant) == "spiral_track":
        theta = math.radians(145.0 + 610.0 * float(t))
        radius = base_radius * (0.36 + 0.64 * float(t))
    elif str(variant) == "double_arc_track":
        theta = math.radians(140.0 + 430.0 * float(t))
        radius = base_radius * (0.78 + 0.24 * math.sin(2.0 * math.pi * float(t)))
    else:
        theta = math.radians(155.0 + 250.0 * float(t))
        radius = base_radius
    x = cx + radius * math.cos(theta)
    y = cy + radius * math.sin(theta)
    return float(x), float(y)


def _chain_centers(count: int, *, variant: str, bbox: Tuple[float, float, float, float]) -> Tuple[Tuple[float, float], ...]:
    if int(count) <= 1:
        return (_track_point(0.5, variant=str(variant), bbox=bbox),)
    return tuple(
        _track_point(float(index) / float(int(count) - 1), variant=str(variant), bbox=bbox)
        for index in range(int(count))
    )


def _slot_center(slot_index: int, centers: Sequence[Tuple[float, float]]) -> Tuple[float, float]:
    slot = int(slot_index)
    if not centers:
        return 0.0, 0.0
    if slot <= 0:
        x0, y0 = centers[0]
        x1, y1 = centers[1] if len(centers) > 1 else (x0 + 42.0, y0)
        return float(x0 - 0.5 * (x1 - x0)), float(y0 - 0.5 * (y1 - y0))
    if slot >= len(centers):
        x0, y0 = centers[-2] if len(centers) > 1 else (centers[-1][0] - 42.0, centers[-1][1])
        x1, y1 = centers[-1]
        return float(x1 + 0.5 * (x1 - x0)), float(y1 + 0.5 * (y1 - y0))
    x0, y0 = centers[slot - 1]
    x1, y1 = centers[slot]
    return float((x0 + x1) / 2.0), float((y0 + y1) / 2.0)


def _circle_bbox(center: Tuple[float, float], radius: float) -> List[float]:
    x, y = center
    r = float(radius)
    return [round(float(x - r), 3), round(float(y - r), 3), round(float(x + r), 3), round(float(y + r), 3)]


def _draw_marble(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[float],
    *,
    fill_rgb: Tuple[int, int, int],
    outline_rgb: Tuple[int, int, int],
    highlight_rgb: Tuple[int, int, int],
    width: int,
) -> None:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    draw.ellipse((x0, y0, x1, y1), fill=tuple(fill_rgb), outline=tuple(outline_rgb), width=int(width))
    inset = max(4.0, (x1 - x0) * 0.18)
    draw.ellipse(
        (x0 + inset, y0 + inset, x0 + inset * 1.75, y0 + inset * 1.75),
        fill=tuple(highlight_rgb),
    )


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    fill: Tuple[int, int, int],
    width: int,
    dash_px: float = 12.0,
    gap_px: float = 9.0,
) -> None:
    x0, y0 = start
    x1, y1 = end
    length = math.hypot(float(x1 - x0), float(y1 - y0))
    if length <= 0:
        return
    ux = float(x1 - x0) / length
    uy = float(y1 - y0) / length
    cursor = 0.0
    while cursor < length:
        seg_end = min(length, cursor + float(dash_px))
        draw.line(
            (
                x0 + ux * cursor,
                y0 + uy * cursor,
                x0 + ux * seg_end,
                y0 + uy * seg_end,
            ),
            fill=tuple(fill),
            width=int(width),
        )
        cursor += float(dash_px) + float(gap_px)


def _arrow_bbox(start: Tuple[float, float], end: Tuple[float, float], *, pad: float) -> List[float]:
    x0, y0 = start
    x1, y1 = end
    return [
        round(min(float(x0), float(x1)) - float(pad), 3),
        round(min(float(y0), float(y1)) - float(pad), 3),
        round(max(float(x0), float(x1)) + float(pad), 3),
        round(max(float(y0), float(y1)) + float(pad), 3),
    ]


def _shot_label_center(
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    avoid_centers: Sequence[Tuple[float, float]],
    bounds: Tuple[float, float, float, float],
    label_radius: float,
) -> Tuple[float, float]:
    """Choose a label marker position along the shot path away from marbles."""

    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    length = max(1.0, math.hypot(ex - sx, ey - sy))
    ux = float(ex - sx) / length
    uy = float(ey - sy) / length
    px, py = -uy, ux
    x0, y0, x1, y1 = [float(value) for value in bounds]
    margin = float(label_radius) + 5.0
    candidates: list[Tuple[float, float]] = []
    for fraction in (0.44, 0.54, 0.64, 0.74):
        base_x = sx + (ex - sx) * float(fraction)
        base_y = sy + (ey - sy) * float(fraction)
        for offset in (label_radius + 16.0, label_radius + 27.0, 0.0):
            if offset == 0.0:
                candidate_offsets = (0.0,)
            else:
                candidate_offsets = (float(offset), -float(offset))
            for signed_offset in candidate_offsets:
                cx = float(base_x + px * signed_offset)
                cy = float(base_y + py * signed_offset)
                if x0 + margin <= cx <= x1 - margin and y0 + margin <= cy <= y1 - margin:
                    candidates.append((cx, cy))
    if not candidates:
        return float(sx + (ex - sx) * 0.55), float(sy + (ey - sy) * 0.55)

    def score(candidate: Tuple[float, float]) -> Tuple[float, float]:
        cx, cy = candidate
        nearest = min(
            (
                math.hypot(float(cx - float(ax)), float(cy - float(ay)))
                for ax, ay in avoid_centers
            ),
            default=9999.0,
        )
        edge_margin = min(cx - x0, x1 - cx, cy - y0, y1 - cy)
        return float(nearest), float(edge_margin)

    return max(candidates, key=score)


def _draw_shot_arrow(
    draw: ImageDraw.ImageDraw,
    start: Tuple[float, float],
    end: Tuple[float, float],
    *,
    line_rgb: Tuple[int, int, int],
    label: str | None,
    label_font: Any,
    text_rgb: Tuple[int, int, int],
    label_fill_rgb: Tuple[int, int, int],
    label_outline_rgb: Tuple[int, int, int],
    width: int,
    emphasize: bool,
    label_center: Tuple[float, float] | None = None,
    label_radius: float = 18.0,
) -> List[float]:
    """Draw one shooter-to-gap arrow and return its full visual bbox."""

    x0, y0 = start
    x1, y1 = end
    length = max(1.0, math.hypot(float(x1 - x0), float(y1 - y0)))
    ux = float(x1 - x0) / length
    uy = float(y1 - y0) / length
    shaft_start = (float(x0 + ux * 34.0), float(y0 + uy * 34.0))
    shaft_end = (float(x1 - ux * 18.0), float(y1 - uy * 18.0))
    line_width = int(width + (2 if bool(emphasize) else 0))
    draw.line((shaft_start[0], shaft_start[1], shaft_end[0], shaft_end[1]), fill=tuple(line_rgb), width=line_width)
    head_len = 20.0 + (4.0 if bool(emphasize) else 0.0)
    head_w = 12.0 + (3.0 if bool(emphasize) else 0.0)
    tip = (float(x1 - ux * 8.0), float(y1 - uy * 8.0))
    base = (float(tip[0] - ux * head_len), float(tip[1] - uy * head_len))
    px = -uy
    py = ux
    draw.polygon(
        (
            tip,
            (base[0] + px * head_w, base[1] + py * head_w),
            (base[0] - px * head_w, base[1] - py * head_w),
        ),
        fill=tuple(line_rgb),
    )
    bbox = _arrow_bbox(shaft_start, tip, pad=max(18.0, float(line_width) * 2.0))
    if label:
        if label_center is None:
            label_cx = float(x0 + (x1 - x0) * 0.55 + px * (float(label_radius) + 16.0))
            label_cy = float(y0 + (y1 - y0) * 0.55 + py * (float(label_radius) + 16.0))
        else:
            label_cx = float(label_center[0])
            label_cy = float(label_center[1])
        label_bbox = (
            label_cx - float(label_radius),
            label_cy - float(label_radius),
            label_cx + float(label_radius),
            label_cy + float(label_radius),
        )
        draw.ellipse(label_bbox, fill=tuple(label_fill_rgb), outline=tuple(label_outline_rgb), width=2)
        _draw_text_center(draw, label_bbox, str(label), font=label_font, fill=text_rgb, stroke_width=0)
        bbox = [
            min(float(bbox[0]), float(label_bbox[0])),
            min(float(bbox[1]), float(label_bbox[1])),
            max(float(bbox[2]), float(label_bbox[2])),
            max(float(bbox[3]), float(label_bbox[3])),
        ]
    return [round(float(value), 3) for value in bbox]


def render_marble_scene(
    *,
    sample: MarbleSample,
    instance_seed: int,
    params: Mapping[str, Any],
    style_variant: str,
    render_defaults: Mapping[str, Any],
    namespace: str,
) -> RenderedMarbleScene:
    """Render board, shooter, shot arrows, and entity maps from task-owned state."""

    canvas_width = _int_default(params, render_defaults, "canvas_width", DEFAULTS.canvas_width)
    canvas_height = _int_default(params, render_defaults, "canvas_height", DEFAULTS.canvas_height)
    margin = _int_default(params, render_defaults, "panel_margin_px", DEFAULTS.panel_margin_px)
    style, style_meta = resolve_game_panel_scene_style(
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.marble_chain_panel_style",
        treatment_weights=group_default(render_defaults, "panel_scene_treatment_weights", None),
        palette_weights=group_default(render_defaults, "panel_scene_palette_weights", None),
    )
    image, background_meta = make_panel_scene_background(
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        style=style,
    )
    image = image.convert("RGBA")
    draw = ImageDraw.Draw(image)
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.marble_chain.label_font",
        params=params,
    )
    marble_style = MARBLE_CHAIN_STYLE_RGB.get(
        str(style_variant),
        MARBLE_CHAIN_STYLE_RGB["classic_track"],
    )
    layout_jitter = resolve_games_layout_jitter(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.marble_chain.layout",
    )
    unit_scale, unit_meta = resolve_games_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{str(namespace)}.marble_chain.unit_size",
    )
    base_track_bbox = (
        float(margin),
        float(_int_default(params, render_defaults, "track_panel_top_px", DEFAULTS.track_panel_top_px)),
        float(canvas_width - margin),
        float(
            _int_default(params, render_defaults, "track_panel_top_px", DEFAULTS.track_panel_top_px)
            + _int_default(params, render_defaults, "track_panel_height_px", DEFAULTS.track_panel_height_px)
        ),
    )
    track_bbox, _dx, _dy, resolved_jitter = apply_games_layout_jitter_to_bbox(
        bbox_px=base_track_bbox,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        jitter=layout_jitter,
    )
    draw_panel_scene_chrome(
        draw,
        bbox=tuple(int(round(value)) for value in track_bbox),
        style=style,
        radius=22,
        border_width=3,
    )

    label_font = load_font(
        _int_default(params, render_defaults, "label_font_size_px", DEFAULTS.label_font_size_px),
        bold=True,
        font_family=str(font_family),
    )
    text_rgb = tuple(int(value) for value in style.text_rgb)
    border_rgb = tuple(int(value) for value in marble_style["rail"])
    accent_rgb = tuple(int(value) for value in marble_style["arrow"])
    track_rgb = tuple(int(value) for value in marble_style["track"])
    shooter_body_rgb = tuple(int(value) for value in marble_style["shooter_body"])
    guide_rgb = tuple(int(max(80, min(210, value))) for value in style.grid_rgb)

    radius_base = _int_default(params, render_defaults, "marble_radius_px", DEFAULTS.marble_radius_px)
    radius = float(scale_games_px(radius_base, float(unit_scale), min_px=14))
    centers = _chain_centers(len(sample.chain_colors), variant=str(sample.scene_variant), bbox=track_bbox)
    if len(centers) >= 2:
        min_spacing = min(
            math.hypot(float(centers[index + 1][0] - centers[index][0]), float(centers[index + 1][1] - centers[index][1]))
            for index in range(len(centers) - 1)
        )
        radius = min(float(radius), max(14.0, (float(min_spacing) - 5.0) / 2.0))
    track_width = scale_games_px(
        _int_default(params, render_defaults, "track_width_px", DEFAULTS.track_width_px),
        float(unit_scale),
        min_px=4,
    )

    track_points = [
        _track_point(float(i) / 180.0, variant=str(sample.scene_variant), bbox=track_bbox)
        for i in range(181)
    ]
    draw.line(track_points, fill=tuple(border_rgb), width=int(track_width + 5), joint="curve")
    draw.line(track_points, fill=track_rgb, width=int(track_width), joint="curve")

    entities: List[Dict[str, Any]] = []
    entity_bboxes: Dict[str, List[float]] = {}
    entity_points: Dict[str, List[float]] = {}
    chain_specs: List[Dict[str, Any]] = []
    shot_specs: List[Dict[str, Any]] = []

    shooter_center = (float((track_bbox[0] + track_bbox[2]) / 2.0), float((track_bbox[1] + track_bbox[3]) / 2.0 + 18.0))
    guide_radius = min(float(track_bbox[2] - track_bbox[0]), float(track_bbox[3] - track_bbox[1])) * 0.46
    for angle_deg in range(0, 360, 45):
        theta = math.radians(float(angle_deg))
        end = (shooter_center[0] + guide_radius * math.cos(theta), shooter_center[1] + guide_radius * math.sin(theta))
        _draw_dashed_line(draw, shooter_center, end, fill=guide_rgb, width=1, dash_px=9.0, gap_px=10.0)

    hole_center = track_points[-1]
    hole_radius = max(20.0, float(radius) * 1.05)
    hole_bbox = _circle_bbox(hole_center, hole_radius)
    draw.ellipse(tuple(hole_bbox), fill=(8, 8, 8), outline=tuple(border_rgb), width=2)
    entity_bboxes["track_black_hole"] = [float(value) for value in hole_bbox]
    entities.append(
        {
            "entity_id": "track_black_hole",
            "entity_type": "track_black_hole",
            "bbox_px": [float(value) for value in hole_bbox],
        }
    )

    for index, (color_key, center) in enumerate(zip(sample.chain_colors, centers)):
        entity_id = marble_entity_id(int(index))
        bbox = _circle_bbox(center, radius)
        _draw_marble(
            draw,
            bbox,
            fill_rgb=tuple(int(value) for value in COLOR_RGB[str(color_key)]),
            outline_rgb=border_rgb,
            highlight_rgb=(255, 255, 255),
            width=2,
        )
        entity_bboxes[str(entity_id)] = [float(value) for value in bbox]
        entity_points[str(entity_id)] = [round(float(center[0]), 3), round(float(center[1]), 3)]
        spec = {
            "entity_id": str(entity_id),
            "entity_type": "chain_marble",
            "index": int(index),
            "color_key": str(color_key),
            "bbox_px": [float(value) for value in bbox],
            "center_px": [round(float(center[0]), 3), round(float(center[1]), 3)],
        }
        chain_specs.append(dict(spec))
        entities.append(dict(spec))

    shooter_body_radius = max(44.0, float(radius) * 2.0)
    shooter_body = (
        (shooter_center[0], shooter_center[1] - shooter_body_radius * 0.96),
        (shooter_center[0] - shooter_body_radius * 0.88, shooter_center[1] + shooter_body_radius * 0.66),
        (shooter_center[0] + shooter_body_radius * 0.88, shooter_center[1] + shooter_body_radius * 0.66),
    )
    draw.polygon(shooter_body, fill=tuple(shooter_body_rgb), outline=tuple(border_rgb))
    shooter_bbox = _circle_bbox(shooter_center, max(13.0, float(radius) * 0.72))
    _draw_marble(
        draw,
        shooter_bbox,
        fill_rgb=tuple(int(value) for value in COLOR_RGB[str(sample.shooter_color)]),
        outline_rgb=border_rgb,
        highlight_rgb=(255, 255, 255),
        width=3,
    )
    entity_bboxes["shooter_marble"] = [float(value) for value in shooter_bbox]
    entity_points["shooter_marble"] = [round(float(shooter_center[0]), 3), round(float(shooter_center[1]), 3)]
    entities.append(
        {
            "entity_id": "shooter_marble",
            "entity_type": "shooter_marble",
            "color_key": str(sample.shooter_color),
            "bbox_px": [float(value) for value in shooter_bbox],
            "center_px": [round(float(shooter_center[0]), 3), round(float(shooter_center[1]), 3)],
        }
    )

    option_label_centers: List[Tuple[float, float]] = []
    for option in sample.option_specs:
        arrow_end = _slot_center(int(option.slot_index), centers)
        label_radius = 18.0
        label_center = _shot_label_center(
            shooter_center,
            arrow_end,
            avoid_centers=tuple(centers) + (shooter_center,) + tuple(option_label_centers),
            bounds=track_bbox,
            label_radius=float(label_radius),
        )
        bbox = _draw_shot_arrow(
            draw,
            shooter_center,
            arrow_end,
            line_rgb=accent_rgb,
            label=str(option.label),
            label_font=label_font,
            text_rgb=text_rgb,
            label_fill_rgb=tuple(style.option_fill_rgb),
            label_outline_rgb=accent_rgb,
            width=4,
            emphasize=False,
            label_center=label_center,
            label_radius=float(label_radius),
        )
        option_label_centers.append((float(label_center[0]), float(label_center[1])))
        entity_bboxes[str(option.entity_id)] = [float(value) for value in bbox]
        entity_points[str(option.entity_id)] = [round(float(arrow_end[0]), 3), round(float(arrow_end[1]), 3)]
        spec = {
            "entity_id": str(option.entity_id),
            "entity_type": "shot_direction_arrow",
            "label": str(option.label),
            "slot_index": int(option.slot_index),
            "pop_count": int(option.outcome.pop_count),
            "remaining_count": int(option.outcome.remaining_count),
            "popped_indices": [int(index) for index in option.outcome.popped_indices],
            "is_answer": bool(option.is_answer),
            "bbox_px": [float(value) for value in bbox],
            "insertion_point_px": [round(float(arrow_end[0]), 3), round(float(arrow_end[1]), 3)],
            "label_center_px": [round(float(label_center[0]), 3), round(float(label_center[1]), 3)],
        }
        shot_specs.append(dict(spec))
        entities.append(dict(spec))

    marked_shot_spec = None
    if sample.marked_slot_index is not None:
        arrow_end = _slot_center(int(sample.marked_slot_index), centers)
        bbox = _draw_shot_arrow(
            draw,
            shooter_center,
            arrow_end,
            line_rgb=accent_rgb,
            label=None,
            label_font=label_font,
            text_rgb=text_rgb,
            label_fill_rgb=tuple(style.option_fill_rgb),
            label_outline_rgb=accent_rgb,
            width=7,
            emphasize=True,
        )
        entity_bboxes["marked_shot_arrow"] = [float(value) for value in bbox]
        entity_points["marked_shot_arrow"] = [round(float(arrow_end[0]), 3), round(float(arrow_end[1]), 3)]
        marked_shot_spec = {
            "entity_id": "marked_shot_arrow",
            "entity_type": "marked_shot_arrow",
            "slot_index": int(sample.marked_slot_index),
            "pop_count": int(sample.marked_outcome.pop_count if sample.marked_outcome else 0),
            "remaining_count": int(sample.marked_outcome.remaining_count if sample.marked_outcome else len(sample.chain_colors)),
            "popped_indices": [int(index) for index in (sample.marked_outcome.popped_indices if sample.marked_outcome else ())],
            "bbox_px": [float(value) for value in bbox],
            "insertion_point_px": [round(float(arrow_end[0]), 3), round(float(arrow_end[1]), 3)],
        }
        entities.append(dict(marked_shot_spec))

    render_map = {
        "entity_bboxes_px": dict(entity_bboxes),
        "entity_points_px": dict(entity_points),
        "chain_marble_bboxes_px": {
            str(spec["entity_id"]): [float(value) for value in spec["bbox_px"]]
            for spec in chain_specs
        },
        "chain_marble_centers_px": {
            str(spec["entity_id"]): [float(value) for value in spec["center_px"]]
            for spec in chain_specs
        },
        "shot_arrow_bboxes_px": {
            str(spec["entity_id"]): [float(value) for value in spec["bbox_px"]]
            for spec in shot_specs
        },
        "shot_arrow_insertion_points_px": {
            str(spec["entity_id"]): [float(value) for value in spec["insertion_point_px"]]
            for spec in shot_specs
        },
        "marked_shot_arrow_bbox_px": None if marked_shot_spec is None else [float(value) for value in marked_shot_spec["bbox_px"]],
        "marked_shot_arrow_insertion_point_px": None
        if marked_shot_spec is None
        else [float(value) for value in marked_shot_spec["insertion_point_px"]],
        "scene_variant": str(sample.scene_variant),
        "panel_scene_style": {
            key: value
            for key, value in dict(style_meta).items()
            if key not in {"text_legibility", "text_color_policy"}
        },
        "marble_chain_style": {
            "style_variant": str(style_variant),
            **{str(key): [int(value) for value in rgb] for key, rgb in marble_style.items()},
        },
        "text_style": {
            "font_family": str(font_family),
            "font_asset": get_font_family_record(str(font_family)).to_trace(),
        },
        "layout_jitter": attach_games_unit_size_jitter(resolved_jitter, unit_meta),
    }
    return RenderedMarbleScene(
        image=image.convert("RGB"),
        entities=tuple(entities),
        render_map=dict(render_map),
        style_meta={
            key: value
            for key, value in dict(style_meta).items()
            if key not in {"text_legibility", "text_color_policy"}
        },
        background_meta=dict(background_meta),
    )


__all__ = ["render_marble_scene"]
