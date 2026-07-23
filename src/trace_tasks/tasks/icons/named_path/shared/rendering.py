"""Neutral renderer for the named-path icons scene."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.text_rendering import draw_text_centered, load_font
from ...shared.icon_noise import serialize_icon_noise_edits
from ...shared.icon_scene import BBox, draw_single_panel, resolve_single_panel_layout, single_panel_geometry_to_trace
from ...shared.procedural_named_icon_field_scene import (
    bbox_center_float,
    bbox_from_center_dimensions,
    bbox_inside,
    boxes_overlap,
    label_bbox_for_icon,
    render_planned_named_icon_sprite,
    union_bbox,
)
from ...shared.procedural_named_icons import procedural_named_icon_display_name

from .defaults import SCENE_ID
from .state import IconPlan, PathScenePayload, RenderedPathIcon


def _route_points(
    *,
    rng,
    stop_count: int,
    content_bbox: BBox,
    render_params: Mapping[str, Any],
) -> Tuple[Tuple[float, float], ...]:
    x0, y0, x1, y1 = tuple(int(value) for value in content_bbox)
    horizontal_margin = max(30, int(render_params["path_horizontal_margin_px"]))
    vertical_margin = max(40, int(render_params["path_vertical_margin_px"]))
    left = float(x0 + horizontal_margin)
    right = float(x1 - horizontal_margin)
    if right <= left:
        raise ValueError("content panel is too narrow for path route")
    top = float(y0 + vertical_margin)
    bottom = float(y1 - vertical_margin)
    if bottom <= top:
        raise ValueError("content panel is too short for path route")
    mid_y = 0.5 * (top + bottom)
    max_amplitude = min(0.42 * (bottom - top), float(render_params["path_amplitude_max_px"]))
    min_amplitude = min(max_amplitude, float(render_params["path_amplitude_min_px"]))
    amplitude = float(rng.uniform(float(min_amplitude), float(max_amplitude))) if max_amplitude > 0 else 0.0
    phase = float(rng.uniform(0.0, 2.0 * math.pi))
    secondary_phase = float(rng.uniform(0.0, 2.0 * math.pi))
    points: List[Tuple[float, float]] = []
    for index in range(int(stop_count)):
        t = 0.0 if int(stop_count) <= 1 else float(index) / float(int(stop_count) - 1)
        x = left + t * (right - left)
        y = mid_y + amplitude * math.sin((2.0 * math.pi * 1.15 * t) + phase)
        y += 0.28 * amplitude * math.sin((2.0 * math.pi * 2.25 * t) + secondary_phase)
        y = max(top, min(bottom, y))
        points.append((float(x), float(y)))
    return tuple(points)


def _draw_label_badge(
    *,
    image: Image.Image,
    icon_bbox: BBox,
    label: str,
    content_bbox: BBox,
    label_font,
    render_params: Mapping[str, Any],
) -> BBox:
    label_bbox = label_bbox_for_icon(
        icon_bbox=tuple(int(value) for value in icon_bbox),
        label=str(label),
        content_bbox=tuple(int(value) for value in content_bbox),
        font=label_font,
        padding_px=int(render_params["candidate_label_padding_px"]),
        gap_px=int(render_params["candidate_label_gap_px"]),
    )
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        label_bbox,
        radius=max(4, int(round(0.28 * float(label_bbox[3] - label_bbox[1])))),
        fill=tuple(int(value) for value in render_params["candidate_label_background_rgb"]) + (240,),
        outline=tuple(int(value) for value in render_params["candidate_label_border_rgb"]) + (255,),
        width=1,
    )
    draw_text_centered(
        draw,
        text=str(label),
        center=bbox_center_float(label_bbox),
        font=label_font,
        fill=tuple(int(value) for value in render_params["candidate_label_color_rgb"]),
        stroke_fill=tuple(int(value) for value in render_params["candidate_label_stroke_rgb"]),
        stroke_width=1,
    )
    return tuple(int(value) for value in label_bbox)


def _label_occupancy_bbox(
    *,
    icon_bbox: BBox,
    label: str,
    content_bbox: BBox,
    label_font,
    render_params: Mapping[str, Any],
) -> BBox:
    if not str(label):
        return tuple(int(value) for value in icon_bbox)
    label_bbox = label_bbox_for_icon(
        icon_bbox=tuple(int(value) for value in icon_bbox),
        label=str(label),
        content_bbox=tuple(int(value) for value in content_bbox),
        font=label_font,
        padding_px=int(render_params["candidate_label_padding_px"]),
        gap_px=int(render_params["candidate_label_gap_px"]),
    )
    return union_bbox(tuple(int(value) for value in icon_bbox), tuple(int(value) for value in label_bbox))


def _draw_endpoint_label(
    *,
    image: Image.Image,
    text: str,
    center_xy: Tuple[float, float],
    content_bbox: BBox,
    font,
    render_params: Mapping[str, Any],
) -> None:
    draw = ImageDraw.Draw(image)
    text_bbox = draw.textbbox((0, 0), str(text), font=font)
    width = int(text_bbox[2] - text_bbox[0]) + 12
    height = int(text_bbox[3] - text_bbox[1]) + 8
    x0 = int(round(float(center_xy[0]) - 0.5 * float(width)))
    y0 = int(round(float(center_xy[1]) - 42.0))
    x0 = max(int(content_bbox[0]), min(int(content_bbox[2]) - width, x0))
    y0 = max(int(content_bbox[1]), min(int(content_bbox[3]) - height, y0))
    box = (int(x0), int(y0), int(x0 + width), int(y0 + height))
    draw.rounded_rectangle(
        box,
        radius=6,
        fill=tuple(int(value) for value in render_params["endpoint_label_background_rgb"]) + (230,),
        outline=tuple(int(value) for value in render_params["stop_outline_rgb"]) + (255,),
        width=1,
    )
    draw_text_centered(
        draw,
        text=str(text),
        center=bbox_center_float(box),
        font=font,
        fill=tuple(int(value) for value in render_params["endpoint_label_color_rgb"]),
        stroke_fill=tuple(int(value) for value in render_params["endpoint_label_stroke_rgb"]),
        stroke_width=1,
    )


def _draw_path_underlay(
    *,
    image: Image.Image,
    points: Sequence[Tuple[float, float]],
    content_bbox: BBox,
    render_params: Mapping[str, Any],
) -> None:
    """Draw the route scaffold and endpoint labels before path-stop icons."""

    draw = ImageDraw.Draw(image)
    path_alpha = max(0, min(255, int(render_params.get("path_line_alpha", 145))))
    path_color = tuple(int(value) for value in render_params["path_color_rgb"]) + (path_alpha,)
    draw.line(
        [(float(x), float(y)) for x, y in points],
        fill=path_color,
        width=int(render_params["path_stroke_width_px"]),
        joint="curve",
    )
    radius = max(2, int(render_params["path_stop_radius_px"]))
    for cx, cy in points:
        box = (
            int(round(cx - radius)),
            int(round(cy - radius)),
            int(round(cx + radius)),
            int(round(cy + radius)),
        )
        draw.ellipse(
            box,
            fill=tuple(int(value) for value in render_params["stop_fill_rgb"]) + (255,),
            outline=tuple(int(value) for value in render_params["stop_outline_rgb"]) + (255,),
            width=2,
        )
    endpoint_font = load_font(int(render_params["endpoint_label_font_size_px"]), bold=True)
    _draw_endpoint_label(
        image=image,
        text="START",
        center_xy=tuple(points[0]),
        content_bbox=content_bbox,
        font=endpoint_font,
        render_params=render_params,
    )
    _draw_endpoint_label(
        image=image,
        text="END",
        center_xy=tuple(points[-1]),
        content_bbox=content_bbox,
        font=endpoint_font,
        render_params=render_params,
    )


def render_named_path_scene(
    *,
    rng,
    plans: Sequence[IconPlan],
    answer_label: str,
    target_shape_id: str,
    target_shape_name: str,
    target_occurrence_count: int,
    stop_count: int,
    distractor_count: int,
    selected_position: int,
    answer_position: int,
    neighbor_direction: str,
    target_positions: Sequence[int],
    option_positions: Sequence[int],
    labels_by_position: Mapping[int, str],
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...],
    render_params: Mapping[str, Any],
) -> PathScenePayload:
    """Render a named icon path and preserve projection state."""

    layout = resolve_single_panel_layout(
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        reserve_title=False,
    )
    content_bbox = tuple(int(value) for value in layout.scene_content_xyxy)
    label_font = load_font(int(render_params["candidate_label_font_size_px"]), bold=True)
    sprites = [render_planned_named_icon_sprite(plan) for plan in plans]
    path_points = _route_points(
        rng=rng,
        stop_count=int(stop_count),
        content_bbox=content_bbox,
        render_params=render_params,
    )
    occupancy: List[BBox] = []
    icon_bboxes: List[BBox] = []
    for plan, sprite in zip(plans, sprites):
        center = tuple(float(value) for value in path_points[int(plan.position_index)])
        bbox = bbox_from_center_dimensions(center, width=int(sprite.size[0]), height=int(sprite.size[1]))
        occupancy_bbox = _label_occupancy_bbox(
            icon_bbox=bbox,
            label=str(plan.label),
            content_bbox=content_bbox,
            label_font=label_font,
            render_params=render_params,
        )
        if not bbox_inside(occupancy_bbox, content_bbox):
            raise ValueError("path stop icon or label is outside content panel")
        if any(boxes_overlap(occupancy_bbox, other, gap_px=int(render_params["icon_collision_gap_px"])) for other in occupancy):
            raise ValueError("path stop icon placement overlaps")
        occupancy.append(tuple(int(value) for value in occupancy_bbox))
        icon_bboxes.append(tuple(int(value) for value in bbox))

    image = Image.new("RGBA", (int(layout.canvas_width), int(layout.canvas_height)))
    draw_single_panel(
        image=image,
        layout=layout,
        background_rgb=tuple(int(value) for value in render_params["background_color_rgb"]),
        panel_fill_rgb=tuple(int(value) for value in render_params["panel_fill_rgb"]),
        panel_border_rgb=tuple(int(value) for value in render_params["panel_border_rgb"]),
        title_color_rgb=tuple(int(value) for value in render_params["header_text_rgb"]),
        corner_radius_px=int(render_params["panel_corner_radius_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        scene_title="",
        icon_canvas_style=render_params.get("_icon_canvas_style_object"),
    )
    _draw_path_underlay(image=image, points=path_points, content_bbox=content_bbox, render_params=render_params)
    for _plan, sprite, bbox in zip(plans, sprites, icon_bboxes):
        image.alpha_composite(sprite, (int(bbox[0]), int(bbox[1])))
    for plan, bbox in zip(plans, icon_bboxes):
        if str(plan.label):
            _draw_label_badge(
                image=image,
                icon_bbox=tuple(int(value) for value in bbox),
                label=str(plan.label),
                content_bbox=content_bbox,
                label_font=label_font,
                render_params=render_params,
            )

    target_rank_by_position = {int(position): int(rank) for rank, position in enumerate(target_positions)}
    rendered_icons: List[RenderedPathIcon] = []
    for plan, bbox in zip(plans, icon_bboxes):
        rendered_icons.append(
            RenderedPathIcon(
                instance_id=f"path_stop_{int(plan.position_index):02d}",
                position_index=int(plan.position_index),
                role=str(plan.role),
                label=str(plan.label),
                shape_id=str(plan.shape_id),
                shape_name=procedural_named_icon_display_name(str(plan.shape_id)),
                color_name=str(plan.color_name),
                tint_rgb=tuple(int(channel) for channel in plan.tint_rgb),
                fill_style=str(plan.fill_style),
                bbox_xyxy=tuple(int(value) for value in bbox),
                center_xy=bbox_center_float(bbox),
                nominal_size_px=int(plan.nominal_size_px),
                rotation_degrees=int(plan.rotation_degrees),
                target_occurrence_rank=(
                    int(target_rank_by_position[int(plan.position_index)])
                    if int(plan.position_index) in target_rank_by_position
                    else None
                ),
                is_query_occurrence=int(plan.position_index) == int(selected_position),
                is_answer_neighbor=int(plan.position_index) == int(answer_position),
                noise_edits=tuple(serialize_icon_noise_edits(plan.noise_edits)),
                noise_seed=plan.noise_seed,
            )
        )
    return PathScenePayload(
        image=image.convert("RGB"),
        answer_label=str(answer_label),
        target_shape_id=str(target_shape_id),
        target_shape_name=str(target_shape_name),
        target_occurrence_count=int(target_occurrence_count),
        stop_count=int(stop_count),
        distractor_count=int(distractor_count),
        query_position_index=int(selected_position),
        answer_position_index=int(answer_position),
        neighbor_direction=str(neighbor_direction),
        target_positions=tuple(int(value) for value in target_positions),
        option_positions=tuple(int(value) for value in option_positions),
        labels_by_position={int(key): str(value) for key, value in labels_by_position.items()},
        path_points_xy=tuple((float(x), float(y)) for x, y in path_points),
        icons=tuple(sorted(rendered_icons, key=lambda icon: int(icon.position_index))),
        panel_geometry=single_panel_geometry_to_trace(layout),
        sampled_palette_rgb=tuple(sampled_palette_rgb),
    )


def serialize_path_icon(icon: RenderedPathIcon) -> Dict[str, Any]:
    """Return one JSON-serializable path icon payload."""

    return {
        "entity_kind": "procedural_named_icon",
        "instance_id": str(icon.instance_id),
        "position_index": int(icon.position_index),
        "role": str(icon.role),
        "label": str(icon.label),
        "shape_id": str(icon.shape_id),
        "shape_name": str(icon.shape_name),
        "color_name": str(icon.color_name),
        "tint_rgb": [int(value) for value in icon.tint_rgb],
        "fill_style": str(icon.fill_style),
        "bbox_xyxy": [int(value) for value in icon.bbox_xyxy],
        "center_xy": [float(icon.center_xy[0]), float(icon.center_xy[1])],
        "nominal_size_px": int(icon.nominal_size_px),
        "rotation_degrees": int(icon.rotation_degrees),
        "target_occurrence_rank": None if icon.target_occurrence_rank is None else int(icon.target_occurrence_rank),
        "is_query_occurrence": bool(icon.is_query_occurrence),
        "is_answer_neighbor": bool(icon.is_answer_neighbor),
        "noise_edits": [dict(edit) for edit in icon.noise_edits],
        "noise_seed": None if icon.noise_seed is None else int(icon.noise_seed),
    }
