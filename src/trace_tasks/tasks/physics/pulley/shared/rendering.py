"""Rendering and layout primitives for pulley diagrams."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any, Dict, List, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.physics.shared.style import build_physics_pulley_theme
from trace_tasks.tasks.shared.bbox_projection import bbox_union_many
from trace_tasks.tasks.shared.drawing import draw_arrow, draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.shared.render_variation import resolve_layout_jitter
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .state import PulleySceneSpec, RenderedPulleyScene, SCENE_NAMESPACE


def _draw_block_texture(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    line_rgb: Tuple[int, int, int],
    spacing_px: int,
    width_px: int,
) -> None:
    """Draw subtle diagonal hatching inside one pulley block."""

    left, top, right, bottom = [float(value) for value in bbox]
    span = int(bottom - top)
    offset = -span
    while offset < int(right - left) + span:
        draw.line(
            [(float(left + offset), float(top)), (float(left + offset + span), float(bottom))],
            fill=tuple(int(value) for value in line_rgb),
            width=max(1, int(width_px)),
        )
        offset += max(8, int(spacing_px))


def _draw_pulley(
    draw: ImageDraw.ImageDraw,
    *,
    center_x: float,
    center_y: float,
    radius_px: int,
    hub_radius_px: int,
    fill_rgb: Tuple[int, int, int],
    outline_rgb: Tuple[int, int, int],
    width_px: int,
) -> List[float]:
    """Draw one pulley wheel and return its bbox."""

    radius = float(radius_px)
    bbox = [
        round(float(center_x - radius), 3),
        round(float(center_y - radius), 3),
        round(float(center_x + radius), 3),
        round(float(center_y + radius), 3),
    ]
    draw.ellipse(
        tuple(float(value) for value in bbox),
        fill=tuple(int(value) for value in fill_rgb),
        outline=tuple(int(value) for value in outline_rgb),
        width=max(2, int(width_px)),
    )
    hub_radius = float(hub_radius_px)
    draw.ellipse(
        (
            float(center_x - hub_radius),
            float(center_y - hub_radius),
            float(center_x + hub_radius),
            float(center_y + hub_radius),
        ),
        fill=tuple(int(value) for value in outline_rgb),
        outline=tuple(int(value) for value in outline_rgb),
    )
    return list(bbox)


def _draw_text_tag(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font,
    fill_rgb: Tuple[int, int, int],
    outline_rgb: Tuple[int, int, int],
    text_rgb: Tuple[int, int, int],
    stroke_width_px: int,
) -> List[float]:
    """Draw one rounded label tag and return its outer bbox."""

    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width_px)))
    text_width = float(text_bbox[2] - text_bbox[0])
    text_height = float(text_bbox[3] - text_bbox[1])
    pad_x = 14.0
    pad_y = 9.0
    center_x, center_y = float(center[0]), float(center[1])
    tag_bbox = [
        round(float(center_x - (0.5 * text_width) - pad_x), 3),
        round(float(center_y - (0.5 * text_height) - pad_y), 3),
        round(float(center_x + (0.5 * text_width) + pad_x), 3),
        round(float(center_y + (0.5 * text_height) + pad_y), 3),
    ]
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in tag_bbox),
        radius=10,
        fill=tuple(int(value) for value in fill_rgb),
        outline=tuple(int(value) for value in outline_rgb),
        width=max(2, int(stroke_width_px)),
    )
    text_draw_bbox = draw_centered_text(
        draw,
        text=str(text),
        center=(float(center_x), float(center_y)),
        font=font,
        fill=tuple(int(value) for value in text_rgb),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(text_rgb)),
        stroke_width=max(1, int(stroke_width_px)),
    )
    return bbox_union_many(tag_bbox, text_draw_bbox)


def render_pulley_scene(
    *,
    background: Image.Image,
    render_defaults: Mapping[str, Any],
    accent_color_name: str,
    scene_spec: PulleySceneSpec,
    diagram_style: Any | None = None,
    font_family: str | None = None,
) -> RenderedPulleyScene:
    """Render one single-system pulley diagram and return trace metadata."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    theme = build_physics_pulley_theme(str(accent_color_name), diagram_style=diagram_style)
    layout_offset_x = float(render_defaults.get("layout_offset_x_px", 0))
    layout_offset_y = float(render_defaults.get("layout_offset_y_px", 0))
    label_font = load_font(int(render_defaults["label_font_size_px"]), bold=True, font_family=font_family)
    small_label_font = load_font(int(render_defaults["small_label_font_size_px"]), bold=True, font_family=font_family)

    top_left = float(render_defaults["top_block_x_px"]) + float(layout_offset_x)
    top_y = float(render_defaults["top_block_y_px"]) + float(layout_offset_y)
    top_width = float(render_defaults["top_block_width_px"])
    top_height = float(render_defaults["top_block_height_px"])
    top_bbox = [
        round(float(top_left), 3),
        round(float(top_y), 3),
        round(float(top_left + top_width), 3),
        round(float(top_y + top_height), 3),
    ]
    top_center_x = float((top_bbox[0] + top_bbox[2]) / 2.0)
    if str(scene_spec.scene_variant) == "compact_block":
        lower_y = float(render_defaults["compact_lower_block_y_px"]) + float(layout_offset_y)
    elif str(scene_spec.scene_variant) == "tall_block":
        lower_y = float(render_defaults["tall_lower_block_y_px"]) + float(layout_offset_y)
    else:
        lower_y = float(render_defaults["lower_block_y_px"]) + float(layout_offset_y)

    segment_gap = float(render_defaults["support_segment_gap_px"])
    total_slots = int(scene_spec.support_segment_count) + int(scene_spec.disconnected_segment_count)
    lower_width = max(250.0, ((int(total_slots) - 1) * segment_gap) + 130.0)
    lower_height = float(render_defaults["lower_block_height_px"])
    lower_bbox = [
        round(float(top_center_x - (0.5 * lower_width)), 3),
        round(float(lower_y), 3),
        round(float(top_center_x + (0.5 * lower_width)), 3),
        round(float(lower_y + lower_height), 3),
    ]

    scene_entities: List[Dict[str, Any]] = [
        {
            "entity_id": "fixed_upper_block",
            "entity_type": "fixed_pulley_block",
            "bbox_px": list(top_bbox),
            "meta": {"scene_variant": str(scene_spec.scene_variant)},
        },
        {
            "entity_id": "moving_lower_block",
            "entity_type": "moving_pulley_block",
            "bbox_px": list(lower_bbox),
            "meta": {
                "support_segment_count": int(scene_spec.support_segment_count),
                "disconnected_segment_count": int(scene_spec.disconnected_segment_count),
            },
        },
    ]

    for block_bbox in (top_bbox, lower_bbox):
        draw_rounded_rect(
            draw,
            tuple(float(value) for value in block_bbox),
            radius=14,
            fill=tuple(int(value) for value in theme.frame_fill_rgb),
            outline=tuple(int(value) for value in theme.frame_outline_rgb),
            width=4,
        )

    slot_xs = {
        int(slot_index): float(top_center_x + ((int(slot_index) - ((int(total_slots) - 1) / 2.0)) * segment_gap))
        for slot_index in range(int(total_slots))
    }
    rope_width = int(render_defaults["rope_width_px"])
    pulley_radius = int(render_defaults["pulley_radius_px"])
    hub_radius = int(render_defaults["pulley_hub_radius_px"])
    top_pulley_y = float((top_bbox[1] + top_bbox[3]) / 2.0)
    lower_pulley_y = float((lower_bbox[1] + lower_bbox[3]) / 2.0)
    rope_top_y = float(top_bbox[3] + 2.0)
    rope_bottom_y = float(lower_bbox[1] - 2.0)
    span = float(rope_bottom_y - rope_top_y)

    support_segment_bboxes: List[List[float]] = []
    support_segment_map: Dict[str, List[float]] = {}
    cut_segment_bboxes: List[List[float]] = []
    cut_segment_map: Dict[str, List[float]] = {}
    pulley_bboxes: List[List[float]] = []
    connected_slots = set(int(slot_index) for slot_index in scene_spec.connected_slot_indices)
    top_cut_slots = {int(segment.x_order) for segment in scene_spec.cut_segments if str(segment.attach_side) == "top"}
    bottom_cut_slots = {int(segment.x_order) for segment in scene_spec.cut_segments if str(segment.attach_side) == "bottom"}

    for support_index, slot_index in enumerate(sorted(connected_slots), start=1):
        x_value = float(slot_xs[int(slot_index)])
        segment_bbox = [
            round(float(x_value - (rope_width / 2.0)), 3),
            round(float(rope_top_y), 3),
            round(float(x_value + (rope_width / 2.0)), 3),
            round(float(rope_bottom_y), 3),
        ]
        draw.line(
            [(float(x_value), float(rope_top_y)), (float(x_value), float(rope_bottom_y))],
            fill=tuple(int(value) for value in theme.rope_rgb),
            width=max(2, int(rope_width)),
        )
        entity_id = f"support_segment_{int(support_index)}"
        support_segment_bboxes.append(list(segment_bbox))
        support_segment_map[entity_id] = list(segment_bbox)
        scene_entities.append(
            {
                "entity_id": entity_id,
                "entity_type": "supporting_rope_segment",
                "bbox_px": list(segment_bbox),
                "meta": {
                    "slot_index": int(slot_index),
                    "segment_index": int(support_index),
                    "supports_moving_block": True,
                    "connects_upper_and_lower_blocks": True,
                },
            }
        )

    cap_radius = float(render_defaults["cut_endpoint_radius_px"])
    for cut_segment in scene_spec.cut_segments:
        x_value = float(slot_xs[int(cut_segment.x_order)])
        cut_y = float(rope_top_y + (span * float(cut_segment.cut_fraction)))
        if str(cut_segment.attach_side) == "top":
            start_y, end_y = float(rope_top_y), float(cut_y)
            cap_y = float(end_y)
        else:
            start_y, end_y = float(cut_y), float(rope_bottom_y)
            cap_y = float(start_y)
        draw.line(
            [(float(x_value), float(start_y)), (float(x_value), float(end_y))],
            fill=tuple(int(value) for value in theme.rope_rgb),
            width=max(2, int(rope_width)),
        )
        draw.line(
            [(float(x_value - (cap_radius * 1.7)), float(cap_y)), (float(x_value + (cap_radius * 1.7)), float(cap_y))],
            fill=tuple(int(value) for value in theme.missing_outline_rgb),
            width=max(2, int(rope_width) - 1),
        )
        segment_bbox = [
            round(float(x_value - max(rope_width / 2.0, cap_radius * 1.7)), 3),
            round(float(min(start_y, end_y) - 1.0), 3),
            round(float(x_value + max(rope_width / 2.0, cap_radius * 1.7)), 3),
            round(float(max(start_y, end_y) + 1.0), 3),
        ]
        cut_segment_bboxes.append(list(segment_bbox))
        cut_segment_map[str(cut_segment.segment_id)] = list(segment_bbox)
        scene_entities.append(
            {
                "entity_id": str(cut_segment.segment_id),
                "entity_type": "cut_non_supporting_rope_segment",
                "bbox_px": list(segment_bbox),
                "meta": {
                    "slot_index": int(cut_segment.x_order),
                    "attach_side": str(cut_segment.attach_side),
                    "cut_fraction": float(cut_segment.cut_fraction),
                    "supports_moving_block": False,
                    "connects_upper_and_lower_blocks": False,
                },
            }
        )

    for slot_index in sorted(connected_slots.union(top_cut_slots)):
        x_value = float(slot_xs[int(slot_index)])
        pulley_bbox = _draw_pulley(
            draw,
            center_x=float(x_value),
            center_y=float(top_pulley_y),
            radius_px=int(pulley_radius),
            hub_radius_px=int(hub_radius),
            fill_rgb=tuple(int(value) for value in theme.pulley_fill_rgb),
            outline_rgb=tuple(int(value) for value in theme.pulley_outline_rgb),
            width_px=3,
        )
        pulley_bboxes.append(list(pulley_bbox))
        scene_entities.append(
            {
                "entity_id": f"upper_pulley_slot_{int(slot_index)}",
                "entity_type": "pulley_wheel",
                "bbox_px": list(pulley_bbox),
                "meta": {"block": "fixed_upper_block", "slot_index": int(slot_index)},
            }
        )
    for slot_index in sorted(connected_slots.union(bottom_cut_slots)):
        x_value = float(slot_xs[int(slot_index)])
        pulley_bbox = _draw_pulley(
            draw,
            center_x=float(x_value),
            center_y=float(lower_pulley_y),
            radius_px=int(pulley_radius),
            hub_radius_px=int(hub_radius),
            fill_rgb=tuple(int(value) for value in theme.pulley_fill_rgb),
            outline_rgb=tuple(int(value) for value in theme.pulley_outline_rgb),
            width_px=3,
        )
        pulley_bboxes.append(list(pulley_bbox))
        scene_entities.append(
            {
                "entity_id": f"lower_pulley_slot_{int(slot_index)}",
                "entity_type": "pulley_wheel",
                "bbox_px": list(pulley_bbox),
                "meta": {"block": "moving_lower_block", "slot_index": int(slot_index)},
            }
        )

    connector_x = float((lower_bbox[0] + lower_bbox[2]) / 2.0)
    load_top = float(lower_bbox[3] + int(render_defaults["load_top_gap_px"]))
    load_width = float(render_defaults["load_width_px"])
    load_height = float(render_defaults["load_height_px"])
    load_bbox = [
        round(float(connector_x - (0.5 * load_width)), 3),
        round(float(load_top), 3),
        round(float(connector_x + (0.5 * load_width)), 3),
        round(float(load_top + load_height), 3),
    ]
    draw.line(
        [(float(connector_x), float(lower_bbox[3])), (float(connector_x), float(load_bbox[1]))],
        fill=tuple(int(value) for value in theme.rope_rgb),
        width=max(2, int(render_defaults["connector_width_px"])),
    )
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in load_bbox),
        radius=12,
        fill=tuple(int(value) for value in theme.load_fill_rgb)
        if scene_spec.shown_load_force_value is not None
        else tuple(int(value) for value in theme.missing_fill_rgb),
        outline=tuple(int(value) for value in theme.load_outline_rgb)
        if scene_spec.shown_load_force_value is not None
        else tuple(int(value) for value in theme.missing_outline_rgb),
        width=4,
    )
    if scene_spec.shown_load_force_value is None:
        load_text = "load ? N"
        load_text_fill = tuple(int(value) for value in theme.missing_text_rgb)
    else:
        load_text = f"load {int(scene_spec.shown_load_force_value)} N"
        load_text_fill = tuple(int(value) for value in theme.load_text_rgb)
    load_text_bbox = draw_centered_text(
        draw,
        text=str(load_text),
        center=(float(connector_x), float((load_bbox[1] + load_bbox[3]) / 2.0)),
        font=label_font,
        fill=tuple(int(value) for value in load_text_fill),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(load_text_fill)),
        stroke_width=max(1, int(render_defaults["label_stroke_width_px"])),
    )
    load_label_bbox = bbox_union_many(load_bbox, load_text_bbox)

    effort_x = min(
        float(image.size[0] - 120.0),
        float(lower_bbox[2] + int(render_defaults["effort_arrow_x_gap_px"])),
    )
    arrow_start_y = float(top_bbox[3] + 24.0)
    arrow_end_y = min(
        float(lower_bbox[1] - 24.0),
        float(arrow_start_y + int(render_defaults["effort_arrow_length_px"])),
    )
    if arrow_end_y <= arrow_start_y + 30.0:
        arrow_end_y = float(arrow_start_y + 30.0)
    draw_arrow(
        draw,
        start=(float(effort_x), float(arrow_start_y)),
        end=(float(effort_x), float(arrow_end_y)),
        fill=tuple(int(value) for value in theme.effort_rgb),
        width=max(3, int(rope_width)),
        head_length_px=20.0,
        head_width_px=18.0,
    )
    effort_arrow_bbox = [
        round(float(effort_x - 12.0), 3),
        round(float(arrow_start_y), 3),
        round(float(effort_x + 12.0), 3),
        round(float(arrow_end_y), 3),
    ]
    if scene_spec.shown_effort_force_value is None:
        effort_text = "effort ? N"
        effort_missing = True
    else:
        effort_text = f"effort {int(scene_spec.shown_effort_force_value)} N"
        effort_missing = False
    effort_tag_bbox = _draw_text_tag(
        draw,
        text=str(effort_text),
        center=(float(min(image.size[0] - 96.0, effort_x + 52.0)), float((arrow_start_y + arrow_end_y) / 2.0)),
        font=small_label_font,
        fill_rgb=tuple(int(value) for value in theme.missing_fill_rgb) if effort_missing else (255, 255, 255),
        outline_rgb=tuple(int(value) for value in theme.missing_outline_rgb)
        if effort_missing
        else tuple(int(value) for value in theme.effort_rgb),
        text_rgb=tuple(int(value) for value in theme.missing_text_rgb)
        if effort_missing
        else tuple(int(value) for value in theme.load_text_rgb),
        stroke_width_px=int(render_defaults["label_stroke_width_px"]),
    )
    effort_label_bbox = bbox_union_many(effort_arrow_bbox, effort_tag_bbox)

    scene_entities.extend(
        [
            {
                "entity_id": "load_force_label",
                "entity_type": "load_force_label",
                "bbox_px": list(load_label_bbox),
                "meta": {
                    "shown_value": None if scene_spec.shown_load_force_value is None else int(scene_spec.shown_load_force_value),
                    "true_value": int(scene_spec.load_force_value),
                },
            },
            {
                "entity_id": "effort_force_label",
                "entity_type": "effort_force_label",
                "bbox_px": list(effort_label_bbox),
                "meta": {
                    "shown_value": None if scene_spec.shown_effort_force_value is None else int(scene_spec.shown_effort_force_value),
                    "true_value": int(scene_spec.effort_force_value),
                },
            },
        ]
    )

    entity_bbox_map = {
        str(entity["entity_id"]): list(entity["bbox_px"])
        for entity in scene_entities
        if entity.get("bbox_px") is not None
    }
    annotation_bboxes = [
        list(entity_bbox_map[entity_id])
        for entity_id in scene_spec.annotation_entity_ids
        if str(entity_id) in entity_bbox_map
    ]
    annotation_bbox_map: Dict[str, List[float]] = {}
    annotation_entity_id_map: Dict[str, str] = {}
    supporting_entity_ids = [
        f"support_segment_{int(support_index)}"
        for support_index in range(1, int(scene_spec.support_segment_count) + 1)
        if f"support_segment_{int(support_index)}" in entity_bbox_map
    ]
    if supporting_entity_ids:
        annotation_bbox_map["supporting_strands_region"] = bbox_union_many(
            *(entity_bbox_map[str(entity_id)] for entity_id in supporting_entity_ids),
            padding=6.0,
        )
        annotation_entity_id_map["supporting_strands_region"] = ",".join(supporting_entity_ids)
    if str(scene_spec.solve_for) == "effort_force":
        known_entity_id = "load_force_label"
        unknown_entity_id = "effort_force_label"
    else:
        known_entity_id = "effort_force_label"
        unknown_entity_id = "load_force_label"
    annotation_bbox_map["known_force_label"] = list(entity_bbox_map[str(known_entity_id)])
    annotation_bbox_map["unknown_force_label"] = list(entity_bbox_map[str(unknown_entity_id)])
    annotation_entity_id_map["known_force_label"] = str(known_entity_id)
    annotation_entity_id_map["unknown_force_label"] = str(unknown_entity_id)

    render_map = {
        "accent_color_name": str(accent_color_name),
        "technical_diagram_frame_mode": str(getattr(diagram_style, "frame_mode", "none")),
        "fixed_upper_block_bbox_px": list(top_bbox),
        "moving_lower_block_bbox_px": list(lower_bbox),
        "support_segment_count": int(scene_spec.support_segment_count),
        "disconnected_segment_count": int(scene_spec.disconnected_segment_count),
        "connected_slot_indices": [int(value) for value in scene_spec.connected_slot_indices],
        "support_segment_bboxes_px": dict(support_segment_map),
        "supporting_strands_region_bbox_px": list(annotation_bbox_map["supporting_strands_region"]),
        "cut_segment_bboxes_px": dict(cut_segment_map),
        "pulley_bboxes_px": [list(bbox) for bbox in pulley_bboxes],
        "load_force_label_bbox_px": list(load_label_bbox),
        "effort_force_label_bbox_px": list(effort_label_bbox),
        "annotation_entity_ids": list(scene_spec.annotation_entity_ids),
        "annotation_bbox_map_px": dict(annotation_bbox_map),
        "annotation_entity_id_map": dict(annotation_entity_id_map),
    }

    return RenderedPulleyScene(
        image=image,
        annotation_bboxes=[list(bbox) for bbox in annotation_bboxes],
        annotation_bbox_map=dict(annotation_bbox_map),
        annotation_entity_ids=list(scene_spec.annotation_entity_ids),
        annotation_entity_id_map=dict(annotation_entity_id_map),
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
        support_segment_bboxes=[list(bbox) for bbox in support_segment_bboxes],
        cut_segment_bboxes=[list(bbox) for bbox in cut_segment_bboxes],
        load_label_bbox=list(load_label_bbox),
        effort_label_bbox=list(effort_label_bbox),
    )


def pulley_content_bbox(
    *,
    render_defaults: Mapping[str, Any],
    scene_spec: PulleySceneSpec,
) -> List[float]:
    """Return an approximate whole-diagram bbox before layout jitter."""

    canvas_width = int(render_defaults["canvas_width"])
    top_left = float(render_defaults["top_block_x_px"])
    top_y = float(render_defaults["top_block_y_px"])
    top_width = float(render_defaults["top_block_width_px"])
    top_height = float(render_defaults["top_block_height_px"])
    top_right = float(top_left + top_width)
    top_center_x = float(top_left + (0.5 * top_width))
    if str(scene_spec.scene_variant) == "compact_block":
        lower_y = float(render_defaults["compact_lower_block_y_px"])
    elif str(scene_spec.scene_variant) == "tall_block":
        lower_y = float(render_defaults["tall_lower_block_y_px"])
    else:
        lower_y = float(render_defaults["lower_block_y_px"])
    total_slots = int(scene_spec.support_segment_count) + int(scene_spec.disconnected_segment_count)
    lower_width = max(250.0, ((int(total_slots) - 1) * float(render_defaults["support_segment_gap_px"])) + 130.0)
    lower_height = float(render_defaults["lower_block_height_px"])
    lower_left = float(top_center_x - (0.5 * lower_width))
    lower_right = float(top_center_x + (0.5 * lower_width))
    load_top = float(lower_y + lower_height + int(render_defaults["load_top_gap_px"]))
    load_width = float(render_defaults["load_width_px"])
    load_height = float(render_defaults["load_height_px"])
    load_left = float(top_center_x - (0.5 * load_width))
    load_right = float(top_center_x + (0.5 * load_width))
    effort_x = min(
        float(canvas_width - 120.0),
        float(lower_right + int(render_defaults["effort_arrow_x_gap_px"])),
    )
    effort_label_right = min(float(canvas_width - 96.0), float(effort_x + 52.0)) + 150.0
    effort_label_left = float(effort_x - 24.0)
    return [
        round(float(min(top_left, lower_left, load_left, effort_label_left) - 24.0), 3),
        round(float(top_y - 24.0), 3),
        round(float(max(top_right, lower_right, load_right, effort_label_right) + 24.0), 3),
        round(float(load_top + load_height + 24.0), 3),
    ]


def resolve_pulley_layout_placement(
    *,
    render_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    scene_spec: PulleySceneSpec,
    namespace: str = SCENE_NAMESPACE,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve whole-diagram placement before rendering and annotation projection."""

    canvas_width = int(render_defaults["canvas_width"])
    canvas_height = int(render_defaults["canvas_height"])
    content_bbox = pulley_content_bbox(render_defaults=render_defaults, scene_spec=scene_spec)
    content_left, content_top, content_right, content_bottom = [float(value) for value in content_bbox]
    jitter = resolve_layout_jitter(
        params,
        rendering_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.layout",
    )
    min_margin = int(jitter.get("min_margin_px", 20))
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
    adjusted["layout_offset_x_px"] = int(dx)
    adjusted["layout_offset_y_px"] = int(dy)
    content_width = round(float(content_right) - float(content_left), 3)
    content_height = round(float(content_bottom) - float(content_top), 3)
    final_bbox = [
        round(float(content_left) + float(dx), 3),
        round(float(content_top) + float(dy), 3),
        round(float(content_right) + float(dx), 3),
        round(float(content_bottom) + float(dy), 3),
    ]
    placement = dict(jitter)
    placement.update(
        {
            "mode": "whole_pulley_diagram_offset",
            "content_bbox_px": list(content_bbox),
            "content_size_px": [float(content_width), float(content_height)],
            "final_content_bbox_px": list(final_bbox),
            "canvas_size_px": [int(canvas_width), int(canvas_height)],
            "free_space_px": [
                round(float(canvas_width) - float(content_width), 3),
                round(float(canvas_height) - float(content_height), 3),
            ],
            "available_offset_x_px": [int(min_dx), int(max_dx)],
            "available_offset_y_px": [int(min_dy), int(max_dy)],
            "sampled_offset_px": [int(requested_dx), int(requested_dy)],
            "final_offset_px": [int(dx), int(dy)],
            "default_origin_px": [round(float(content_left), 3), round(float(content_top), 3)],
            "final_origin_px": [round(float(content_left) + float(dx), 3), round(float(content_top) + float(dy), 3)],
            "dx_px": int(dx),
            "dy_px": int(dy),
        }
    )
    return adjusted, placement


RENDER_DEFAULT_KEYS: tuple[str, ...] = (
    "canvas_width",
    "canvas_height",
    "top_block_x_px",
    "top_block_y_px",
    "top_block_width_px",
    "top_block_height_px",
    "lower_block_y_px",
    "compact_lower_block_y_px",
    "tall_lower_block_y_px",
    "lower_block_height_px",
    "support_segment_gap_px",
    "rope_width_px",
    "pulley_radius_px",
    "pulley_hub_radius_px",
    "load_width_px",
    "load_height_px",
    "load_top_gap_px",
    "connector_width_px",
    "effort_arrow_x_gap_px",
    "effort_arrow_length_px",
    "label_font_size_px",
    "small_label_font_size_px",
    "label_stroke_width_px",
    "texture_line_width_px",
    "texture_spacing_px",
    "cut_endpoint_radius_px",
)


__all__ = [
    "RENDER_DEFAULT_KEYS",
    "render_pulley_scene",
    "resolve_pulley_layout_placement",
]
