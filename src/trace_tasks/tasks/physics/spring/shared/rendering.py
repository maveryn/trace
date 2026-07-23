"""Rendering helpers for the spring physics scene."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.drawing import draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill
from trace_tasks.tasks.physics.shared.style import build_physics_spring_theme

from .annotations import expanded_spring_extension_marker_bbox
from .state import RenderedSpringScene, SpringColumnSpec, SpringSceneSpec


def _draw_card_texture(draw: ImageDraw.ImageDraw, *, bbox: Sequence[float], line_rgb: Tuple[int, int, int], spacing_px: int, width_px: int) -> None:
    """Draw light diagonal texture within one spring card."""

    left, top, right, bottom = [float(value) for value in bbox]
    span = int(bottom - top)
    offset = -span
    while offset < int(right - left) + span:
        start = (float(left + offset), float(top))
        end = (float(left + offset + span), float(bottom))
        draw.line([start, end], fill=tuple(int(value) for value in line_rgb), width=int(width_px))
        offset += max(8, int(spacing_px))


def _draw_weight_block(
    draw: ImageDraw.ImageDraw,
    *,
    center_x: float,
    top_y: float,
    width_px: int,
    height_px: int,
    label_text: str,
    fill_rgb: Tuple[int, int, int],
    outline_rgb: Tuple[int, int, int],
    text_rgb: Tuple[int, int, int],
    font,
    stroke_width: int,
) -> List[float]:
    """Draw one labeled weight block and return its bbox."""

    bbox = [
        round(float(center_x - (width_px / 2.0)), 3),
        round(float(top_y), 3),
        round(float(center_x + (width_px / 2.0)), 3),
        round(float(top_y + height_px), 3),
    ]
    draw_rounded_rect(
        draw,
        bbox,
        radius=min(16, int(height_px // 3)),
        fill=tuple(int(value) for value in fill_rgb),
        outline=tuple(int(value) for value in outline_rgb),
        width=max(2, int(stroke_width)),
    )
    draw_centered_text(
        draw,
        text=str(label_text),
        center=(float(center_x), float(top_y + (height_px / 2.0))),
        font=font,
        fill=tuple(int(value) for value in text_rgb),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(text_rgb)),
        stroke_width=max(1, int(stroke_width)),
    )
    return list(bbox)


def _draw_spring(
    draw: ImageDraw.ImageDraw,
    *,
    center_x: float,
    top_y: float,
    bottom_y: float,
    half_width_px: int,
    turn_count: int,
    line_width_px: int,
    line_rgb: Tuple[int, int, int],
    dashed: bool,
) -> List[float]:
    """Draw one vertical coil spring and return its bounding bbox."""

    if float(bottom_y) <= float(top_y) + 4.0:
        bottom_y = float(top_y) + 4.0
    points: List[Tuple[float, float]] = [(float(center_x), float(top_y))]
    total_height = float(bottom_y - top_y)
    segment_count = max(4, int(turn_count) * 2)
    for index in range(1, int(segment_count)):
        frac = float(index) / float(segment_count)
        y = float(top_y + (frac * total_height))
        x = float(center_x + (half_width_px if index % 2 else -half_width_px))
        points.append((x, y))
    points.append((float(center_x), float(bottom_y)))
    if bool(dashed):
        dash_length = 6
        for start_index in range(len(points) - 1):
            if start_index % 2:
                continue
            draw.line(
                [points[start_index], points[start_index + 1]],
                fill=tuple(int(value) for value in line_rgb),
                width=int(line_width_px),
            )
    else:
        draw.line(points, fill=tuple(int(value) for value in line_rgb), width=int(line_width_px))
    return [
        round(float(center_x - half_width_px - line_width_px), 3),
        round(float(top_y), 3),
        round(float(center_x + half_width_px + line_width_px), 3),
        round(float(bottom_y), 3),
    ]


def _draw_ruler(
    draw: ImageDraw.ImageDraw,
    *,
    x_px: float,
    top_px: float,
    unit_px: int,
    max_value: int,
    width_px: int,
    tick_long_px: int,
    tick_short_px: int,
    font,
    label_fill_rgb: Tuple[int, int, int],
    line_rgb: Tuple[int, int, int],
    stroke_width: int,
) -> List[float]:
    """Draw one vertical extension ruler and return its bbox."""

    bottom_px = float(top_px + (int(max_value) * int(unit_px)))
    draw.line(
        [(float(x_px), float(top_px)), (float(x_px), float(bottom_px))],
        fill=tuple(int(value) for value in line_rgb),
        width=max(2, int(width_px)),
    )
    for value in range(0, int(max_value) + 1):
        y = float(top_px + (int(value) * int(unit_px)))
        tick_length = int(tick_long_px if value % 2 == 0 else tick_short_px)
        draw.line(
            [(float(x_px - tick_length), y), (float(x_px), y)],
            fill=tuple(int(value) for value in line_rgb),
            width=max(2, int(width_px)),
        )
        label_center = (float(x_px - tick_length - 18), float(y))
        draw_centered_text(
            draw,
            text=str(int(value)),
            center=label_center,
            font=font,
            fill=tuple(int(value) for value in label_fill_rgb),
            stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(label_fill_rgb)),
            stroke_width=max(1, int(stroke_width)),
        )
    return [
        round(float(x_px - tick_long_px - 44), 3),
        round(float(top_px - 12), 3),
        round(float(x_px + width_px), 3),
        round(float(bottom_px + 12), 3),
    ]


def _render_column(
    draw: ImageDraw.ImageDraw,
    *,
    theme,
    spec: SpringColumnSpec,
    card_bbox: Sequence[float],
    render_defaults: Mapping[str, Any],
    scene_variant: str,
    font_family: str | None = None,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Render one spring card and return trace metadata for the column."""

    left, top, right, bottom = [float(value) for value in card_bbox]
    card_center_x = float((left + right) / 2.0)
    support_y = float(top + 34.0)
    support_width = int(render_defaults["support_width_px"])
    support_height = int(render_defaults["support_height_px"])
    support_bbox = [
        round(float(card_center_x - (support_width / 2.0)), 3),
        round(float(support_y), 3),
        round(float(card_center_x + (support_width / 2.0)), 3),
        round(float(support_y + support_height), 3),
    ]
    draw_rounded_rect(
        draw,
        support_bbox,
        radius=int(render_defaults["support_corner_radius_px"]),
        fill=tuple(int(value) for value in theme.support_fill_rgb),
        outline=tuple(int(value) for value in theme.support_outline_rgb),
        width=3,
    )
    if str(scene_variant) == "textured_spring":
        _draw_card_texture(
            draw,
            bbox=support_bbox,
            line_rgb=tuple(int(value) for value in theme.texture_rgb),
            spacing_px=int(render_defaults["texture_spacing_px"]),
            width_px=int(render_defaults["texture_line_width_px"]),
        )

    anchor_y = float(support_bbox[3] + int(render_defaults["anchor_y_gap_px"]))
    ruler_x = float(right - int(render_defaults["ruler_right_gap_px"]))
    ruler_top = float(anchor_y + int(render_defaults["ruler_top_gap_px"]))
    ruler_bbox = _draw_ruler(
        draw,
        x_px=float(ruler_x),
        top_px=float(ruler_top),
        unit_px=int(render_defaults["ruler_unit_px"]),
        max_value=int(render_defaults["ruler_value_max"]),
        width_px=int(render_defaults["ruler_width_px"]),
        tick_long_px=int(render_defaults["ruler_tick_long_px"]),
        tick_short_px=int(render_defaults["ruler_tick_short_px"]),
        font=load_font(int(render_defaults["ruler_font_size_px"]), bold=False, font_family=font_family),
        label_fill_rgb=tuple(int(value) for value in theme.ruler_text_rgb),
        line_rgb=tuple(int(value) for value in theme.ruler_rgb),
        stroke_width=int(render_defaults["label_stroke_width_px"]),
    )

    spring_x = float(card_center_x - 26.0)
    marker_width = int(render_defaults["marker_width_px"])
    marker_height = int(render_defaults["marker_height_px"])
    weight_box_width = int(render_defaults["weight_box_width_px"])
    weight_box_height = int(render_defaults["weight_box_height_px"])
    weight_font = load_font(int(render_defaults["weight_font_size_px"]), bold=True, font_family=font_family)

    shown_extension = None if spec.shown_extension_value is None else int(spec.shown_extension_value)
    true_extension = int(spec.true_extension_value)
    spring_end_y = float(ruler_top + ((shown_extension if shown_extension is not None else int(render_defaults["spring_neutral_units"])) * int(render_defaults["ruler_unit_px"])))

    hanger_bbox = [
        round(float(spring_x - (int(render_defaults["hanger_line_width_px"]) / 2.0) - 1.0), 3),
        round(float(support_bbox[3]), 3),
        round(float(spring_x + (int(render_defaults["hanger_line_width_px"]) / 2.0) + 1.0), 3),
        round(float(anchor_y), 3),
    ]
    draw.line(
        [(float(spring_x), float(support_bbox[3])), (float(spring_x), float(anchor_y))],
        fill=tuple(int(value) for value in theme.spring_rgb),
        width=max(2, int(render_defaults["hanger_line_width_px"])),
    )

    extension_marker_bbox: List[float] | None = None
    if shown_extension is not None:
        marker_y = float(ruler_top + (shown_extension * int(render_defaults["ruler_unit_px"])))
        extension_marker_bbox = [
            round(float(ruler_x - marker_width), 3),
            round(float(marker_y - (marker_height / 2.0)), 3),
            round(float(ruler_x + 2.0), 3),
            round(float(marker_y + (marker_height / 2.0)), 3),
        ]
        draw_rounded_rect(
            draw,
            extension_marker_bbox,
            radius=max(4, int(marker_height // 2)),
            fill=tuple(int(value) for value in theme.marker_fill_rgb),
            outline=tuple(int(value) for value in theme.marker_outline_rgb),
            width=2,
        )
        draw_centered_text(
            draw,
            text=str(int(shown_extension)),
            center=(
                float((extension_marker_bbox[0] + extension_marker_bbox[2]) / 2.0),
                float((extension_marker_bbox[1] + extension_marker_bbox[3]) / 2.0),
            ),
            font=load_font(max(14, int(render_defaults["ruler_font_size_px"]) - 2), bold=True, font_family=font_family),
            fill=tuple(int(value) for value in theme.ruler_text_rgb),
            stroke_fill=tuple(int(value) for value in theme.ruler_text_rgb),
            stroke_width=0,
        )
    else:
        missing_tag_width = int(render_defaults["missing_tag_width_px"])
        missing_tag_height = int(render_defaults["missing_tag_height_px"])
        marker_center_y = float(top + int(render_defaults["missing_tag_top_gap_px"]) + (missing_tag_height / 2.0))
        extension_marker_bbox = [
            round(float(ruler_x - missing_tag_width), 3),
            round(float(marker_center_y - (missing_tag_height / 2.0)), 3),
            round(float(ruler_x + 2.0), 3),
            round(float(marker_center_y + (missing_tag_height / 2.0)), 3),
        ]
        draw_rounded_rect(
            draw,
            extension_marker_bbox,
            radius=max(6, int(missing_tag_height // 3)),
            fill=tuple(int(value) for value in theme.missing_fill_rgb),
            outline=tuple(int(value) for value in theme.missing_outline_rgb),
            width=2,
        )
        draw_centered_text(
            draw,
            text="?",
            center=(
                float((extension_marker_bbox[0] + extension_marker_bbox[2]) / 2.0),
                float((extension_marker_bbox[1] + extension_marker_bbox[3]) / 2.0),
            ),
            font=load_font(max(22, int(render_defaults["weight_font_size_px"])), bold=True, font_family=font_family),
            fill=tuple(int(value) for value in theme.missing_text_rgb),
            stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.missing_text_rgb)),
            stroke_width=max(1, int(render_defaults["label_stroke_width_px"])),
        )

    spring_bbox = _draw_spring(
        draw,
        center_x=float(spring_x),
        top_y=float(anchor_y),
        bottom_y=float(spring_end_y),
        half_width_px=int(render_defaults["spring_half_width_px"]),
        turn_count=int(render_defaults["spring_turn_count"]),
        line_width_px=int(render_defaults["spring_line_width_px"]),
        line_rgb=tuple(int(value) for value in theme.spring_rgb),
        dashed=bool(spec.missing_extension),
    )

    weight_bbox: List[float] | None = None
    if bool(spec.detached_weight):
        detached_center_x = float(card_center_x + 22.0)
        detached_top_y = float(bottom - weight_box_height - 42.0)
        draw.line(
            [(float(detached_center_x), float(detached_top_y - 24.0)), (float(detached_center_x), float(detached_top_y - 6.0))],
            fill=tuple(int(value) for value in theme.spring_rgb),
            width=3,
        )
        weight_bbox = _draw_weight_block(
            draw,
            center_x=float(detached_center_x),
            top_y=float(detached_top_y),
            width_px=int(weight_box_width),
            height_px=int(weight_box_height),
            label_text=str(spec.shown_weight_value),
            fill_rgb=tuple(int(value) for value in theme.weight_fill_rgb),
            outline_rgb=tuple(int(value) for value in theme.weight_outline_rgb),
            text_rgb=tuple(int(value) for value in theme.weight_text_rgb),
            font=weight_font,
            stroke_width=int(render_defaults["label_stroke_width_px"]),
        )
    else:
        weight_top_y = float(spring_end_y)
        draw.line(
            [(float(spring_x), float(spring_end_y)), (float(spring_x), float(weight_top_y))],
            fill=tuple(int(value) for value in theme.spring_rgb),
            width=3,
        )
        if bool(spec.missing_weight):
            weight_bbox = _draw_weight_block(
                draw,
                center_x=float(spring_x),
                top_y=float(weight_top_y),
                width_px=int(weight_box_width),
                height_px=int(weight_box_height),
                label_text="?",
                fill_rgb=tuple(int(value) for value in theme.missing_fill_rgb),
                outline_rgb=tuple(int(value) for value in theme.missing_outline_rgb),
                text_rgb=tuple(int(value) for value in theme.missing_text_rgb),
                font=weight_font,
                stroke_width=int(render_defaults["label_stroke_width_px"]),
            )
        else:
            weight_bbox = _draw_weight_block(
                draw,
                center_x=float(spring_x),
                top_y=float(weight_top_y),
                width_px=int(weight_box_width),
                height_px=int(weight_box_height),
                label_text=str(spec.shown_weight_value),
                fill_rgb=tuple(int(value) for value in theme.weight_fill_rgb),
                outline_rgb=tuple(int(value) for value in theme.weight_outline_rgb),
                text_rgb=tuple(int(value) for value in theme.weight_text_rgb),
                font=weight_font,
                stroke_width=int(render_defaults["label_stroke_width_px"]),
            )

    entities = [
        {
            "entity_id": f"{spec.column_id}_card",
            "entity_type": "spring_card",
            "bbox_px": [round(float(value), 3) for value in card_bbox],
            "scene_role": spec.column_id,
        },
        {
            "entity_id": f"{spec.column_id}_support_bar",
            "entity_type": "support_bar",
            "bbox_px": list(support_bbox),
            "scene_role": spec.column_id,
        },
        {
            "entity_id": f"{spec.column_id}_spring_hanger",
            "entity_type": "spring_hanger",
            "bbox_px": list(hanger_bbox),
            "scene_role": spec.column_id,
        },
        {
            "entity_id": f"{spec.column_id}_spring_body",
            "entity_type": "spring_body",
            "bbox_px": list(spring_bbox),
            "scene_role": spec.column_id,
            "missing_extension": bool(spec.missing_extension),
        },
        {
            "entity_id": f"{spec.column_id}_ruler",
            "entity_type": "ruler",
            "bbox_px": list(ruler_bbox),
            "scene_role": spec.column_id,
        },
        {
            "entity_id": f"{spec.column_id}_extension_marker",
            "entity_type": "extension_marker" if not spec.missing_extension else "missing_extension_marker",
            "bbox_px": list(extension_marker_bbox),
            "scene_role": spec.column_id,
            "value": None if spec.shown_extension_value is None else int(spec.shown_extension_value),
            "true_value": int(true_extension),
        },
    ]
    if weight_bbox is not None:
        entities.append(
            {
                "entity_id": f"{spec.column_id}_weight_block",
                "entity_type": "missing_weight_block" if spec.missing_weight else "weight_block",
                "bbox_px": list(weight_bbox),
                "scene_role": spec.column_id,
                "value": None if spec.shown_weight_value is None else int(spec.shown_weight_value),
                "true_value": int(spec.true_weight_value),
                "detached": bool(spec.detached_weight),
            }
        )
    column_trace = {
        "column_id": str(spec.column_id),
        "shown_weight_value": None if spec.shown_weight_value is None else int(spec.shown_weight_value),
        "true_weight_value": int(spec.true_weight_value),
        "shown_extension_value": None if spec.shown_extension_value is None else int(spec.shown_extension_value),
        "true_extension_value": int(spec.true_extension_value),
        "missing_weight": bool(spec.missing_weight),
        "missing_extension": bool(spec.missing_extension),
        "detached_weight": bool(spec.detached_weight),
        "card_bbox_px": [round(float(value), 3) for value in card_bbox],
        "spring_bbox_px": list(spring_bbox),
        "hanger_bbox_px": list(hanger_bbox),
        "weight_bbox_px": None if weight_bbox is None else list(weight_bbox),
        "extension_marker_bbox_px": list(extension_marker_bbox),
    }
    return column_trace, entities


def render_spring_scene(
    *,
    background: Image.Image,
    render_defaults: Mapping[str, Any],
    accent_color_name: str,
    scene_spec: SpringSceneSpec,
    diagram_style: Any | None = None,
    font_family: str | None = None,
) -> RenderedSpringScene:
    """Render one spring-extension diagram and return trace metadata."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    theme = build_physics_spring_theme(str(accent_color_name), diagram_style=diagram_style)

    card_left = float(render_defaults["card_left_px"])
    card_top = float(render_defaults["card_top_px"])
    card_width = float(render_defaults["card_width_px"])
    card_height = float(render_defaults["card_height_px"])
    card_gap = float(render_defaults["card_gap_px"])
    stagger_offset = float(render_defaults["stagger_offset_y_px"])
    card_outline_width = int(render_defaults["card_outline_width_px"])
    card_radius = int(render_defaults["card_corner_radius_px"])

    left_card_bbox = [card_left, card_top, card_left + card_width, card_top + card_height]
    right_top = card_top + (stagger_offset if str(scene_spec.scene_variant) == "staggered_springs" else 0.0)
    right_left = card_left + card_width + card_gap
    right_card_bbox = [right_left, right_top, right_left + card_width, right_top + card_height]

    for card_bbox in (left_card_bbox, right_card_bbox):
        draw_rounded_rect(
            draw,
            [round(float(value), 3) for value in card_bbox],
            radius=int(card_radius),
            fill=tuple(int(value) for value in theme.card_fill_rgb),
            outline=tuple(int(value) for value in theme.card_outline_rgb),
            width=max(2, int(card_outline_width)),
        )
        if str(scene_spec.scene_variant) == "textured_spring":
            _draw_card_texture(
                draw,
                bbox=card_bbox,
                line_rgb=tuple(int(value) for value in theme.texture_rgb),
                spacing_px=int(render_defaults["texture_spacing_px"]),
                width_px=int(render_defaults["texture_line_width_px"]),
            )

    left_trace, left_entities = _render_column(
        draw,
        theme=theme,
        spec=scene_spec.left,
        card_bbox=left_card_bbox,
        render_defaults=render_defaults,
        scene_variant=str(scene_spec.scene_variant),
        font_family=font_family,
    )
    right_trace, right_entities = _render_column(
        draw,
        theme=theme,
        spec=scene_spec.right,
        card_bbox=right_card_bbox,
        render_defaults=render_defaults,
        scene_variant=str(scene_spec.scene_variant),
        font_family=font_family,
    )

    scene_entities = left_entities + right_entities
    entity_bbox_map = {
        str(entity["entity_id"]): list(entity["bbox_px"])
        for entity in scene_entities
        if entity.get("bbox_px") is not None
    }
    annotation_bboxes = []
    for entity_id in scene_spec.annotation_entity_ids:
        if entity_id not in entity_bbox_map:
            continue
        bbox = list(entity_bbox_map[entity_id])
        if str(entity_id).endswith("_extension_marker"):
            bbox = expanded_spring_extension_marker_bbox(
                bbox,
                canvas_width=int(image.size[0]),
                canvas_height=int(image.size[1]),
            )
        annotation_bboxes.append(list(bbox))
    shown_measurement_count = sum(
        1
        for spec in (scene_spec.left, scene_spec.right)
        if spec.shown_weight_value is not None and spec.shown_extension_value is not None
    )
    render_map = {
        "accent_color_name": str(accent_color_name),
        "technical_diagram_frame_mode": str(getattr(diagram_style, "frame_mode", "none")),
        "left_card_bbox_px": [round(float(value), 3) for value in left_card_bbox],
        "right_card_bbox_px": [round(float(value), 3) for value in right_card_bbox],
        "annotation_entity_ids": list(scene_spec.annotation_entity_ids),
        "annotation_bboxes_px": [list(bbox) for bbox in annotation_bboxes],
        "entity_bbox_map_px": {str(key): list(value) for key, value in entity_bbox_map.items()},
        "columns": {
            "left": dict(left_trace),
            "right": dict(right_trace),
        },
    }
    return RenderedSpringScene(
        image=image,
        annotation_bboxes=[list(bbox) for bbox in annotation_bboxes],
        annotation_entity_ids=list(scene_spec.annotation_entity_ids),
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
        shown_measurement_count=int(shown_measurement_count),
    )

__all__ = ["render_spring_scene"]
