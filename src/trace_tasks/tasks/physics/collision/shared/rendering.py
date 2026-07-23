"""Rendering helpers for physics collision scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.label_tags import draw_text_tag
from trace_tasks.tasks.physics.shared.style import build_physics_collision_theme
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.bbox_projection import bbox_union_many
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.drawing import draw_arrow, draw_centered_text, draw_rounded_rect
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.render_variation import resolve_layout_jitter, resolve_render_int
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .mechanics import angle_degrees, direction_label, direction_unit
from .sampling import sample_sticky_spec_with_retries
from .state import (
    ANNOTATION_ENTITY_KEY_BY_ID,
    CollisionAftermathRenderedScene,
    CollisionAftermathSpec,
    DIRECTION_ANGLE_DEGREES,
    OPTION_LETTERS,
    PreparedRender,
    SCENE_ID,
    StickyRenderedScene,
    StickySceneSpec,
)

POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def _bbox(values: Sequence[float]) -> List[float]:
    return [round(float(value), 3) for value in values]


def _point(values: Sequence[float]) -> List[float]:
    return [round(float(values[0]), 3), round(float(values[1]), 3)]


def _segment(start: Sequence[float], end: Sequence[float]) -> List[List[float]]:
    return [_point(start), _point(end)]


def _clip_bbox(bbox: Sequence[float], *, width: int, height: int) -> List[float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return _bbox(
        (
            max(0.0, min(float(width), min(x0, x1))),
            max(0.0, min(float(height), min(y0, y1))),
            max(0.0, min(float(width), max(x0, x1))),
            max(0.0, min(float(height), max(y0, y1))),
        )
    )


def direction_unit(direction: str) -> Tuple[float, float]:
    angle = math.radians(float(DIRECTION_ANGLE_DEGREES[str(direction)]))
    return (float(math.cos(angle)), float(-math.sin(angle)))


def _draw_aftermath_puck(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    radius_px: float,
    label: str,
    fill_rgb: Tuple[int, int, int],
    outline_rgb: Tuple[int, int, int],
    text_rgb: Tuple[int, int, int],
    font: Any,
) -> List[float]:
    cx, cy = float(center[0]), float(center[1])
    radius = float(radius_px)
    bbox = _bbox((cx - radius, cy - radius, cx + radius, cy + radius))
    draw.ellipse(tuple(float(value) for value in bbox), fill=fill_rgb, outline=outline_rgb, width=4)
    draw.ellipse((cx - radius * 0.42, cy - radius * 0.42, cx + radius * 0.42, cy + radius * 0.42), outline=outline_rgb, width=2)
    draw_centered_text(
        draw,
        text=str(label),
        center=(cx, cy),
        font=font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=1,
    )
    return bbox


def render_aftermath_scene(
    *,
    background: Image.Image,
    render_defaults: Mapping[str, int],
    accent_color_name: str,
    spec: CollisionAftermathSpec,
    font_family: str,
    diagram_style: Any | None = None,
) -> CollisionAftermathRenderedScene:
    """Draw the aftermath table while preserving keyed causal annotation boxes."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    canvas_width, canvas_height = image.size
    theme = build_physics_collision_theme(str(accent_color_name), diagram_style=diagram_style)
    label_font = load_font(int(render_defaults["label_font_size_px"]), bold=True, font_family=font_family)
    puck_font = load_font(int(render_defaults["puck_font_size_px"]), bold=True, font_family=font_family)

    table_bbox = [
        float(render_defaults["table_left_px"]),
        float(render_defaults["table_top_px"]),
        float(render_defaults["table_left_px"]) + float(render_defaults["table_width_px"]),
        float(render_defaults["table_top_px"]) + float(render_defaults["table_height_px"]),
    ]
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in table_bbox),
        radius=int(render_defaults["table_corner_radius_px"]),
        fill=tuple(int(value) for value in theme.table_fill_rgb),
        outline=tuple(int(value) for value in theme.table_outline_rgb),
        width=4,
    )
    if str(spec.scene_variant) == "aftermath_gridded_table":
        _draw_grid(
            draw,
            table_bbox=table_bbox,
            spacing_px=int(render_defaults["grid_spacing_px"]),
            fill_rgb=tuple(int(value) for value in theme.grid_rgb),
        )

    impact_center = (
        float(render_defaults["impact_center_x_px"]),
        float(render_defaults["compact_impact_center_y_px"] if str(spec.scene_variant) == "aftermath_compact_table" else render_defaults["impact_center_y_px"]),
    )
    final_unit = direction_unit(str(spec.final_motion_direction))
    target_distance = float(render_defaults["target_distance_px"])
    if str(spec.scene_variant) == "aftermath_compact_table":
        target_distance -= 18.0
    target_center = (
        float(impact_center[0] + final_unit[0] * target_distance),
        float(impact_center[1] + final_unit[1] * target_distance),
    )

    option_bboxes: Dict[str, List[float]] = {}
    option_arrow_bboxes: Dict[str, List[float]] = {}
    scene_entities: List[Dict[str, Any]] = [
        {
            "entity_id": "collision_table",
            "entity_type": "collision_table",
            "bbox_px": [round(float(value), 3) for value in table_bbox],
            "meta": {"scene_variant": str(spec.scene_variant)},
        }
    ]

    candidate_stroke = tuple(int(value) for value in theme.option_arrow_rgb)
    for letter in OPTION_LETTERS:
        direction = str(spec.option_directions[str(letter)])
        unit = direction_unit(direction)
        inner_gap = float(render_defaults["incoming_arrow_inner_gap_px"])
        arrow_length = float(render_defaults["incoming_arrow_length_px"])
        arrow_start = (
            float(impact_center[0] - unit[0] * (inner_gap + arrow_length)),
            float(impact_center[1] - unit[1] * (inner_gap + arrow_length)),
        )
        arrow_end = (
            float(impact_center[0] - unit[0] * inner_gap),
            float(impact_center[1] - unit[1] * inner_gap),
        )
        draw_arrow(
            draw,
            start=arrow_start,
            end=arrow_end,
            fill=candidate_stroke,
            width=int(render_defaults["candidate_arrow_width_px"]),
            head_length_px=float(render_defaults["candidate_arrow_head_length_px"]),
            head_width_px=float(render_defaults["candidate_arrow_head_width_px"]),
        )
        arrow_bbox = _arrow_bbox(
            arrow_start,
            arrow_end,
            padding_px=float(render_defaults["candidate_arrow_head_width_px"]),
        )
        label_center = (
            float(arrow_start[0] - unit[0] * float(render_defaults["candidate_label_offset_px"])),
            float(arrow_start[1] - unit[1] * float(render_defaults["candidate_label_offset_px"])),
        )
        letter_bbox = draw_text_tag(
            draw,
            text=str(letter),
            center=label_center,
            font=label_font,
            fill_rgb=tuple(int(value) for value in theme.label_fill_rgb),
            outline_rgb=tuple(int(value) for value in theme.option_outline_rgb),
            text_rgb=tuple(int(value) for value in theme.label_text_rgb),
            stroke_width_px=int(render_defaults["label_stroke_width_px"]),
        )
        option_bbox = bbox_union_many(arrow_bbox, letter_bbox)
        option_bboxes[str(letter)] = _clip_bbox(option_bbox, width=canvas_width, height=canvas_height)
        option_arrow_bboxes[str(letter)] = _clip_bbox(arrow_bbox, width=canvas_width, height=canvas_height)
        scene_entities.append(
            {
                "entity_id": f"option_{str(letter)}",
                "entity_type": "candidate_incoming_path",
                "bbox_px": list(option_bboxes[str(letter)]),
                "meta": {
                    "option_letter": str(letter),
                    "incoming_direction": str(direction),
                    "angle_degrees": float(DIRECTION_ANGLE_DEGREES[str(direction)]),
                    "is_correct": str(letter) == str(spec.correct_option_letter),
                },
            }
        )

    impact_radius = float(render_defaults["impact_radius_px"])
    impact_bbox = _bbox(
        (
            impact_center[0] - impact_radius,
            impact_center[1] - impact_radius,
            impact_center[0] + impact_radius,
            impact_center[1] + impact_radius,
        )
    )
    impact_rgb = tuple(int(value) for value in theme.collision_outline_rgb)
    draw.ellipse(tuple(float(value) for value in impact_bbox), fill=tuple(int(value) for value in theme.table_fill_rgb), outline=impact_rgb, width=4)
    draw.line((impact_center[0] - 9.0, impact_center[1], impact_center[0] + 9.0, impact_center[1]), fill=impact_rgb, width=3)
    draw.line((impact_center[0], impact_center[1] - 9.0, impact_center[0], impact_center[1] + 9.0), fill=impact_rgb, width=3)

    trail_start = (
        float(impact_center[0] + final_unit[0] * float(render_defaults["trail_start_gap_px"])),
        float(impact_center[1] + final_unit[1] * float(render_defaults["trail_start_gap_px"])),
    )
    trail_end = (
        float(target_center[0] - final_unit[0] * (float(render_defaults["puck_radius_px"]) + 5.0)),
        float(target_center[1] - final_unit[1] * (float(render_defaults["puck_radius_px"]) + 5.0)),
    )
    draw.line(
        (trail_start[0], trail_start[1], trail_end[0], trail_end[1]),
        fill=tuple(int(value) for value in theme.grid_rgb),
        width=max(3, int(render_defaults["motion_arrow_width_px"]) - 3),
    )
    draw_arrow(
        draw,
        start=trail_start,
        end=trail_end,
        fill=tuple(int(value) for value in theme.motion_arrow_rgb),
        width=int(render_defaults["motion_arrow_width_px"]),
        head_length_px=float(render_defaults["arrow_head_length_px"]),
        head_width_px=float(render_defaults["arrow_head_width_px"]),
    )
    trail_bbox = _arrow_bbox(trail_start, trail_end, padding_px=float(render_defaults["arrow_head_width_px"]))
    target_bbox = _draw_aftermath_puck(
        draw,
        center=target_center,
        radius_px=float(render_defaults["puck_radius_px"]),
        label="after",
        fill_rgb=tuple(int(value) for value in theme.collision_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.collision_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.puck_text_rgb),
        font=puck_font,
    )
    aftermath_bbox = bbox_union_many(target_bbox, trail_bbox)

    annotation_bbox_map = {
        "impact_point": _clip_bbox(impact_bbox, width=canvas_width, height=canvas_height),
        "target_after_motion": _clip_bbox(aftermath_bbox, width=canvas_width, height=canvas_height),
    }
    scene_entities.extend(
        [
            {
                "entity_id": "impact_point",
                "entity_type": "impact_marker",
                "bbox_px": list(annotation_bbox_map["impact_point"]),
                "point_px": [round(float(impact_center[0]), 3), round(float(impact_center[1]), 3)],
                "meta": {"role": "initial_target_location"},
            },
            {
                "entity_id": "target_after_motion",
                "entity_type": "target_puck_after_impact",
                "bbox_px": list(annotation_bbox_map["target_after_motion"]),
                "point_px": [round(float(target_center[0]), 3), round(float(target_center[1]), 3)],
                "meta": {
                    "final_motion_direction": str(spec.final_motion_direction),
                    "angle_degrees": float(DIRECTION_ANGLE_DEGREES[str(spec.final_motion_direction)]),
                },
            },
        ]
    )
    render_map = {
        "accent_color_name": str(accent_color_name),
        "technical_diagram_frame_mode": str(getattr(diagram_style, "frame_mode", "none")),
        "table_bbox_px": [round(float(value), 3) for value in table_bbox],
        "impact_center_px": [round(float(impact_center[0]), 3), round(float(impact_center[1]), 3)],
        "target_center_px": [round(float(target_center[0]), 3), round(float(target_center[1]), 3)],
        "target_bbox_px": list(_clip_bbox(target_bbox, width=canvas_width, height=canvas_height)),
        "trail_bbox_px": list(_clip_bbox(trail_bbox, width=canvas_width, height=canvas_height)),
        "impact_bbox_px": list(annotation_bbox_map["impact_point"]),
        "aftermath_bbox_px": list(annotation_bbox_map["target_after_motion"]),
        "final_motion_direction": str(spec.final_motion_direction),
        "final_motion_angle_degrees": float(DIRECTION_ANGLE_DEGREES[str(spec.final_motion_direction)]),
        "correct_option_letter": str(spec.correct_option_letter),
        "option_directions": dict(spec.option_directions),
        "option_angles_degrees": dict(spec.option_angles_degrees),
        "option_bboxes_px": {str(letter): list(bbox) for letter, bbox in option_bboxes.items()},
        "option_arrow_bboxes_px": {str(letter): list(bbox) for letter, bbox in option_arrow_bboxes.items()},
        "annotation_keyed_bboxes_px": {str(key): list(value) for key, value in annotation_bbox_map.items()},
    }
    return CollisionAftermathRenderedScene(
        image=image,
        annotation_bbox_map={str(key): list(value) for key, value in annotation_bbox_map.items()},
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
    )


def resolve_aftermath_layout_placement(
    *,
    render_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
    canvas_width: int,
    canvas_height: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply whole-diagram jitter without moving content outside the canvas."""

    content_bbox = [
        float(render_defaults["table_left_px"]),
        float(render_defaults["table_top_px"]),
        float(render_defaults["table_left_px"]) + float(render_defaults["table_width_px"]),
        float(render_defaults["table_top_px"]) + float(render_defaults["table_height_px"]),
    ]
    base_bbox = [round(float(value), 3) for value in content_bbox]
    jitter = resolve_layout_jitter(
        params,
        rendering_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.layout",
    )
    min_margin = int(jitter.get("min_margin_px", 24))
    requested_dx = int(jitter.get("requested_dx_px", 0))
    requested_dy = int(jitter.get("requested_dy_px", 0))
    min_dx = int(math.ceil(float(min_margin) - float(content_bbox[0])))
    max_dx = int(math.floor(float(canvas_width) - float(min_margin) - float(content_bbox[2])))
    min_dy = int(math.ceil(float(min_margin) - float(content_bbox[1])))
    max_dy = int(math.floor(float(canvas_height) - float(min_margin) - float(content_bbox[3])))
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
    for key in ("table_left_px", "impact_center_x_px"):
        adjusted[key] = int(round(float(adjusted[key]) + float(dx)))
    for key in ("table_top_px", "impact_center_y_px", "compact_impact_center_y_px"):
        adjusted[key] = int(round(float(adjusted[key]) + float(dy)))
    final_bbox = [
        round(float(content_bbox[0]) + float(dx), 3),
        round(float(content_bbox[1]) + float(dy), 3),
        round(float(content_bbox[2]) + float(dx), 3),
        round(float(content_bbox[3]) + float(dy), 3),
    ]
    placement = dict(jitter)
    placement.update(
        {
            "mode": "whole_collision_aftermath_offset",
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



def _arrow_bbox(start: Tuple[float, float], end: Tuple[float, float], *, padding_px: float) -> List[float]:
    """Return a conservative bbox for one arrow."""

    return [
        round(float(min(float(start[0]), float(end[0])) - float(padding_px)), 3),
        round(float(min(float(start[1]), float(end[1])) - float(padding_px)), 3),
        round(float(max(float(start[0]), float(end[0])) + float(padding_px)), 3),
        round(float(max(float(start[1]), float(end[1])) + float(padding_px)), 3),
    ]


def _screen_unit_from_physics_angle(angle_degrees: float) -> Tuple[float, float]:
    """Map a physics-plane angle to a screen-space unit vector."""

    radians = math.radians(float(angle_degrees))
    return (float(math.cos(radians)), float(-math.sin(radians)))


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
    """Draw one rounded text tag and return its outer bbox."""

    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=max(0, int(stroke_width_px)))
    text_width = float(text_bbox[2] - text_bbox[0])
    text_height = float(text_bbox[3] - text_bbox[1])
    pad_x = 12.0
    pad_y = 7.0
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
        radius=9,
        fill=tuple(int(value) for value in fill_rgb),
        outline=tuple(int(value) for value in outline_rgb),
        width=max(1, int(stroke_width_px)),
    )
    text_draw_bbox = draw_centered_text(
        draw,
        text=str(text),
        center=(float(center_x), float(center_y)),
        font=font,
        fill=tuple(int(value) for value in text_rgb),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(text_rgb)),
        stroke_width=1,
    )
    return bbox_union_many(tag_bbox, text_draw_bbox)


def _draw_grid(
    draw: ImageDraw.ImageDraw,
    *,
    table_bbox: Sequence[float],
    spacing_px: int,
    fill_rgb: Tuple[int, int, int],
) -> None:
    """Draw a light deterministic grid inside the collision table."""

    left, top, right, bottom = [float(value) for value in table_bbox]
    spacing = max(16, int(spacing_px))
    x = float(left + spacing)
    while x < float(right):
        draw.line([(float(x), float(top)), (float(x), float(bottom))], fill=tuple(int(value) for value in fill_rgb), width=1)
        x += float(spacing)
    y = float(top + spacing)
    while y < float(bottom):
        draw.line([(float(left), float(y)), (float(right), float(y))], fill=tuple(int(value) for value in fill_rgb), width=1)
        y += float(spacing)


def _puck_bbox(center: Tuple[float, float], *, radius_px: float) -> List[float]:
    """Return the bbox for one circular puck."""

    cx, cy = float(center[0]), float(center[1])
    radius = float(radius_px)
    return [
        round(float(cx - radius), 3),
        round(float(cy - radius), 3),
        round(float(cx + radius), 3),
        round(float(cy + radius), 3),
    ]


def _draw_puck(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    radius_px: int,
    label: str,
    fill_rgb: Tuple[int, int, int],
    outline_rgb: Tuple[int, int, int],
    text_rgb: Tuple[int, int, int],
    font,
) -> Tuple[List[float], List[float]]:
    """Draw one labeled puck and return puck/text bboxes."""

    bbox = _puck_bbox(center, radius_px=float(radius_px))
    draw.ellipse(
        tuple(float(value) for value in bbox),
        fill=tuple(int(value) for value in fill_rgb),
        outline=tuple(int(value) for value in outline_rgb),
        width=4,
    )
    text_bbox = draw_centered_text(
        draw,
        text=str(label),
        center=(float(center[0]), float(center[1])),
        font=font,
        fill=tuple(int(value) for value in text_rgb),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(text_rgb)),
        stroke_width=1,
    )
    return list(bbox), list(text_bbox)


def render_sticky_scene(
    *,
    background: Image.Image,
    render_defaults: Mapping[str, Any],
    accent_color_name: str,
    scene_spec: StickySceneSpec,
    font_family: str,
    diagram_style: Any | None = None,
) -> StickyRenderedScene:
    """Render one sticky-collision diagram and return trace metadata."""

    image = background.copy()
    draw = ImageDraw.Draw(image)
    theme = build_physics_collision_theme(str(accent_color_name), diagram_style=diagram_style)
    resolved_font_family = str(font_family)
    label_font = load_font(int(render_defaults["label_font_size_px"]), bold=True, font_family=resolved_font_family)
    small_font = load_font(max(14, int(render_defaults["label_font_size_px"]) - 3), bold=False, font_family=resolved_font_family)
    puck_font = load_font(int(render_defaults["puck_font_size_px"]), bold=True, font_family=resolved_font_family)
    option_font = load_font(int(render_defaults["option_font_size_px"]), bold=True, font_family=resolved_font_family)

    table_bbox = [
        float(render_defaults["table_left_px"]),
        float(render_defaults["table_top_px"]),
        float(render_defaults["table_left_px"]) + float(render_defaults["table_width_px"]),
        float(render_defaults["table_top_px"]) + float(render_defaults["table_height_px"]),
    ]
    draw_rounded_rect(
        draw,
        tuple(float(value) for value in table_bbox),
        radius=int(render_defaults["table_corner_radius_px"]),
        fill=tuple(int(value) for value in theme.table_fill_rgb),
        outline=tuple(int(value) for value in theme.table_outline_rgb),
        width=4,
    )
    if str(scene_spec.scene_variant) == "gridded_table":
        _draw_grid(
            draw,
            table_bbox=table_bbox,
            spacing_px=int(render_defaults["grid_spacing_px"]),
            fill_rgb=tuple(int(value) for value in theme.grid_rgb),
        )

    if str(scene_spec.scene_variant) == "compact_table":
        center_x = float(render_defaults["compact_collision_center_x_px"])
        center_y = float(render_defaults["compact_collision_center_y_px"])
        distance_delta = float(render_defaults["compact_start_distance_delta_px"])
    else:
        center_x = float(render_defaults["collision_center_x_px"])
        center_y = float(render_defaults["collision_center_y_px"])
        distance_delta = 0.0

    puck_radius = int(render_defaults["puck_radius_px"])
    stuck_radius = int(render_defaults["stuck_radius_px"])
    horizontal_distance = float(render_defaults["horizontal_start_distance_px"]) + float(distance_delta)
    vertical_distance = float(render_defaults["vertical_start_distance_px"]) + float(distance_delta)
    scenario = scene_spec.scenario
    horizontal_screen_unit = (float(scenario.horizontal_sign), 0.0)
    vertical_screen_unit = (0.0, float(-scenario.vertical_sign))
    horizontal_center = (
        float(center_x - (horizontal_screen_unit[0] * horizontal_distance)),
        float(center_y - (horizontal_screen_unit[1] * horizontal_distance)),
    )
    vertical_center = (
        float(center_x - (vertical_screen_unit[0] * vertical_distance)),
        float(center_y - (vertical_screen_unit[1] * vertical_distance)),
    )
    collision_center = (float(center_x), float(center_y))

    # Draw approach tracks first so pucks and arrows remain visually dominant.
    draw.line(
        [horizontal_center, collision_center],
        fill=tuple(int(value) for value in theme.grid_rgb),
        width=3,
    )
    draw.line(
        [vertical_center, collision_center],
        fill=tuple(int(value) for value in theme.grid_rgb),
        width=3,
    )

    horizontal_arrow_start = (
        float(horizontal_center[0] + (horizontal_screen_unit[0] * (puck_radius + 12))),
        float(horizontal_center[1] + (horizontal_screen_unit[1] * (puck_radius + 12))),
    )
    horizontal_arrow_end = (
        float(center_x - (horizontal_screen_unit[0] * (stuck_radius + 20))),
        float(center_y - (horizontal_screen_unit[1] * (stuck_radius + 20))),
    )
    vertical_arrow_start = (
        float(vertical_center[0] + (vertical_screen_unit[0] * (puck_radius + 12))),
        float(vertical_center[1] + (vertical_screen_unit[1] * (puck_radius + 12))),
    )
    vertical_arrow_end = (
        float(center_x - (vertical_screen_unit[0] * (stuck_radius + 20))),
        float(center_y - (vertical_screen_unit[1] * (stuck_radius + 20))),
    )
    draw_arrow(
        draw,
        start=horizontal_arrow_start,
        end=horizontal_arrow_end,
        fill=tuple(int(value) for value in theme.motion_arrow_rgb),
        width=int(render_defaults["motion_arrow_width_px"]),
        head_length_px=float(render_defaults["arrow_head_length_px"]),
        head_width_px=float(render_defaults["arrow_head_width_px"]),
    )
    draw_arrow(
        draw,
        start=vertical_arrow_start,
        end=vertical_arrow_end,
        fill=tuple(int(value) for value in theme.motion_arrow_rgb),
        width=int(render_defaults["motion_arrow_width_px"]),
        head_length_px=float(render_defaults["arrow_head_length_px"]),
        head_width_px=float(render_defaults["arrow_head_width_px"]),
    )
    horizontal_arrow_bbox = _arrow_bbox(
        horizontal_arrow_start,
        horizontal_arrow_end,
        padding_px=float(render_defaults["arrow_head_width_px"]),
    )
    vertical_arrow_bbox = _arrow_bbox(
        vertical_arrow_start,
        vertical_arrow_end,
        padding_px=float(render_defaults["arrow_head_width_px"]),
    )
    horizontal_arrow_segment = _segment(horizontal_arrow_start, horizontal_arrow_end)
    vertical_arrow_segment = _segment(vertical_arrow_start, vertical_arrow_end)

    horizontal_puck_bbox, _ = _draw_puck(
        draw,
        center=horizontal_center,
        radius_px=puck_radius,
        label="A",
        fill_rgb=tuple(int(value) for value in theme.puck_a_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.puck_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.puck_text_rgb),
        font=puck_font,
    )
    vertical_puck_bbox, _ = _draw_puck(
        draw,
        center=vertical_center,
        radius_px=puck_radius,
        label="B",
        fill_rgb=tuple(int(value) for value in theme.puck_b_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.puck_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.puck_text_rgb),
        font=puck_font,
    )
    stuck_bbox, _ = _draw_puck(
        draw,
        center=collision_center,
        radius_px=stuck_radius,
        label="A+B",
        fill_rgb=tuple(int(value) for value in theme.collision_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.collision_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.puck_text_rgb),
        font=load_font(max(18, int(render_defaults["puck_font_size_px"]) - 8), bold=True, font_family=resolved_font_family),
    )

    final_angle = angle_degrees(float(scenario.final_vx), float(scenario.final_vy))

    # Quantity tags near the pucks keep the arithmetic grounded in the image.
    horizontal_label_x = float(horizontal_center[0])
    horizontal_label_y = float(horizontal_center[1] - 78.0)
    if horizontal_label_y < float(table_bbox[1] + 34.0):
        horizontal_label_y = float(horizontal_center[1] + 78.0)
    horizontal_mass_bbox = _draw_text_tag(
        draw,
        text=f"mA={int(scenario.horizontal_mass)} kg",
        center=(horizontal_label_x, horizontal_label_y),
        font=label_font,
        fill_rgb=tuple(int(value) for value in theme.label_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.label_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.label_text_rgb),
        stroke_width_px=int(render_defaults["label_stroke_width_px"]),
    )
    horizontal_speed_bbox = _draw_text_tag(
        draw,
        text=f"vA={int(scenario.horizontal_speed)} m/s",
        center=(float((horizontal_arrow_start[0] + horizontal_arrow_end[0]) / 2.0), float(center_y + 44.0)),
        font=label_font,
        fill_rgb=tuple(int(value) for value in theme.label_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.label_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.label_text_rgb),
        stroke_width_px=int(render_defaults["label_stroke_width_px"]),
    )
    vertical_label_x = float(vertical_center[0] + 96.0)
    if vertical_label_x > float(table_bbox[2] - 95.0):
        vertical_label_x = float(vertical_center[0] - 96.0)
    if float(vertical_center[1]) > float(center_y):
        vertical_mass_y = float(vertical_center[1] - 62.0)
        vertical_speed_y = float(vertical_center[1] - 8.0)
    else:
        vertical_mass_y = float(vertical_center[1] + 8.0)
        vertical_speed_y = float(vertical_center[1] + 62.0)
    vertical_mass_bbox = _draw_text_tag(
        draw,
        text=f"mB={int(scenario.vertical_mass)} kg",
        center=(vertical_label_x, vertical_mass_y),
        font=label_font,
        fill_rgb=tuple(int(value) for value in theme.label_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.label_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.label_text_rgb),
        stroke_width_px=int(render_defaults["label_stroke_width_px"]),
    )
    vertical_speed_bbox = _draw_text_tag(
        draw,
        text=f"vB={int(scenario.vertical_speed)} m/s",
        center=(vertical_label_x, vertical_speed_y),
        font=label_font,
        fill_rgb=tuple(int(value) for value in theme.label_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.label_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.label_text_rgb),
        stroke_width_px=int(render_defaults["label_stroke_width_px"]),
    )
    combined_mass_bbox = _draw_text_tag(
        draw,
        text=f"M={int(scenario.total_mass)} kg",
        center=(float(center_x + 140.0), float(center_y - 72.0)),
        font=label_font,
        fill_rgb=tuple(int(value) for value in theme.label_fill_rgb),
        outline_rgb=tuple(int(value) for value in theme.label_outline_rgb),
        text_rgb=tuple(int(value) for value in theme.label_text_rgb),
        stroke_width_px=int(render_defaults["label_stroke_width_px"]),
    )

    # Small axis cue for signed-component tasks.
    axis_origin = (float(table_bbox[0] + 86.0), float(table_bbox[1] + 94.0))
    draw_arrow(
        draw,
        start=axis_origin,
        end=(float(axis_origin[0] + 64.0), float(axis_origin[1])),
        fill=tuple(int(value) for value in theme.option_arrow_rgb),
        width=4,
        head_length_px=14.0,
        head_width_px=12.0,
    )
    draw_arrow(
        draw,
        start=axis_origin,
        end=(float(axis_origin[0]), float(axis_origin[1] - 64.0)),
        fill=tuple(int(value) for value in theme.option_arrow_rgb),
        width=4,
        head_length_px=14.0,
        head_width_px=12.0,
    )
    x_axis_label_bbox = draw_centered_text(
        draw,
        text="+x",
        center=(float(axis_origin[0] + 82.0), float(axis_origin[1])),
        font=small_font,
        fill=tuple(int(value) for value in theme.label_text_rgb),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.label_text_rgb)),
        stroke_width=1,
    )
    y_axis_label_bbox = draw_centered_text(
        draw,
        text="+y",
        center=(float(axis_origin[0]), float(axis_origin[1] - 82.0)),
        font=small_font,
        fill=tuple(int(value) for value in theme.label_text_rgb),
        stroke_fill=tuple(int(value) for value in resolve_text_stroke_fill(theme.label_text_rgb)),
        stroke_width=1,
    )
    axis_bbox = bbox_union_many(
        [axis_origin[0], axis_origin[1] - 76.0, axis_origin[0] + 76.0, axis_origin[1]],
        x_axis_label_bbox,
        y_axis_label_bbox,
    )

    scene_entities: List[Dict[str, Any]] = [
        {
            "entity_id": "collision_table",
            "entity_type": "collision_table",
            "bbox_px": [round(float(value), 3) for value in table_bbox],
            "meta": {"scene_variant": str(scene_spec.scene_variant)},
        },
        {
            "entity_id": "axis_cue",
            "entity_type": "signed_axis_cue",
            "bbox_px": list(axis_bbox),
            "meta": {"positive_x": "right", "positive_y": "up"},
        },
        {
            "entity_id": "horizontal_puck",
            "entity_type": "puck",
            "bbox_px": list(horizontal_puck_bbox),
            "point_px": [round(float(horizontal_center[0]), 3), round(float(horizontal_center[1]), 3)],
            "meta": {
                "label": "A",
                "mass": int(scenario.horizontal_mass),
                "speed": int(scenario.horizontal_speed),
                "direction": "right" if int(scenario.horizontal_sign) > 0 else "left",
            },
        },
        {
            "entity_id": "vertical_puck",
            "entity_type": "puck",
            "bbox_px": list(vertical_puck_bbox),
            "point_px": [round(float(vertical_center[0]), 3), round(float(vertical_center[1]), 3)],
            "meta": {
                "label": "B",
                "mass": int(scenario.vertical_mass),
                "speed": int(scenario.vertical_speed),
                "direction": "up" if int(scenario.vertical_sign) > 0 else "down",
            },
        },
        {
            "entity_id": "stuck_pucks",
            "entity_type": "stuck_puck_pair",
            "bbox_px": list(stuck_bbox),
            "point_px": [round(float(collision_center[0]), 3), round(float(collision_center[1]), 3)],
            "meta": {"total_mass": int(scenario.total_mass)},
        },
        {
            "entity_id": "horizontal_motion_arrow",
            "entity_type": "input_velocity_arrow",
            "bbox_px": list(horizontal_arrow_bbox),
            "segment_px": list(horizontal_arrow_segment),
            "meta": {"puck": "A", "axis": "x", "sign": int(scenario.horizontal_sign)},
        },
        {
            "entity_id": "vertical_motion_arrow",
            "entity_type": "input_velocity_arrow",
            "bbox_px": list(vertical_arrow_bbox),
            "segment_px": list(vertical_arrow_segment),
            "meta": {"puck": "B", "axis": "y", "sign": int(scenario.vertical_sign)},
        },
        {
            "entity_id": "horizontal_mass_label",
            "entity_type": "mass_label",
            "bbox_px": list(horizontal_mass_bbox),
            "meta": {"puck": "A", "value": int(scenario.horizontal_mass)},
        },
        {
            "entity_id": "horizontal_speed_label",
            "entity_type": "speed_label",
            "bbox_px": list(horizontal_speed_bbox),
            "meta": {"puck": "A", "value": int(scenario.horizontal_speed)},
        },
        {
            "entity_id": "vertical_mass_label",
            "entity_type": "mass_label",
            "bbox_px": list(vertical_mass_bbox),
            "meta": {"puck": "B", "value": int(scenario.vertical_mass)},
        },
        {
            "entity_id": "vertical_speed_label",
            "entity_type": "speed_label",
            "bbox_px": list(vertical_speed_bbox),
            "meta": {"puck": "B", "value": int(scenario.vertical_speed)},
        },
        {
            "entity_id": "combined_mass_label",
            "entity_type": "combined_mass_label",
            "bbox_px": list(combined_mass_bbox),
            "meta": {"value": int(scenario.total_mass)},
        },
    ]
    horizontal_component_witness_bbox = bbox_union_many(
        horizontal_puck_bbox,
        horizontal_arrow_bbox,
        horizontal_mass_bbox,
        horizontal_speed_bbox,
        combined_mass_bbox,
    )
    vertical_component_witness_bbox = bbox_union_many(
        vertical_puck_bbox,
        vertical_arrow_bbox,
        vertical_mass_bbox,
        vertical_speed_bbox,
        combined_mass_bbox,
    )
    scene_entities.extend(
        [
            {
                "entity_id": "horizontal_component_witness",
                "entity_type": "component_momentum_witness",
                "bbox_px": list(horizontal_component_witness_bbox),
                "meta": {
                    "axis": "x",
                    "puck": "A",
                    "members": [
                        "horizontal_puck",
                        "horizontal_motion_arrow",
                        "horizontal_mass_label",
                        "horizontal_speed_label",
                        "combined_mass_label",
                    ],
                },
            },
            {
                "entity_id": "vertical_component_witness",
                "entity_type": "component_momentum_witness",
                "bbox_px": list(vertical_component_witness_bbox),
                "meta": {
                    "axis": "y",
                    "puck": "B",
                    "members": [
                        "vertical_puck",
                        "vertical_motion_arrow",
                        "vertical_mass_label",
                        "vertical_speed_label",
                        "combined_mass_label",
                    ],
                },
            },
        ]
    )

    option_bboxes: Dict[str, List[float]] = {}
    if bool(render_defaults.get("show_candidate_options", True)):
        option_letters = tuple(str(letter) for letter in scene_spec.option_letters)
        option_top = float(render_defaults["option_panel_top_px"])
        option_left = float(render_defaults["option_cell_left_px"])
        option_width = float(render_defaults["option_cell_width_px"])
        option_arrow_length = float(render_defaults["option_arrow_length_px"])
        for option_index, letter in enumerate(option_letters):
            cell_center_x = float(
                option_left + (option_index * option_width) + (0.5 * option_width)
            )
            cell_center_y = float(option_top + 76.0)
            letter_bbox = _draw_text_tag(
                draw,
                text=str(letter),
                center=(float(cell_center_x), float(option_top + 24.0)),
                font=option_font,
                fill_rgb=tuple(int(value) for value in theme.label_fill_rgb),
                outline_rgb=tuple(int(value) for value in theme.option_outline_rgb),
                text_rgb=tuple(int(value) for value in theme.label_text_rgb),
                stroke_width_px=int(render_defaults["label_stroke_width_px"]),
            )
            unit = _screen_unit_from_physics_angle(
                float(scene_spec.option_angles_degrees[str(letter)])
            )
            arrow_start = (
                float(cell_center_x - (unit[0] * option_arrow_length * 0.5)),
                float(cell_center_y - (unit[1] * option_arrow_length * 0.5)),
            )
            arrow_end = (
                float(cell_center_x + (unit[0] * option_arrow_length * 0.5)),
                float(cell_center_y + (unit[1] * option_arrow_length * 0.5)),
            )
            draw_arrow(
                draw,
                start=arrow_start,
                end=arrow_end,
                fill=tuple(int(value) for value in theme.option_arrow_rgb),
                width=int(render_defaults["option_arrow_width_px"]),
                head_length_px=float(render_defaults["option_arrow_head_length_px"]),
                head_width_px=float(render_defaults["option_arrow_head_width_px"]),
            )
            arrow_bbox = _arrow_bbox(
                arrow_start,
                arrow_end,
                padding_px=float(render_defaults["option_arrow_head_width_px"]),
            )
            option_bbox = bbox_union_many(letter_bbox, arrow_bbox)
            option_bboxes[str(letter)] = list(option_bbox)
            scene_entities.append(
                {
                    "entity_id": f"option_{str(letter)}",
                    "entity_type": "candidate_resultant_arrow",
                    "bbox_px": list(option_bbox),
                    "meta": {
                        "option_letter": str(letter),
                        "angle_degrees": float(scene_spec.option_angles_degrees[str(letter)]),
                        "is_correct": str(letter) == str(scene_spec.correct_option_letter),
                    },
                }
            )

    entity_bbox_map = {
        str(entity["entity_id"]): list(entity["bbox_px"])
        for entity in scene_entities
        if entity.get("bbox_px") is not None
    }
    entity_point_map = {
        str(entity["entity_id"]): list(entity["point_px"])
        for entity in scene_entities
        if entity.get("point_px") is not None
    }
    annotation_points = [
        list(entity_point_map[entity_id])
        for entity_id in scene_spec.annotation_entity_ids
        if str(entity_id) in entity_point_map
    ]
    annotation_point_map = {
        ANNOTATION_ENTITY_KEY_BY_ID[str(entity_id)]: list(entity_point_map[str(entity_id)])
        for entity_id in scene_spec.annotation_entity_ids
        if str(entity_id) in entity_point_map
    }
    render_map = {
        "accent_color_name": str(accent_color_name),
        "technical_diagram_frame_mode": str(getattr(diagram_style, "frame_mode", "none")),
        "table_bbox_px": [round(float(value), 3) for value in table_bbox],
        "collision_center_px": [round(float(center_x), 3), round(float(center_y), 3)],
        "horizontal_puck_bbox_px": list(horizontal_puck_bbox),
        "horizontal_puck_center_px": [round(float(horizontal_center[0]), 3), round(float(horizontal_center[1]), 3)],
        "vertical_puck_bbox_px": list(vertical_puck_bbox),
        "vertical_puck_center_px": [round(float(vertical_center[0]), 3), round(float(vertical_center[1]), 3)],
        "stuck_pucks_bbox_px": list(stuck_bbox),
        "stuck_pucks_center_px": [round(float(collision_center[0]), 3), round(float(collision_center[1]), 3)],
        "horizontal_motion_arrow_bbox_px": list(horizontal_arrow_bbox),
        "vertical_motion_arrow_bbox_px": list(vertical_arrow_bbox),
        "horizontal_motion_arrow_segment_px": [list(point) for point in horizontal_arrow_segment],
        "vertical_motion_arrow_segment_px": [list(point) for point in vertical_arrow_segment],
        "motion_arrow_segments_px": {
            "horizontal_motion_arrow": [list(point) for point in horizontal_arrow_segment],
            "vertical_motion_arrow": [list(point) for point in vertical_arrow_segment],
        },
        "horizontal_component_witness_bbox_px": list(horizontal_component_witness_bbox),
        "vertical_component_witness_bbox_px": list(vertical_component_witness_bbox),
        "directionangle_degrees": round(float(final_angle), 3),
        "direction_label": str(scene_spec.direction_label),
        "show_candidate_options": bool(render_defaults.get("show_candidate_options", True)),
        "option_bboxes_px": {str(letter): list(bbox) for letter, bbox in option_bboxes.items()},
        "option_letters": list(scene_spec.option_letters),
        "option_angles_degrees": dict(scene_spec.option_angles_degrees),
        "correct_option_letter": str(scene_spec.correct_option_letter),
        "annotation_entity_ids": list(scene_spec.annotation_entity_ids),
        "annotation_key_by_entity_id": dict(ANNOTATION_ENTITY_KEY_BY_ID),
        "entity_points_px": {str(key): list(value) for key, value in entity_point_map.items()},
        "annotation_points_px": [list(point) for point in annotation_points],
        "annotation_keyed_points_px": {str(key): list(value) for key, value in annotation_point_map.items()},
    }
    return StickyRenderedScene(
        image=image,
        annotation_points=[list(point) for point in annotation_points],
        annotation_point_map={str(key): list(value) for key, value in annotation_point_map.items()},
        annotation_entity_ids=list(scene_spec.annotation_entity_ids),
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
    )


def resolve_sticky_layout_placement(
    *,
    render_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    namespace: str,
    instance_seed: int,
    canvas_width: int,
    canvas_height: int,
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve a whole-diagram offset before rendering and annotation projection."""

    table_left = float(render_defaults["table_left_px"])
    table_top = float(render_defaults["table_top_px"])
    table_right = table_left + float(render_defaults["table_width_px"])
    table_bottom = table_top + float(render_defaults["table_height_px"])
    if bool(render_defaults.get("show_candidate_options", True)):
        option_count = int(render_defaults.get("option_count", len(OPTION_LETTERS)))
        content_left = min(table_left, float(render_defaults["option_cell_left_px"]))
        content_top = min(table_top, float(render_defaults["option_panel_top_px"]))
        content_right = max(
            table_right,
            float(render_defaults["option_cell_left_px"])
            + (float(render_defaults["option_cell_width_px"]) * option_count),
        )
        content_bottom = max(table_bottom, float(render_defaults["option_panel_top_px"]) + 144.0)
    else:
        content_left = table_left
        content_top = table_top
        content_right = table_right
        content_bottom = table_bottom
    base_bbox = [round(content_left, 3), round(content_top, 3), round(content_right, 3), round(content_bottom, 3)]
    jitter = resolve_layout_jitter(
        params,
        rendering_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.layout",
    )
    min_margin = int(jitter.get("min_margin_px", 24))
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
    for key in ("table_left_px", "collision_center_x_px", "compact_collision_center_x_px", "option_cell_left_px"):
        adjusted[key] = int(adjusted[key]) + int(dx)
    for key in ("table_top_px", "collision_center_y_px", "compact_collision_center_y_px", "option_panel_top_px"):
        adjusted[key] = int(adjusted[key]) + int(dy)

    final_bbox = [
        round(float(content_left) + float(dx), 3),
        round(float(content_top) + float(dy), 3),
        round(float(content_right) + float(dx), 3),
        round(float(content_bottom) + float(dy), 3),
    ]
    placement = dict(jitter)
    placement.update(
        {
            "mode": "whole_collision_diagram_offset",
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



def _resolve_render_defaults(
    *,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    fallback: Any,
    keys: Sequence[str],
    instance_seed: int,
    namespace: str,
) -> Dict[str, int]:
    """Resolve integer render defaults for one collision scene."""

    return {
        str(key): resolve_render_int(
            params,
            rendering_defaults,
            str(key),
            int(getattr(fallback, str(key))),
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
        for key in keys
    }


def prepare_aftermath_render(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    spec: CollisionAftermathSpec,
    rendering_defaults: Mapping[str, Any],
    fallback: Any,
    namespace: str,
) -> PreparedRender:
    """Render one complete collision-aftermath diagram with final annotations."""

    keys = (
        "table_left_px",
        "table_top_px",
        "table_width_px",
        "table_height_px",
        "table_corner_radius_px",
        "impact_center_x_px",
        "impact_center_y_px",
        "compact_impact_center_y_px",
        "puck_radius_px",
        "impact_radius_px",
        "target_distance_px",
        "trail_start_gap_px",
        "incoming_arrow_inner_gap_px",
        "incoming_arrow_length_px",
        "candidate_label_offset_px",
        "motion_arrow_width_px",
        "candidate_arrow_width_px",
        "arrow_head_length_px",
        "arrow_head_width_px",
        "candidate_arrow_head_length_px",
        "candidate_arrow_head_width_px",
        "title_font_size_px",
        "label_font_size_px",
        "puck_font_size_px",
        "label_stroke_width_px",
        "grid_spacing_px",
    )
    canvas_width = int(params.get("canvas_width", group_default(rendering_defaults, "canvas_width", fallback.canvas_width)))
    canvas_height = int(params.get("canvas_height", group_default(rendering_defaults, "canvas_height", fallback.canvas_height)))
    background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
        scene_id=SCENE_ID,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        instance_seed=int(instance_seed),
        params=params,
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font",
        params=params,
    )
    render_defaults = _resolve_render_defaults(
        params=params,
        rendering_defaults=rendering_defaults,
        fallback=fallback,
        keys=keys,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    render_defaults, layout_meta = resolve_aftermath_layout_placement(
        render_defaults=render_defaults,
        params=params,
        rendering_defaults=rendering_defaults,
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
    )
    rendered = render_aftermath_scene(
        background=background,
        render_defaults=render_defaults,
        accent_color_name=str(params.get("_resolved_accent_color_name", "blue")),
        spec=spec,
        font_family=str(font_family),
        diagram_style=diagram_style,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return PreparedRender(
        image=image,
        annotation_bbox_map=dict(rendered.annotation_bbox_map),
        annotation_point_map={},
        annotation_entity_ids=list(rendered.annotation_bbox_map.keys()),
        scene_entities=list(rendered.scene_entities),
        render_map={
            **dict(rendered.render_map),
            "technical_diagram_style": dict(diagram_style_meta),
            "background_style": dict(background_meta),
            "layout_placement": dict(layout_meta),
            "post_image_noise": dict(post_noise_meta),
        },
        background_meta=dict(background_meta),
        diagram_style_meta=dict(diagram_style_meta),
        layout_placement_meta=dict(layout_meta),
        post_noise_meta=dict(post_noise_meta),
        font_family=str(font_family),
    )


def prepare_sticky_render(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scene_spec: StickySceneSpec,
    rendering_defaults: Mapping[str, Any],
    fallback: Any,
    namespace: str,
) -> PreparedRender:
    """Render one complete sticky-collision diagram with final annotations."""

    keys = (
        "table_left_px",
        "table_top_px",
        "table_width_px",
        "table_height_px",
        "table_corner_radius_px",
        "collision_center_x_px",
        "collision_center_y_px",
        "compact_collision_center_x_px",
        "compact_collision_center_y_px",
        "horizontal_start_distance_px",
        "vertical_start_distance_px",
        "compact_start_distance_delta_px",
        "puck_radius_px",
        "stuck_radius_px",
        "motion_arrow_width_px",
        "final_arrow_width_px",
        "arrow_head_length_px",
        "arrow_head_width_px",
        "label_font_size_px",
        "puck_font_size_px",
        "option_font_size_px",
        "label_stroke_width_px",
        "option_panel_top_px",
        "option_cell_left_px",
        "option_cell_width_px",
        "option_arrow_length_px",
        "option_arrow_width_px",
        "option_arrow_head_length_px",
        "option_arrow_head_width_px",
        "grid_spacing_px",
    )
    canvas_width = int(params.get("canvas_width", group_default(rendering_defaults, "canvas_width", fallback.canvas_width)))
    canvas_height = int(params.get("canvas_height", group_default(rendering_defaults, "canvas_height", fallback.canvas_height)))
    background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
        scene_id=SCENE_ID,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        instance_seed=int(instance_seed),
        params=params,
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font",
        params=params,
    )
    render_defaults = _resolve_render_defaults(
        params=params,
        rendering_defaults=rendering_defaults,
        fallback=fallback,
        keys=keys,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    render_defaults["show_candidate_options"] = bool(
        params.get(
            "show_candidate_options",
            group_default(
                rendering_defaults,
                "show_candidate_options",
                fallback.show_candidate_options,
            ),
        )
    )
    render_defaults["option_count"] = len(tuple(scene_spec.option_letters))
    render_defaults, layout_meta = resolve_sticky_layout_placement(
        render_defaults=render_defaults,
        params=params,
        rendering_defaults=rendering_defaults,
        namespace=str(namespace),
        instance_seed=int(instance_seed),
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
    )
    rendered = render_sticky_scene(
        background=background,
        render_defaults=render_defaults,
        accent_color_name=str(params.get("_resolved_accent_color_name", "blue")),
        scene_spec=scene_spec,
        font_family=str(font_family),
        diagram_style=diagram_style,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return PreparedRender(
        image=image,
        annotation_bbox_map={},
        annotation_point_map=dict(rendered.annotation_point_map),
        annotation_entity_ids=list(rendered.annotation_entity_ids),
        scene_entities=list(rendered.scene_entities),
        render_map={
            **dict(rendered.render_map),
            "technical_diagram_style": dict(diagram_style_meta),
            "background_style": dict(background_meta),
            "layout_placement": dict(layout_meta),
            "post_image_noise": dict(post_noise_meta),
        },
        background_meta=dict(background_meta),
        diagram_style_meta=dict(diagram_style_meta),
        layout_placement_meta=dict(layout_meta),
        post_noise_meta=dict(post_noise_meta),
        font_family=str(font_family),
    )


def prepare_sticky_scene(
    instance_seed: int,
    max_attempts: int,
    axes: Any,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    fallback: Any,
    namespace: str,
) -> Tuple[StickySceneSpec, PreparedRender]:
    """Sample a valid sticky setup and render it with the same resolved scene axes."""

    scene_spec = sample_sticky_spec_with_retries(
        instance_seed=int(instance_seed),
        max_attempts=int(max_attempts),
        axes=axes,
        params=params,
        defaults=generation_defaults,
        namespace=str(namespace),
    )
    render_params = {**dict(params), "_resolved_accent_color_name": str(axes.accent_color_name)}
    rendered = prepare_sticky_render(
        instance_seed=int(instance_seed),
        params=render_params,
        scene_spec=scene_spec,
        rendering_defaults=rendering_defaults,
        fallback=fallback,
        namespace=str(namespace),
    )
    return scene_spec, rendered


def prepare_resolved_sticky_scene_or_raise(
    *,
    instance_seed: int,
    max_attempts: int,
    axes: Any,
    params: Mapping[str, Any],
    generation_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    fallback: Any,
    namespace: str,
) -> Tuple[StickySceneSpec, PreparedRender]:
    """Prepare a rendered sticky-collision scene for already-resolved axes."""

    try:
        return prepare_sticky_scene(
            int(instance_seed),
            int(max_attempts),
            axes,
            params,
            generation_defaults,
            rendering_defaults,
            fallback,
            str(namespace),
        )
    except ValueError as exc:
        raise RuntimeError("failed to generate a valid sticky-collision scene") from exc


def sticky_input_motion_arrow_segments(rendered: PreparedRender) -> List[List[List[float]]]:
    """Return both input velocity-arrow segments from a rendered sticky scene."""

    return [
        [list(point) for point in rendered.render_map["horizontal_motion_arrow_segment_px"]],
        [list(point) for point in rendered.render_map["vertical_motion_arrow_segment_px"]],
    ]
