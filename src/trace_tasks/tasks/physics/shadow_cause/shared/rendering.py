"""Rendering helpers for the shadow-cause physics scene."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.shared.drawing import draw_centered_text
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .state import DIRECTION_VECTORS, LampSpec, RenderedShadowCauseScene, ShadowSceneSpec


LAMP_GLOW_RGB = (255, 213, 91)
LAMP_CORE_RGB = (255, 241, 173)
LAMP_STROKE_RGB = (117, 86, 30)


def _bbox(values: Sequence[float]) -> List[float]:
    """Return one rounded pixel box as JSON-stable floats."""

    return [round(float(value), 3) for value in values]


def _clip_bbox(values: Sequence[float], *, width: int, height: int) -> List[float]:
    """Clip one box to the rendered image extents."""

    x0, y0, x1, y1 = [float(value) for value in values]
    return _bbox(
        (
            max(0.0, min(float(width), x0)),
            max(0.0, min(float(height), y0)),
            max(0.0, min(float(width), x1)),
            max(0.0, min(float(height), y1)),
        )
    )


def _expand_bbox(values: Sequence[float], padding: float) -> List[float]:
    """Expand one bounding box by a fixed padding."""

    return _bbox(
        (
            float(values[0]) - float(padding),
            float(values[1]) - float(padding),
            float(values[2]) + float(padding),
            float(values[3]) + float(padding),
        )
    )


def _bbox_union(boxes: Sequence[Sequence[float]]) -> List[float]:
    """Return the bounding box enclosing all boxes."""

    return _bbox(
        (
            min(float(box[0]) for box in boxes),
            min(float(box[1]) for box in boxes),
            max(float(box[2]) for box in boxes),
            max(float(box[3]) for box in boxes),
        )
    )


def _blend_rgb(a: Sequence[int], b: Sequence[int], amount: float) -> Tuple[int, int, int]:
    """Blend two RGB colors by the requested amount."""

    return tuple(
        int(round((1.0 - float(amount)) * int(a[idx]) + float(amount) * int(b[idx])))
        for idx in range(3)
    )


def _draw_floor(
    draw: ImageDraw.ImageDraw,
    *,
    floor_bbox: Sequence[float],
    style: Any,
    font_family: str,
    render_defaults: Mapping[str, int],
) -> None:
    """Draw the shadow-receiving floor and candidate title."""

    left, top, right, bottom = [float(value) for value in floor_bbox]
    draw.rounded_rectangle(
        floor_bbox,
        radius=22,
        fill=tuple(int(v) for v in style.panel_fill_rgb),
        outline=tuple(int(v) for v in style.panel_border_rgb),
        width=3,
    )
    grid_rgb = tuple(int(v) for v in style.grid_minor_rgb)
    for x in range(int(left) + 36, int(right), 56):
        draw.line([(float(x), top + 34.0), (float(x), bottom - 26.0)], fill=grid_rgb, width=1)
    for y in range(int(top) + 38, int(bottom), 48):
        draw.line([(left + 26.0, float(y)), (right - 26.0, float(y))], fill=grid_rgb, width=1)
    title_font = load_font(int(render_defaults["title_font_size_px"]), bold=True, font_family=font_family)
    label_rgb = tuple(int(v) for v in style.label_rgb)
    draw_centered_text(
        draw,
        text="light source candidates",
        center=((left + right) * 0.5, top + 28.0),
        font=title_font,
        fill=label_rgb,
        stroke_fill=resolve_text_stroke_fill(label_rgb),
        stroke_width=1,
    )


def _draw_shadow(
    image: Image.Image,
    *,
    object_base: Tuple[float, float],
    shadow_direction: str,
    render_defaults: Mapping[str, int],
) -> Tuple[Image.Image, List[float], Dict[str, Any]]:
    """Draw the cast shadow and return its witness box."""

    width, height = image.size
    dx, dy = DIRECTION_VECTORS[str(shadow_direction)]
    perp = (-float(dy), float(dx))
    length = float(render_defaults["shadow_length_px"])
    base_width = float(render_defaults["shadow_base_width_px"])
    tip_width = float(render_defaults["shadow_tip_width_px"])
    base_center = (
        float(object_base[0] + dx * 22.0),
        float(object_base[1] + dy * 22.0),
    )
    tip_center = (
        float(object_base[0] + dx * length),
        float(object_base[1] + dy * length),
    )
    polygon = [
        (base_center[0] + perp[0] * base_width * 0.5, base_center[1] + perp[1] * base_width * 0.5),
        (tip_center[0] + perp[0] * tip_width * 0.5, tip_center[1] + perp[1] * tip_width * 0.5),
        (tip_center[0] - perp[0] * tip_width * 0.5, tip_center[1] - perp[1] * tip_width * 0.5),
        (base_center[0] - perp[0] * base_width * 0.5, base_center[1] - perp[1] * base_width * 0.5),
    ]
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    overlay_draw = ImageDraw.Draw(overlay)
    overlay_draw.polygon(polygon, fill=(32, 38, 44, 112))
    tip_ellipse = (
        tip_center[0] - tip_width * 0.58,
        tip_center[1] - tip_width * 0.36,
        tip_center[0] + tip_width * 0.58,
        tip_center[1] + tip_width * 0.36,
    )
    overlay_draw.ellipse(tip_ellipse, fill=(32, 38, 44, 102))
    composited = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    polygon_bbox = _bbox_union(
        [
            (
                min(point[0] for point in polygon),
                min(point[1] for point in polygon),
                max(point[0] for point in polygon),
                max(point[1] for point in polygon),
            ),
            tip_ellipse,
        ]
    )
    annotation_bbox = _clip_bbox(_expand_bbox(polygon_bbox, 5.0), width=width, height=height)
    return (
        composited,
        annotation_bbox,
        {
            "polygon_px": [[round(float(x), 3), round(float(y), 3)] for x, y in polygon],
            "tip_center_px": [round(float(tip_center[0]), 3), round(float(tip_center[1]), 3)],
            "length_px": round(float(length), 3),
        },
    )


def _draw_block_object(
    draw: ImageDraw.ImageDraw,
    *,
    base_center: Tuple[float, float],
    size: float,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
) -> List[float]:
    """Draw one cuboid-like block and return its box."""

    cx, by = float(base_center[0]), float(base_center[1])
    front = (cx - size * 0.50, by - size * 0.72, cx + size * 0.50, by + size * 0.12)
    shift = (size * 0.24, -size * 0.22)
    top = [
        (front[0], front[1]),
        (front[0] + shift[0], front[1] + shift[1]),
        (front[2] + shift[0], front[1] + shift[1]),
        (front[2], front[1]),
    ]
    side = [
        (front[2], front[1]),
        (front[2] + shift[0], front[1] + shift[1]),
        (front[2] + shift[0], front[3] + shift[1]),
        (front[2], front[3]),
    ]
    draw.polygon(top, fill=_blend_rgb(fill, (255, 255, 255), 0.30), outline=outline)
    draw.polygon(side, fill=_blend_rgb(fill, (0, 0, 0), 0.16), outline=outline)
    draw.rounded_rectangle(front, radius=9, fill=fill, outline=outline, width=3)
    shifted_front = (
        front[0] + shift[0],
        front[1] + shift[1],
        front[2] + shift[0],
        front[3] + shift[1],
    )
    return _expand_bbox(_bbox_union([front, shifted_front]), 3.0)


def _draw_cylinder_object(
    draw: ImageDraw.ImageDraw,
    *,
    base_center: Tuple[float, float],
    size: float,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
) -> List[float]:
    """Draw one cylinder object and return its box."""

    cx, by = float(base_center[0]), float(base_center[1])
    half_w = size * 0.44
    body_top = by - size * 0.92
    body_bottom = by + size * 0.08
    ellipse_h = size * 0.22
    body = (cx - half_w, body_top, cx + half_w, body_bottom)
    draw.rectangle(body, fill=fill, outline=outline, width=3)
    draw.ellipse(
        (cx - half_w, body_top - ellipse_h * 0.5, cx + half_w, body_top + ellipse_h * 0.5),
        fill=_blend_rgb(fill, (255, 255, 255), 0.26),
        outline=outline,
        width=3,
    )
    draw.arc(
        (cx - half_w, body_bottom - ellipse_h * 0.5, cx + half_w, body_bottom + ellipse_h * 0.5),
        0,
        180,
        fill=outline,
        width=3,
    )
    draw.arc(
        (cx - half_w, body_bottom - ellipse_h * 0.5, cx + half_w, body_bottom + ellipse_h * 0.5),
        180,
        360,
        fill=_blend_rgb(outline, (255, 255, 255), 0.2),
        width=2,
    )
    return _bbox(
        (
            cx - half_w - 3.0,
            body_top - ellipse_h * 0.5 - 3.0,
            cx + half_w + 3.0,
            body_bottom + ellipse_h * 0.5 + 3.0,
        )
    )


def _draw_sphere_object(
    draw: ImageDraw.ImageDraw,
    *,
    base_center: Tuple[float, float],
    size: float,
    fill: Tuple[int, int, int],
    outline: Tuple[int, int, int],
) -> List[float]:
    """Draw one sphere object and return its box."""

    cx, by = float(base_center[0]), float(base_center[1])
    radius = size * 0.48
    cy = by - radius * 0.70
    bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(bbox, fill=fill, outline=outline, width=3)
    highlight = (cx - radius * 0.46, cy - radius * 0.50, cx - radius * 0.04, cy - radius * 0.12)
    draw.ellipse(highlight, fill=_blend_rgb(fill, (255, 255, 255), 0.48))
    draw.arc(
        (cx - radius * 0.72, cy - radius * 0.54, cx + radius * 0.70, cy + radius * 0.80),
        208,
        318,
        fill=_blend_rgb(outline, fill, 0.40),
        width=2,
    )
    return _expand_bbox(bbox, 3.0)


def _draw_object(
    draw: ImageDraw.ImageDraw,
    *,
    spec: ShadowSceneSpec,
    base_center: Tuple[float, float],
    render_defaults: Mapping[str, int],
) -> List[float]:
    """Draw the shadow-casting object and return its box."""

    fill = tuple(int(value) for value in spec.object_fill_rgb)
    outline = tuple(max(0, int(value) - 70) for value in fill)
    size = float(render_defaults["object_size_px"])
    if spec.object_shape == "cylinder":
        return _draw_cylinder_object(draw, base_center=base_center, size=size, fill=fill, outline=outline)
    if spec.object_shape == "sphere":
        return _draw_sphere_object(draw, base_center=base_center, size=size, fill=fill, outline=outline)
    return _draw_block_object(draw, base_center=base_center, size=size, fill=fill, outline=outline)


def _draw_lamp(
    draw: ImageDraw.ImageDraw,
    *,
    lamp: LampSpec,
    object_base: Tuple[float, float],
    style: Any,
    font_family: str,
    render_defaults: Mapping[str, int],
    canvas_width: int,
    canvas_height: int,
) -> Dict[str, Any]:
    """Draw one labeled candidate lamp and metadata."""

    cx, cy = float(lamp.center_px[0]), float(lamp.center_px[1])
    radius = float(render_defaults["lamp_bulb_radius_px"])
    label_font = load_font(int(render_defaults["lamp_label_font_size_px"]), bold=True, font_family=font_family)
    dx = float(cx - object_base[0])
    dy = float(cy - object_base[1])
    distance = max(1.0, math.hypot(dx, dy))
    unit_x = dx / distance
    unit_y = dy / distance
    stand_end = (cx - unit_x * (radius + 22.0), cy - unit_y * (radius + 22.0))
    draw.line(
        [stand_end, (cx - unit_x * radius * 0.55, cy - unit_y * radius * 0.55)],
        fill=tuple(int(v) for v in style.stroke_rgb),
        width=4,
    )
    glow_bbox = (cx - radius * 1.62, cy - radius * 1.62, cx + radius * 1.62, cy + radius * 1.62)
    draw.ellipse(glow_bbox, fill=(255, 230, 138), outline=None)
    bulb_bbox = (cx - radius, cy - radius, cx + radius, cy + radius)
    draw.ellipse(bulb_bbox, fill=LAMP_GLOW_RGB, outline=LAMP_STROKE_RGB, width=3)
    core_r = radius * 0.48
    draw.ellipse((cx - core_r, cy - core_r, cx + core_r, cy + core_r), fill=LAMP_CORE_RGB)
    for angle in range(0, 360, 60):
        radians = math.radians(float(angle))
        start = (cx + math.cos(radians) * radius * 1.22, cy + math.sin(radians) * radius * 1.22)
        end = (cx + math.cos(radians) * radius * 1.58, cy + math.sin(radians) * radius * 1.58)
        draw.line([start, end], fill=LAMP_STROKE_RGB, width=2)

    label_center = (
        max(34.0, min(float(canvas_width - 34), cx - unit_x * 44.0)),
        max(34.0, min(float(canvas_height - 34), cy - unit_y * 44.0)),
    )
    label_bbox = (
        label_center[0] - 23.0,
        label_center[1] - 21.0,
        label_center[0] + 23.0,
        label_center[1] + 21.0,
    )
    draw.rounded_rectangle(
        label_bbox,
        radius=10,
        fill=tuple(int(v) for v in style.label_fill_rgb),
        outline=tuple(int(v) for v in style.label_border_rgb),
        width=2,
    )
    text_rgb = tuple(int(v) for v in style.label_rgb)
    text_bbox = draw_centered_text(
        draw,
        text=str(lamp.label),
        center=label_center,
        font=label_font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=int(render_defaults["label_stroke_width_px"]),
    )
    lamp_bbox = _clip_bbox(_expand_bbox(_bbox_union([glow_bbox, bulb_bbox]), 4.0), width=canvas_width, height=canvas_height)
    clipped_label_bbox = _clip_bbox(_bbox_union([label_bbox, text_bbox]), width=canvas_width, height=canvas_height)
    return {
        "label": str(lamp.label),
        "direction": str(lamp.direction),
        "center_px": [round(float(cx), 3), round(float(cy), 3)],
        "lamp_bbox_px": lamp_bbox,
        "label_bbox_px": clipped_label_bbox,
        "option_bbox_px": _clip_bbox(_bbox_union([lamp_bbox, clipped_label_bbox]), width=canvas_width, height=canvas_height),
    }


def render_shadow_cause_scene(
    *,
    image: Image.Image,
    spec: ShadowSceneSpec,
    render_defaults: Mapping[str, int],
    font_family: str,
    style: Any,
) -> RenderedShadowCauseScene:
    """Render the full shadow-cause diagram and projected boxes."""

    width, height = image.size
    draw = ImageDraw.Draw(image)
    floor_bbox = (
        float(render_defaults["floor_left_px"]),
        float(render_defaults["floor_top_px"]),
        float(width - int(render_defaults["floor_right_margin_px"])),
        float(height - int(render_defaults["floor_bottom_margin_px"])),
    )
    _draw_floor(
        draw,
        floor_bbox=floor_bbox,
        style=style,
        font_family=font_family,
        render_defaults=render_defaults,
    )

    object_base = (
        float(render_defaults["object_center_x_px"]),
        float(render_defaults["object_base_y_px"]),
    )
    image, shadow_bbox, shadow_meta = _draw_shadow(
        image,
        object_base=object_base,
        shadow_direction=str(spec.shadow_direction),
        render_defaults=render_defaults,
    )
    draw = ImageDraw.Draw(image)
    object_bbox = _clip_bbox(
        _draw_object(
            draw,
            spec=spec,
            base_center=object_base,
            render_defaults=render_defaults,
        ),
        width=width,
        height=height,
    )
    lamp_records = [
        _draw_lamp(
            draw,
            lamp=lamp,
            object_base=object_base,
            style=style,
            font_family=font_family,
            render_defaults=render_defaults,
            canvas_width=width,
            canvas_height=height,
        )
        for lamp in spec.lamps
    ]
    context_bbox_map = {
        "object": list(object_bbox),
        "shadow": list(shadow_bbox),
    }
    entities: List[Dict[str, Any]] = [
        {
            "entity_id": "shadow_object",
            "entity_type": f"{spec.object_shape}_object",
            "bbox_px": list(object_bbox),
            "meta": {
                "object_shape": str(spec.object_shape),
                "object_fill_rgb": list(spec.object_fill_rgb),
            },
        },
        {
            "entity_id": "cast_shadow",
            "entity_type": "cast_shadow",
            "bbox_px": list(shadow_bbox),
            "meta": {
                "shadow_direction": str(spec.shadow_direction),
                "source_direction": str(spec.source_direction),
            },
        },
    ]
    for record in lamp_records:
        entities.append(
            {
                "entity_id": f"lamp_{record['label']}",
                "entity_type": "candidate_light_source",
                "bbox_px": list(record["lamp_bbox_px"]),
                "meta": {
                    "option_letter": str(record["label"]),
                    "direction": str(record["direction"]),
                    "is_correct": str(record["label"]) == str(spec.correct_option_letter),
                },
            }
        )
    render_map = {
        "floor_bbox_px": _bbox(floor_bbox),
        "object_base_px": [round(float(object_base[0]), 3), round(float(object_base[1]), 3)],
        "object_bbox_px": list(object_bbox),
        "object_shape": str(spec.object_shape),
        "shadow_direction": str(spec.shadow_direction),
        "source_direction": str(spec.source_direction),
        "shadow_bbox_px": list(shadow_bbox),
        "shadow_geometry": dict(shadow_meta),
        "correct_option_letter": str(spec.correct_option_letter),
        "candidate_light_sources": {str(record["label"]): dict(record) for record in lamp_records},
        "candidate_directions": {str(record["label"]): str(record["direction"]) for record in lamp_records},
        "context_bboxes_px": dict(context_bbox_map),
    }
    return RenderedShadowCauseScene(
        image=image,
        context_bbox_map={str(key): list(value) for key, value in context_bbox_map.items()},
        scene_entities=[dict(entity) for entity in entities],
        render_map=dict(render_map),
    )
