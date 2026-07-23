"""Rendering primitives for analog-meter diagrams."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.bbox_projection import bbox_union_many as _bbox_union
from trace_tasks.tasks.shared.drawing import draw_centered_text
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .annotations import bbox, normalize_bbox_map
from .state import DEFAULTS, SCENE_ID, SCENE_NAMESPACE, MeterScenario, RenderedScene


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def angle_for_value(value: float, *, scale_max: int) -> float:
    """Map a scale value onto the visible analog-meter arc."""

    fraction = max(0.0, min(1.0, float(value) / float(scale_max)))
    return math.radians(210.0 + (120.0 * fraction))


def point_on_meter(center: Tuple[float, float], radius: float, angle_rad: float) -> Tuple[float, float]:
    """Return a point on a circular meter arc."""

    return (
        float(center[0] + float(radius) * math.cos(float(angle_rad))),
        float(center[1] + float(radius) * math.sin(float(angle_rad))),
    )


def draw_meter_face(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    scenario: MeterScenario,
    style: Any,
    font_family: str,
    accent_rgb: Tuple[int, int, int],
    render_defaults: Mapping[str, Any],
) -> Tuple[Dict[str, List[float]], Dict[str, Any]]:
    """Draw one meter face and return role-keyed witness boxes."""

    profile = scenario.profile
    cx, cy = float(center[0]), float(center[1])
    face_radius = float(render_defaults.get("face_radius_px", DEFAULTS.face_radius_px))
    scale_radius = float(render_defaults.get("scale_radius_px", DEFAULTS.scale_radius_px))
    needle_radius = float(render_defaults.get("needle_radius_px", DEFAULTS.needle_radius_px))
    tick_font = load_font(
        int(render_defaults.get("tick_font_size_px", DEFAULTS.tick_font_size_px)),
        bold=True,
        font_family=font_family,
    )
    unit_font = load_font(
        int(render_defaults.get("unit_font_size_px", DEFAULTS.unit_font_size_px)),
        bold=True,
        font_family=font_family,
    )
    title_font = load_font(
        int(render_defaults.get("title_font_size_px", DEFAULTS.title_font_size_px)),
        bold=True,
        font_family=font_family,
    )
    stroke = tuple(int(v) for v in style.stroke_rgb)
    guide = tuple(int(v) for v in style.guide_rgb)
    label_rgb = tuple(int(v) for v in style.label_rgb)
    casing_fill = tuple(int(v) for v in style.muted_fill_rgb)

    body_bbox = bbox((cx - face_radius - 38.0, cy - scale_radius - 122.0, cx + face_radius + 38.0, cy + 142.0))
    draw.rounded_rectangle(tuple(body_bbox), radius=34, fill=casing_fill, outline=stroke, width=4)
    face_bbox = bbox((cx - face_radius, cy - scale_radius - 80.0, cx + face_radius, cy + 120.0))
    draw.rounded_rectangle(
        tuple(face_bbox),
        radius=34,
        fill=tuple(int(v) for v in style.panel_alt_fill_rgb),
        outline=stroke,
        width=4,
    )

    label_bboxes: List[List[float]] = []
    tick_bboxes: List[List[float]] = []
    for value in range(0, int(profile.scale_max) + 1, int(profile.minor_step)):
        angle = angle_for_value(float(value), scale_max=int(profile.scale_max))
        is_major = int(value) % int(profile.major_step) == 0
        outer = point_on_meter((cx, cy), scale_radius, angle)
        inner = point_on_meter((cx, cy), scale_radius - (30.0 if is_major else 18.0), angle)
        draw.line((inner[0], inner[1], outer[0], outer[1]), fill=stroke if is_major else guide, width=4 if is_major else 2)
        tick_bboxes.append(
            bbox(
                (
                    min(inner[0], outer[0]) - 3.0,
                    min(inner[1], outer[1]) - 3.0,
                    max(inner[0], outer[0]) + 3.0,
                    max(inner[1], outer[1]) + 3.0,
                )
            )
        )
        if is_major:
            label_center = point_on_meter((cx, cy), scale_radius - 62.0, angle)
            label_bbox = draw_centered_text(
                draw,
                text=str(int(value)),
                center=label_center,
                font=tick_font,
                fill=label_rgb,
                stroke_fill=resolve_text_stroke_fill(label_rgb),
                stroke_width=1,
            )
            label_bboxes.append(label_bbox)

    arc_bbox = bbox((cx - scale_radius, cy - scale_radius, cx + scale_radius, cy + scale_radius))
    arc_visual_bbox = bbox((cx - scale_radius - 6.0, cy - scale_radius - 6.0, cx + scale_radius + 6.0, cy - (scale_radius * 0.42) + 6.0))
    draw.arc(tuple(arc_bbox), start=210, end=330, fill=stroke, width=5)
    title_bbox = draw_centered_text(
        draw,
        text=profile.meter_name.title(),
        center=(cx, cy - scale_radius - 24.0),
        font=title_font,
        fill=label_rgb,
        stroke_fill=resolve_text_stroke_fill(label_rgb),
        stroke_width=1,
    )

    unit_box = bbox((cx - 58.0, cy + 36.0, cx + 58.0, cy + 88.0))
    draw.rounded_rectangle(
        tuple(unit_box),
        radius=13,
        fill=tuple(int(v) for v in style.label_fill_rgb),
        outline=tuple(int(v) for v in style.label_border_rgb),
        width=3,
    )
    unit_text_bbox = draw_centered_text(
        draw,
        text=str(profile.unit),
        center=(cx, cy + 62.0),
        font=unit_font,
        fill=label_rgb,
        stroke_fill=resolve_text_stroke_fill(label_rgb),
        stroke_width=1,
    )
    unit_bbox = bbox(_bbox_union(unit_box, unit_text_bbox))

    needle_angle = angle_for_value(float(scenario.readout_value), scale_max=int(profile.scale_max))
    needle_end = point_on_meter((cx, cy), needle_radius, needle_angle)
    tail = point_on_meter((cx, cy), -16.0, needle_angle)
    draw.line((tail[0], tail[1], needle_end[0], needle_end[1]), fill=accent_rgb, width=8)
    draw.ellipse((cx - 16.0, cy - 16.0, cx + 16.0, cy + 16.0), fill=accent_rgb, outline=stroke, width=3)
    needle_bbox = bbox(
        (
            min(tail[0], needle_end[0], cx - 18.0) - 8.0,
            min(tail[1], needle_end[1], cy - 18.0) - 8.0,
            max(tail[0], needle_end[0], cx + 18.0) + 8.0,
            max(tail[1], needle_end[1], cy + 18.0) + 8.0,
        )
    )

    scale_region = bbox(_bbox_union(arc_visual_bbox, title_bbox, *tick_bboxes, *label_bboxes, padding=6.0))
    annotation_map = normalize_bbox_map(
        {
            "needle": needle_bbox,
            "scale_region": scale_region,
            "unit_label": unit_bbox,
        }
    )
    render_map = {
        "meter_profile": str(profile.profile_id),
        "meter_name": str(profile.meter_name),
        "unit": str(profile.unit),
        "scale_min": 0,
        "scale_max": int(profile.scale_max),
        "major_step": int(profile.major_step),
        "minor_step": int(profile.minor_step),
        "readout_value": int(scenario.readout_value),
        "needle_angle_deg": round(math.degrees(float(needle_angle)), 3),
        "needle_tail_px": [round(float(tail[0]), 3), round(float(tail[1]), 3)],
        "needle_end_px": [round(float(needle_end[0]), 3), round(float(needle_end[1]), 3)],
        "needle_segment_px": [
            [round(float(tail[0]), 3), round(float(tail[1]), 3)],
            [round(float(needle_end[0]), 3), round(float(needle_end[1]), 3)],
        ],
        "face_bbox_px": face_bbox,
        "body_bbox_px": body_bbox,
    }
    return annotation_map, render_map


def render_analog_meter(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: MeterScenario,
    render_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> RenderedScene:
    """Render a complete analog meter scene and project its witness boxes."""

    canvas_width = int(render_defaults.get("canvas_width", DEFAULTS.canvas_width))
    canvas_height = int(render_defaults.get("canvas_height", DEFAULTS.canvas_height))
    background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        require_grid=True,
    )
    draw = ImageDraw.Draw(background)
    rng = spawn_rng(int(instance_seed), f"{namespace}.render")
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font",
        params=params,
    )
    panel = (
        float(render_defaults.get("panel_left_px", DEFAULTS.panel_left_px)),
        float(render_defaults.get("panel_top_px", DEFAULTS.panel_top_px)),
        float(canvas_width - int(render_defaults.get("panel_right_margin_px", DEFAULTS.panel_right_margin_px))),
        float(canvas_height - int(render_defaults.get("panel_bottom_margin_px", DEFAULTS.panel_bottom_margin_px))),
    )
    draw.rounded_rectangle(
        panel,
        radius=20,
        fill=tuple(int(v) for v in diagram_style.panel_fill_rgb),
        outline=tuple(int(v) for v in diagram_style.panel_border_rgb),
        width=3,
    )
    center = (
        float(render_defaults.get("meter_center_x_px", DEFAULTS.meter_center_x_px)) + float(rng.randint(-14, 14)),
        float(render_defaults.get("meter_center_y_px", DEFAULTS.meter_center_y_px)) + float(rng.randint(-8, 8)),
    )
    accent_palette = (
        (190, 45, 42),
        (36, 100, 190),
        (44, 143, 91),
        (173, 94, 35),
        (132, 78, 171),
    )
    accent_rgb = spawn_rng(int(instance_seed), f"{namespace}.needle_color").choice(accent_palette)
    annotation_map, render_map = draw_meter_face(
        draw,
        center=center,
        scenario=scenario,
        style=diagram_style,
        font_family=str(font_family),
        accent_rgb=accent_rgb,
        render_defaults=render_defaults,
    )
    image, post_noise_meta = apply_post_image_noise(
        background,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    render_map.update(
        {
            "center_px": [round(float(center[0]), 3), round(float(center[1]), 3)],
            "needle_rgb": list(int(v) for v in accent_rgb),
            "technical_diagram_style": dict(diagram_style_meta),
            "background_style": background_meta,
            "post_image_noise": post_noise_meta,
        }
    )
    scene_entities = [
        {"id": "needle", "bbox_px": list(annotation_map["needle"]), "readout_value": int(scenario.readout_value)},
        {"id": "scale_region", "bbox_px": list(annotation_map["scale_region"]), "scale_max": int(scenario.profile.scale_max)},
        {"id": "unit_label", "bbox_px": list(annotation_map["unit_label"]), "unit": str(scenario.profile.unit)},
    ]
    return RenderedScene(
        image=image,
        annotation_bbox_map={str(key): list(value) for key, value in annotation_map.items()},
        scene_entities=scene_entities,
        render_map=render_map,
        font_family=str(font_family),
    )


__all__ = [
    "angle_for_value",
    "draw_meter_face",
    "point_on_meter",
    "render_analog_meter",
]
