"""Rendering primitives for orbital-motion diagrams."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.drawing import draw_centered_text
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .sampling import rotated_point
from .state import SCENE_ID, SCENE_NAMESPACE, OrbitRenderDefaults, OrbitSpec, RenderedOrbitScene


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def bbox_from_center(center: Tuple[float, float], half_w: float, half_h: float) -> List[float]:
    """Return a JSON-stable bbox centered on a point."""

    return [
        round(float(center[0] - half_w), 3),
        round(float(center[1] - half_h), 3),
        round(float(center[0] + half_w), 3),
        round(float(center[1] + half_h), 3),
    ]


def _bbox_overlaps(a: List[float], b: List[float], *, pad: float = 5.0) -> bool:
    """Return whether two bboxes overlap after applying a small visual margin."""

    return not (
        float(a[2]) + float(pad) <= float(b[0])
        or float(b[2]) + float(pad) <= float(a[0])
        or float(a[3]) + float(pad) <= float(b[1])
        or float(b[3]) + float(pad) <= float(a[1])
    )


def _bbox_inside(bounds: List[float], bbox: List[float]) -> bool:
    """Return whether bbox stays inside the usable panel bounds."""

    return (
        float(bounds[0]) <= float(bbox[0])
        and float(bbox[2]) <= float(bounds[2])
        and float(bounds[1]) <= float(bbox[1])
        and float(bbox[3]) <= float(bounds[3])
    )


def _candidate_label_position(
    point: Tuple[float, float],
    *,
    panel: List[float],
    placed_bboxes: List[List[float]],
) -> tuple[Tuple[float, float], List[float]]:
    """Choose a nearby label badge position that avoids previous label badges."""

    x, y = point
    offsets = (
        (31.0, -30.0),
        (-31.0, -30.0),
        (31.0, 30.0),
        (-31.0, 30.0),
        (0.0, -46.0),
        (0.0, 46.0),
        (48.0, 0.0),
        (-48.0, 0.0),
        (54.0, -44.0),
        (-54.0, -44.0),
        (54.0, 44.0),
        (-54.0, 44.0),
    )
    fallback_center = (float(x + offsets[0][0]), float(y + offsets[0][1]))
    fallback_bbox = bbox_from_center(fallback_center, 19, 18)
    for dx, dy in offsets:
        label_center = (float(x + dx), float(y + dy))
        label_bbox = bbox_from_center(label_center, 19, 18)
        if not _bbox_inside(panel, label_bbox):
            continue
        if any(_bbox_overlaps(label_bbox, placed) for placed in placed_bboxes):
            continue
        return label_center, label_bbox
    return fallback_center, fallback_bbox


def _draw_orbit_scene(
    *,
    image: Image.Image,
    spec: OrbitSpec,
    font_family: str,
    style: Any,
) -> tuple[Image.Image, Dict[str, Any], List[Dict[str, Any]]]:
    """Draw the ellipse, candidate markers, and optional Sun in one fixed coordinate frame."""

    draw = ImageDraw.Draw(image)
    label_font = load_font(30, bold=True, font_family=font_family)
    small_font = load_font(23, bold=True, font_family=font_family)
    title_font = load_font(26, bold=True, font_family=font_family)
    stroke = tuple(int(v) for v in style.stroke_rgb)
    muted = tuple(int(v) for v in style.secondary_stroke_rgb)
    guide = tuple(int(v) for v in style.guide_rgb)
    accent = tuple(int(v) for v in style.accent_rgb)
    label_fill = tuple(int(v) for v in style.label_fill_rgb)
    label_outline = tuple(int(v) for v in style.label_border_rgb)
    label_text = tuple(int(v) for v in style.label_rgb)

    panel = [54, 52, image.size[0] - 54, image.size[1] - 58]
    draw.rounded_rectangle(
        panel,
        radius=18,
        fill=tuple(style.panel_fill_rgb),
        outline=tuple(style.panel_border_rgb),
        width=3,
    )

    points = [
        rotated_point(
            spec.center,
            spec.semi_major * math.cos(t),
            spec.semi_minor * math.sin(t),
            spec.rotation_rad,
        )
        for t in [i * 2 * math.pi / 240 for i in range(241)]
    ]
    draw.line(points, fill=stroke, width=5, joint="curve")
    draw.line([spec.major_axis_endpoints[0], spec.major_axis_endpoints[1]], fill=guide, width=2)
    minor_1 = rotated_point(spec.center, 0.0, spec.semi_minor, spec.rotation_rad)
    minor_2 = rotated_point(spec.center, 0.0, -spec.semi_minor, spec.rotation_rad)
    draw.line([minor_1, minor_2], fill=guide, width=2)
    draw.ellipse(tuple(bbox_from_center(spec.center, 6, 6)), fill=muted, outline=stroke, width=2)
    draw_centered_text(
        draw,
        text="center",
        center=(spec.center[0], spec.center[1] + 28),
        font=small_font,
        fill=muted,
        stroke_fill=resolve_text_stroke_fill(muted),
        stroke_width=1,
    )

    entities: List[Dict[str, Any]] = [
        {
            "id": "ellipse_center",
            "label": "center",
            "center": [round(float(spec.center[0]), 3), round(float(spec.center[1]), 3)],
        }
    ]
    if spec.sun_point is not None:
        sx, sy = spec.sun_point
        draw.ellipse(tuple(bbox_from_center((sx, sy), 20, 20)), fill=(255, 205, 68), outline=(164, 96, 22), width=3)
        draw_centered_text(
            draw,
            text="Sun",
            center=(sx, sy + 42),
            font=small_font,
            fill=(120, 72, 20),
            stroke_fill=(255, 248, 210),
            stroke_width=2,
        )
        entities.append({"id": "sun", "label": "Sun", "center": [round(float(sx), 3), round(float(sy), 3)]})

    candidate_bboxes: Dict[str, List[float]] = {}
    candidate_label_bboxes: Dict[str, List[float]] = {}
    placed_label_bboxes: List[List[float]] = []
    for label, point in sorted(spec.candidate_points.items()):
        x, y = point
        draw.ellipse(tuple(bbox_from_center((x, y), 13, 13)), fill=accent, outline=stroke, width=3)
        label_center, label_bbox = _candidate_label_position(
            (float(x), float(y)),
            panel=panel,
            placed_bboxes=placed_label_bboxes,
        )
        draw.rounded_rectangle(tuple(label_bbox), radius=8, fill=label_fill, outline=label_outline, width=2)
        draw_centered_text(
            draw,
            text=str(label),
            center=label_center,
            font=label_font,
            fill=label_text,
            stroke_fill=resolve_text_stroke_fill(label_text),
            stroke_width=1,
        )
        candidate_bboxes[str(label)] = bbox_from_center((x, y), 18, 18)
        candidate_label_bboxes[str(label)] = list(label_bbox)
        placed_label_bboxes.append(list(label_bbox))
        entities.append(
            {
                "id": f"candidate_{label}",
                "label": str(label),
                "center": [round(float(x), 3), round(float(y), 3)],
            }
        )

    draw_centered_text(
        draw,
        text="elliptical orbit",
        center=(image.size[0] * 0.5, panel[1] + 28),
        font=title_font,
        fill=tuple(style.label_rgb),
        stroke_fill=resolve_text_stroke_fill(tuple(style.label_rgb)),
        stroke_width=1,
    )
    render_map = {
        "candidate_bboxes": candidate_bboxes,
        "candidate_label_bboxes": candidate_label_bboxes,
        "candidate_points": {
            str(k): [round(float(v[0]), 3), round(float(v[1]), 3)]
            for k, v in spec.candidate_points.items()
        },
        "selected_label": str(spec.selected_label),
        "selected_point": [round(float(spec.selected_point[0]), 3), round(float(spec.selected_point[1]), 3)],
        "center": [round(float(spec.center[0]), 3), round(float(spec.center[1]), 3)],
        "major_axis_endpoints": [
            [round(float(point[0]), 3), round(float(point[1]), 3)]
            for point in spec.major_axis_endpoints
        ],
        "sun_point": (
            [round(float(spec.sun_point[0]), 3), round(float(spec.sun_point[1]), 3)]
            if spec.sun_point is not None
            else None
        ),
    }
    return image, render_map, entities


def render_orbit_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    spec: OrbitSpec,
    render_defaults: Mapping[str, Any],
    fallback: OrbitRenderDefaults = OrbitRenderDefaults(),
    namespace: str = SCENE_NAMESPACE,
) -> RenderedOrbitScene:
    """Render a complete orbital-motion diagram and return projected metadata."""

    canvas_width = int(render_defaults.get("canvas_width", fallback.canvas_width))
    canvas_height = int(render_defaults.get("canvas_height", fallback.canvas_height))
    background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        require_grid=True,
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font",
        params=params,
    )
    image, render_map, entities = _draw_orbit_scene(
        image=background,
        spec=spec,
        font_family=str(font_family),
        style=diagram_style,
    )
    image, post_noise_meta = apply_post_image_noise(
        image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    render_map.update(
        {
            "technical_diagram_style": dict(diagram_style_meta),
            "background_style": dict(background_meta),
            "post_image_noise": dict(post_noise_meta),
        }
    )
    return RenderedOrbitScene(
        image=image,
        scene_entities=entities,
        render_map=render_map,
        font_family=str(font_family),
    )


__all__ = ["bbox_from_center", "render_orbit_scene"]
