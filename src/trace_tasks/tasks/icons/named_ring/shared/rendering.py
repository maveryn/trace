"""Rendering helpers for named-ring icon scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.text_rendering import draw_text_centered, load_font
from ...shared.icon_noise import serialize_icon_noise_edits
from ...shared.icon_scene import BBox, draw_single_panel, resolve_single_panel_layout, single_panel_geometry_to_trace
from ...shared.icon_style import sample_icon_palette
from ...shared.icon_task_rendering import sample_icon_instance_noise
from ...shared.procedural_named_icon_field_scene import bbox_center_float, bbox_from_center_and_size, rotation_for_named_shape
from ...shared.procedural_named_icons import (
    procedural_named_icon_display_name,
    render_procedural_named_icon_rgba,
    sample_procedural_named_icon_fill_style,
)

from .state import RenderedRingIcon, RingArcPlan, RingScenePayload


def _icon_bbox(center_xy: Sequence[float], sprite_size: Sequence[int]) -> Tuple[int, int, int, int]:
    return bbox_from_center_and_size(center_xy, sprite_size)


def _marker_label_bbox(
    *,
    icon_center_xy: Tuple[float, float],
    ring_center_xy: Tuple[float, float],
    content_bbox: BBox,
    label_radius_px: int,
    gap_px: int,
    icon_size_px: int,
) -> Tuple[int, int, int, int]:
    dx = float(icon_center_xy[0]) - float(ring_center_xy[0])
    dy = float(icon_center_xy[1]) - float(ring_center_xy[1])
    norm = max(1e-6, math.hypot(dx, dy))
    offset = (0.5 * float(icon_size_px)) + float(gap_px) + float(label_radius_px)
    cx = float(icon_center_xy[0]) + offset * dx / norm
    cy = float(icon_center_xy[1]) + offset * dy / norm
    radius = int(label_radius_px)
    x0 = int(round(cx - radius))
    y0 = int(round(cy - radius))
    x0 = max(int(content_bbox[0]), min(int(content_bbox[2]) - 2 * radius, x0))
    y0 = max(int(content_bbox[1]), min(int(content_bbox[3]) - 2 * radius, y0))
    return (int(x0), int(y0), int(x0 + 2 * radius), int(y0 + 2 * radius))


def _draw_marker_labels(
    *,
    image: Image.Image,
    icons: Sequence[RenderedRingIcon],
    marker_indices: Mapping[str, int],
    ring_center_xy: Tuple[float, float],
    content_bbox: BBox,
    render_params: Mapping[str, Any],
) -> Dict[str, Tuple[int, int, int, int]]:
    """Draw endpoint marker labels after icons and return their boxes."""

    draw = ImageDraw.Draw(image)
    label_font = load_font(int(render_params["marker_label_font_size_px"]), bold=True)
    label_bg = tuple(int(value) for value in render_params["marker_label_background_rgb"])
    label_border = tuple(int(value) for value in render_params["marker_label_border_rgb"])
    label_color = tuple(int(value) for value in render_params["marker_label_color_rgb"])
    label_stroke = tuple(int(value) for value in render_params.get("marker_label_stroke_rgb", label_bg))
    marker_label_bboxes: Dict[str, Tuple[int, int, int, int]] = {}
    for marker_label in ("A", "B"):
        marker_icon = icons[int(marker_indices[str(marker_label)])]
        label_bbox = _marker_label_bbox(
            icon_center_xy=marker_icon.center_xy,
            ring_center_xy=ring_center_xy,
            content_bbox=content_bbox,
            label_radius_px=int(render_params["marker_label_radius_px"]),
            gap_px=int(render_params["marker_label_gap_px"]),
            icon_size_px=int(marker_icon.nominal_size_px),
        )
        draw.rounded_rectangle(
            label_bbox,
            radius=int(render_params["marker_label_radius_px"]),
            fill=label_bg + (245,),
            outline=label_border + (255,),
            width=2,
        )
        draw_text_centered(
            draw,
            text=str(marker_label),
            center=bbox_center_float(label_bbox),
            font=label_font,
            fill=label_color,
            stroke_fill=label_stroke,
            stroke_width=1,
        )
        marker_label_bboxes[str(marker_label)] = tuple(int(value) for value in label_bbox)
    return marker_label_bboxes


def _ring_geometry(*, content_bbox: BBox, render_params: Mapping[str, Any]) -> Tuple[Tuple[int, int, int, int], Tuple[float, float], Tuple[float, float]]:
    """Compute the ellipse geometry while reserving marker-label clearance."""

    min_icon_size = max(16, int(render_params["scene_icon_size_min_px"]))
    max_icon_size = max(min_icon_size, int(render_params["scene_icon_size_max_px"]))
    marker_radius = int(render_params["marker_label_radius_px"])
    marker_gap = int(render_params["marker_label_gap_px"])
    safe_margin = max(
        int(render_params["ring_margin_px"]),
        int(0.5 * max_icon_size) + marker_radius + marker_gap + 8,
    )
    cx = 0.5 * float(content_bbox[0] + content_bbox[2])
    cy = 0.5 * float(content_bbox[1] + content_bbox[3])
    rx = 0.5 * float(content_bbox[2] - content_bbox[0]) - float(safe_margin)
    ry = 0.5 * float(content_bbox[3] - content_bbox[1]) - float(safe_margin)
    if rx < 120.0 or ry < 90.0:
        raise ValueError("named-ring content area is too small")
    ring_bbox = (int(round(cx - rx)), int(round(cy - ry)), int(round(cx + rx)), int(round(cy + ry)))
    return ring_bbox, (float(cx), float(cy)), (float(rx), float(ry))


def _draw_ring_scaffold(
    *,
    image: Image.Image,
    centers: Sequence[Tuple[float, float]],
    ring_bbox: Tuple[int, int, int, int],
    render_params: Mapping[str, Any],
) -> None:
    """Draw the ring outline and stop markers before icon sprites are added."""

    draw = ImageDraw.Draw(image)
    ring_outline = tuple(int(value) for value in render_params["ring_outline_rgb"])
    stop_fill = tuple(int(value) for value in render_params["ring_stop_fill_rgb"])
    stop_outline = tuple(int(value) for value in render_params["ring_stop_outline_rgb"])
    draw.ellipse(
        ring_bbox,
        outline=ring_outline + (210,),
        width=max(1, int(render_params["ring_stroke_width_px"])),
    )
    stop_radius = int(render_params["ring_stop_radius_px"])
    for center in centers:
        draw.ellipse(
            (
                float(center[0] - stop_radius),
                float(center[1] - stop_radius),
                float(center[0] + stop_radius),
                float(center[1] + stop_radius),
            ),
            fill=stop_fill + (230,),
            outline=stop_outline + (240,),
            width=1,
        )


def _ring_centers(
    *,
    rng,
    center_xy: Tuple[float, float],
    radius_xy: Tuple[float, float],
    count: int,
) -> Tuple[Tuple[float, float], ...]:
    start_angle = float(rng.uniform(-18.0, 18.0))
    centers: List[Tuple[float, float]] = []
    for index in range(int(count)):
        angle = math.radians(float(start_angle) + (360.0 * float(index) / float(count)))
        centers.append(
            (
                float(center_xy[0] + radius_xy[0] * math.cos(angle)),
                float(center_xy[1] + radius_xy[1] * math.sin(angle)),
            )
        )
    return tuple(centers)


def render_named_ring_scene(
    *,
    rng,
    plan: RingArcPlan,
    instance_seed: int,
    render_params: Mapping[str, Any],
    noise_namespace: str,
) -> RingScenePayload:
    """Render a complete named-ring scene from a symbolic arc plan."""

    layout = resolve_single_panel_layout(
        canvas_width=int(render_params["canvas_width"]),
        canvas_height=int(render_params["canvas_height"]),
        outer_margin_px=int(render_params["outer_margin_px"]),
        panel_padding_px=int(render_params["panel_padding_px"]),
        title_font_size_px=int(render_params["panel_title_font_size_px"]),
        reserve_title=False,
    )
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

    content_bbox = tuple(int(value) for value in layout.scene_content_xyxy)
    ring_bbox, ring_center_xy, ring_radius_xy = _ring_geometry(content_bbox=content_bbox, render_params=render_params)
    centers = _ring_centers(
        rng=rng,
        center_xy=ring_center_xy,
        radius_xy=ring_radius_xy,
        count=int(plan.ring_icon_count),
    )
    _draw_ring_scaffold(image=image, centers=centers, ring_bbox=ring_bbox, render_params=render_params)

    palette_size = int(rng.randint(int(render_params["palette_size_min"]), int(render_params["palette_size_max"])))
    palette = sample_icon_palette(
        rng,
        palette_size=int(palette_size),
        channel_min=int(render_params["color_channel_min"]),
        channel_max=int(render_params["color_channel_max"]),
        anchor_colors=(
            tuple(int(value) for value in render_params["background_color_rgb"]),
            tuple(int(value) for value in render_params["panel_fill_rgb"]),
            tuple(int(value) for value in render_params["panel_border_rgb"]),
            tuple(int(value) for value in render_params["header_text_rgb"]),
        ),
        min_color_distance=float(render_params["min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
    )
    min_icon_size = max(16, int(render_params["scene_icon_size_min_px"]))
    max_icon_size = max(min_icon_size, int(render_params["scene_icon_size_max_px"]))
    counted_set = set(int(value) for value in plan.counted_indices)
    arc_set = set(int(value) for value in plan.arc_indices)
    icons: List[RenderedRingIcon] = []
    for index, center in enumerate(centers):
        shape_id = str(plan.shape_ids_by_index[int(index)])
        fill_style = sample_procedural_named_icon_fill_style(
            rng,
            support=plan.fill_style_support,
            probabilities=plan.fill_style_probabilities,
        )
        tint_rgb = tuple(int(value) for value in rng.choice(palette))
        nominal_size_px = int(rng.randint(int(min_icon_size), int(max_icon_size)))
        noise_edits, noise_seed = sample_icon_instance_noise(
            instance_seed=int(instance_seed),
            namespace=f"{noise_namespace}.{int(index)}",
            render_params=render_params,
        )
        rotation_degrees = rotation_for_named_shape(rng, str(shape_id))
        sprite = render_procedural_named_icon_rgba(
            shape_id=str(shape_id),
            size_px=int(nominal_size_px),
            tint_rgb=tuple(int(value) for value in tint_rgb),
            fill_style=str(fill_style),
            rotation_degrees=int(rotation_degrees),
            noise_edits=tuple(noise_edits),
            noise_seed=int(noise_seed),
        )
        bbox = _icon_bbox(center, sprite.size)
        image.alpha_composite(sprite, (int(bbox[0]), int(bbox[1])))
        marker_label = "A" if int(index) == int(plan.start_index) else "B" if int(index) == int(plan.end_index) else ""
        role = (
            "start_marker"
            if marker_label == "A"
            else "end_marker"
            if marker_label == "B"
            else "arc_icon"
            if int(index) in arc_set
            else "outside_arc_icon"
        )
        icons.append(
            RenderedRingIcon(
                instance_id=f"ring_icon_{int(index):02d}",
                ring_index=int(index),
                clockwise_position_number=int(index) + 1,
                role=str(role),
                marker_label=str(marker_label),
                shape_id=str(shape_id),
                shape_name=procedural_named_icon_display_name(str(shape_id)),
                bbox_xyxy=tuple(int(value) for value in bbox),
                center_xy=(float(center[0]), float(center[1])),
                nominal_size_px=int(nominal_size_px),
                rotation_degrees=int(rotation_degrees),
                tint_rgb=tuple(int(value) for value in tint_rgb),
                fill_style=str(fill_style),
                noise_edits=tuple(serialize_icon_noise_edits(noise_edits)),
                noise_seed=int(noise_seed),
                is_target_shape=str(shape_id) == str(plan.target_shape_id),
                is_arc_member=int(index) in arc_set,
                is_counted=int(index) in counted_set,
            )
        )

    marker_label_bboxes = _draw_marker_labels(
        image=image,
        icons=tuple(icons),
        marker_indices={"A": int(plan.start_index), "B": int(plan.end_index)},
        ring_center_xy=ring_center_xy,
        content_bbox=content_bbox,
        render_params=render_params,
    )
    return RingScenePayload(
        image=image.convert("RGB"),
        panel_geometry=single_panel_geometry_to_trace(layout),
        ring_bbox_xyxy=tuple(int(value) for value in ring_bbox),
        ring_center_xy=(float(ring_center_xy[0]), float(ring_center_xy[1])),
        ring_radius_xy=(float(ring_radius_xy[0]), float(ring_radius_xy[1])),
        icons=tuple(icons),
        marker_label_bboxes={str(key): tuple(int(value) for value in bbox) for key, bbox in marker_label_bboxes.items()},
        sampled_palette_rgb=tuple(tuple(int(channel) for channel in color) for color in palette),
    )


def serialize_ring_icon(icon: RenderedRingIcon) -> Dict[str, Any]:
    """Return JSON-stable metadata for one rendered ring icon."""

    return {
        "entity_kind": "named_icon",
        "instance_id": str(icon.instance_id),
        "ring_index": int(icon.ring_index),
        "clockwise_position_number": int(icon.clockwise_position_number),
        "role": str(icon.role),
        "marker_label": str(icon.marker_label),
        "shape_id": str(icon.shape_id),
        "shape_name": str(icon.shape_name),
        "bbox_xyxy": [int(value) for value in icon.bbox_xyxy],
        "center_xy": [float(icon.center_xy[0]), float(icon.center_xy[1])],
        "nominal_size_px": int(icon.nominal_size_px),
        "rotation_degrees": int(icon.rotation_degrees),
        "tint_rgb": [int(value) for value in icon.tint_rgb],
        "fill_style": str(icon.fill_style),
        "noise_edits": [dict(value) for value in icon.noise_edits],
        "noise_seed": None if icon.noise_seed is None else int(icon.noise_seed),
        "is_target_shape": bool(icon.is_target_shape),
        "is_arc_member": bool(icon.is_arc_member),
        "is_counted": bool(icon.is_counted),
    }
