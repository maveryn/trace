"""Rendering helpers for marked-point overlays in 3D spatial tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ....shared.text_legibility import draw_text_traced, text_legibility_metadata_for_surfaces
from ....shared.text_rendering import load_font
from ...shared.object_scene import POINT_COLORS, POINT_LABELS, _RenderParams


def _wrap_color_index(index: int) -> int:
    color_count = len(POINT_COLORS)
    if color_count <= 0:
        return 0
    resolved = abs(int(index))
    while resolved >= color_count:
        resolved -= color_count
    return int(resolved)


def _marker_color_for_label(label: str) -> Tuple[int, int, int]:
    """Return a deterministic marker color for standard or task-local point labels."""

    if str(label) in POINT_LABELS:
        color_index = POINT_LABELS.index(str(label))
    else:
        color_index = sum(ord(char) for char in str(label))
    return tuple(int(channel) for channel in POINT_COLORS[_wrap_color_index(int(color_index))])


def _text_bbox_at_xy(
    *,
    draw: ImageDraw.ImageDraw,
    text: str,
    xy: Sequence[float],
    font,
    stroke_width: int,
) -> List[float]:
    bbox = draw.textbbox((float(xy[0]), float(xy[1])), str(text), font=font, stroke_width=int(stroke_width))
    return [round(float(value), 3) for value in bbox]


def _bbox_union(*bboxes: Sequence[float]) -> List[float]:
    clean = [list(bbox) for bbox in bboxes if bbox]
    return [
        round(float(min(bbox[0] for bbox in clean)), 3),
        round(float(min(bbox[1] for bbox in clean)), 3),
        round(float(max(bbox[2] for bbox in clean)), 3),
        round(float(max(bbox[3] for bbox in clean)), 3),
    ]


def draw_marked_points(
    image: Image.Image,
    *,
    marked_points: Sequence[Mapping[str, Any]],
    render_params: _RenderParams,
) -> Tuple[Image.Image, Dict[str, Any], List[Dict[str, Any]]]:
    """Draw marked point glyphs and labels; marker centers are scalar point witnesses."""
    output = image.convert("RGB")
    draw = ImageDraw.Draw(output)
    marker_radius = max(9.0, float(render_params.marker_radius_px) * 0.56)
    center_radius = max(3.4, marker_radius * 0.26)
    label_font = load_font(max(int(render_params.label_font_size_px) + 2, int(round(marker_radius * 1.45))), bold=True)
    marker_centers: Dict[str, List[float]] = {}
    marker_glyph_bboxes: Dict[str, List[float]] = {}
    marker_label_bboxes: Dict[str, List[float]] = {}
    marker_bboxes: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = []

    for point in sorted(marked_points, key=lambda item: float(item["camera_distance"]), reverse=True):
        label = str(point["point_label"])
        x, y = (float(point["screen_xy"][0]), float(point["screen_xy"][1]))
        color = _marker_color_for_label(label)
        glyph_bbox = [
            round(float(x - marker_radius), 3),
            round(float(y - marker_radius), 3),
            round(float(x + marker_radius), 3),
            round(float(y + marker_radius), 3),
        ]
        draw.ellipse(tuple(glyph_bbox), fill=(255, 255, 255), outline=(22, 28, 36), width=4)
        inner_bbox = [
            float(x - center_radius),
            float(y - center_radius),
            float(x + center_radius),
            float(y + center_radius),
        ]
        draw.ellipse(tuple(inner_bbox), fill=color, outline=(22, 28, 36), width=1)
        label_fill_rgb = (255, 255, 255)
        label_backing_rgb = (23, 28, 36)
        raw_label_bbox = draw.textbbox((0.0, 0.0), label, font=label_font, stroke_width=0)
        label_width = float(raw_label_bbox[2] - raw_label_bbox[0])
        label_height = float(raw_label_bbox[3] - raw_label_bbox[1])
        label_x = x + marker_radius + 5.0
        if label_x + label_width + 6.0 > float(render_params.canvas_width):
            label_x = x - marker_radius - label_width - 7.0
        label_y = y - label_height * 0.56
        label_y = max(4.0, min(float(render_params.canvas_height) - label_height - 5.0, label_y))
        text_xy = (round(float(label_x), 3), round(float(label_y), 3))
        label_bbox = _text_bbox_at_xy(
            draw=draw,
            text=label,
            xy=text_xy,
            font=label_font,
            stroke_width=0,
        )
        backing_bbox = [
            max(0.0, float(label_bbox[0]) - 4.0),
            max(0.0, float(label_bbox[1]) - 3.0),
            min(float(render_params.canvas_width), float(label_bbox[2]) + 4.0),
            min(float(render_params.canvas_height), float(label_bbox[3]) + 3.0),
        ]
        draw.rounded_rectangle(tuple(backing_bbox), radius=5, fill=label_backing_rgb, outline=(255, 255, 255), width=1)
        draw_text_traced(
            draw,
            text_xy,
            label,
            font=label_font,
            fill=label_fill_rgb,
            stroke_width=0,
            stroke_fill=label_backing_rgb,
            role="marked_point_label_halo",
            required=False,
        )
        draw_text_traced(
            draw,
            text_xy,
            label,
            font=label_font,
            fill=label_fill_rgb,
            stroke_width=0,
            stroke_fill=label_backing_rgb,
            role="marked_point_label",
            required=True,
            extra_metadata=text_legibility_metadata_for_surfaces(
                fill_rgb=label_fill_rgb,
                surface_rgbs=(label_backing_rgb,),
            ),
        )
        combined_bbox = _bbox_union(glyph_bbox, label_bbox)
        marker_centers[label] = [round(float(x), 3), round(float(y), 3)]
        marker_glyph_bboxes[label] = list(glyph_bbox)
        marker_label_bboxes[label] = list(label_bbox)
        marker_bboxes[label] = list(combined_bbox)
        entities.append(
            {
                "entity_id": str(point["point_id"]),
                "entity_type": "three_d_marked_point",
                "bbox_px": list(combined_bbox),
                "attrs": {
                    "point_label": str(label),
                    "marker_id": str(point["marker_id"]),
                    "surface_kind": str(point["surface_kind"]),
                    "attached_object_id": point.get("attached_object_id"),
                    "is_answer_candidate": True,
                    "world_xyz": list(point["world_xyz"]),
                    "screen_xy": [round(float(x), 3), round(float(y), 3)],
                    "camera_xyz": list(point["camera_xyz"]),
                    "camera_distance": float(point["camera_distance"]),
                    "marker_color_rgb": [int(channel) for channel in color],
                    "marker_style": "ring_dot_with_offset_label",
                },
            }
        )

    return (
        output,
        {
            "marked_point_centers_px": dict(marker_centers),
            "marked_point_glyph_bboxes_px": dict(marker_glyph_bboxes),
            "marked_point_circle_bboxes_px": dict(marker_glyph_bboxes),
            "marked_point_label_bboxes_px": dict(marker_label_bboxes),
            "marked_point_bboxes_px": dict(marker_bboxes),
        },
        entities,
    )


__all__ = ["draw_marked_points"]
