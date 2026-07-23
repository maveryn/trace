"""Shared multi-polygon scene helpers for geometry tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

from PIL import ImageDraw

from ...shared.geometry_primitives import Point
from ...shared.text_rendering import draw_text_centered, load_font, resolve_text_label_center
from .graph_rendering import scale_point
from .shape_style import GeometryShapeStyle


@dataclass(frozen=True)
class PolygonSceneObject:
    """One labeled polygon object rendered in a multi-object geometry scene."""

    label: str
    vertices: Tuple[Point, ...]
    center: Point


def draw_polygon_objects(
    draw: ImageDraw.ImageDraw,
    *,
    objects: Sequence[PolygonSceneObject],
    scene_scale: int,
    line_width: int,
    label_font_size_px: int,
    label_stroke_width: int,
    object_label_offset_px: float,
    render_canvas_size: int,
    shape_style: GeometryShapeStyle,
    draw_object_labels: bool = True,
) -> Dict[str, List[float]]:
    """Draw polygon outlines plus object labels and return label centers."""

    scaled_objects = [
        {
            "label": str(obj.label),
            "vertices": [scale_point(point, int(scene_scale)) for point in obj.vertices],
            "center": scale_point(obj.center, int(scene_scale)),
        }
        for obj in objects
    ]
    blocked_segments: List[Tuple[Point, Point]] = []
    line_color = tuple(int(value) for value in shape_style.line_color)
    for scaled in scaled_objects:
        polygon = list(scaled["vertices"])
        draw.line([*polygon, polygon[0]], fill=line_color, width=max(1, int(line_width)), joint="curve")
        for index in range(len(polygon)):
            point_a = polygon[index]
            point_b = polygon[(index + 1) % len(polygon)]
            blocked_segments.append(
                ((float(point_a[0]), float(point_a[1])), (float(point_b[0]), float(point_b[1])))
            )

    if not bool(draw_object_labels):
        return {}

    font = load_font(int(label_font_size_px), bold=True)
    occupied_boxes: List[Tuple[float, float, float, float]] = []
    label_centers: Dict[str, List[float]] = {}
    for scaled in scaled_objects:
        polygon = list(scaled["vertices"])
        min_x = min(float(point[0]) for point in polygon)
        max_x = max(float(point[0]) for point in polygon)
        min_y = min(float(point[1]) for point in polygon)
        max_y = max(float(point[1]) for point in polygon)
        center = (float(scaled["center"][0]), float(scaled["center"][1]))
        direction = (
            float(center[0]) - (0.5 * float(render_canvas_size)),
            float(center[1]) - (0.5 * float(render_canvas_size)),
        )
        width_px = float(max_x - min_x)
        height_px = float(max_y - min_y)
        outward_offset = max(
            float(object_label_offset_px) * float(scene_scale),
            0.55 * max(float(width_px), float(height_px)),
        )
        label_center, label_bbox = resolve_text_label_center(
            draw,
            text=str(scaled["label"]),
            anchor=center,
            base_direction=direction,
            offset_px=float(outward_offset),
            font=font,
            blocked_segments=blocked_segments,
            occupied_boxes=occupied_boxes,
            stroke_width=int(label_stroke_width),
            line_clearance_px=max(6.0, 1.2 * float(max(1, int(line_width)))),
            canvas_size=int(render_canvas_size),
        )
        draw_text_centered(
            draw,
            text=str(scaled["label"]),
            center=(float(label_center[0]), float(label_center[1])),
            font=font,
            fill=tuple(int(value) for value in shape_style.label_color),
            stroke_fill=tuple(int(value) for value in shape_style.label_stroke_color),
            stroke_width=int(label_stroke_width),
        )
        occupied_boxes.append(label_bbox)
        label_centers[str(scaled["label"])] = [
            float(label_center[0]) / float(max(1, int(scene_scale))),
            float(label_center[1]) / float(max(1, int(scene_scale))),
        ]
    return label_centers


__all__ = [
    "PolygonSceneObject",
    "draw_polygon_objects",
]
