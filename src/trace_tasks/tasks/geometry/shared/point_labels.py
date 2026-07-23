"""Shared point-label rendering helper with overlap-aware placement."""

from __future__ import annotations

from typing import List, Sequence, Tuple

from PIL import ImageDraw

from ...shared.text_rendering import draw_text_centered, load_font, resolve_text_label_center

Point = Tuple[float, float]
Segment = Tuple[Point, Point]


def draw_labeled_points(
    draw: ImageDraw.ImageDraw,
    *,
    points: Sequence[Point],
    labels: Sequence[str],
    label_offset_px: float,
    font_size_px: int,
    text_stroke_width: int | None = None,
    blocked_segments: Sequence[Segment] | None = None,
    blocked_points: Sequence[Point] | None = None,
    blocked_point_clearance_px: float | None = None,
    marker_radius_px: int = 0,
    marker_color: Tuple[int, int, int] = (22, 22, 22),
    label_color: Tuple[int, int, int] = (24, 24, 24),
    label_stroke_color: Tuple[int, int, int] = (252, 252, 252),
    canvas_size: int | None = None,
) -> None:
    """Draw labels for one set of points using overlap-aware placement.

    The helper keeps labels off both the labeled points themselves and any
    additional blocked points so point names stay legible without covering
    nearby markers in dense geometry scenes. The blocked-point clearance grows
    with the rendered font size so supersampled scenes do not let large labels
    drift back onto their anchor markers.
    """
    if not points or len(points) != len(labels):
        return
    centroid_x = float(sum(float(point[0]) for point in points) / float(len(points)))
    centroid_y = float(sum(float(point[1]) for point in points) / float(len(points)))
    offset = float(max(10.0, float(label_offset_px), 0.35 * float(max(8, int(font_size_px)))))
    font = load_font(int(font_size_px), bold=True)
    stroke_width = (
        int(text_stroke_width)
        if text_stroke_width is not None
        else max(1, int(round(0.04 * float(max(8, int(font_size_px))))))
    )
    segments = [
        (
            (float(seg_a[0]), float(seg_a[1])),
            (float(seg_b[0]), float(seg_b[1])),
        )
        for seg_a, seg_b in list(blocked_segments or ())
    ]
    blocked_point_list = [
        *((float(point[0]), float(point[1])) for point in points),
        *((float(point[0]), float(point[1])) for point in list(blocked_points or ())),
    ]
    occupied_boxes: List[Tuple[float, float, float, float]] = []
    radius = max(0, int(marker_radius_px))
    point_clearance = (
        float(blocked_point_clearance_px)
        if blocked_point_clearance_px is not None
        else float(max(6, radius + 3, int(round(0.30 * float(max(8, int(font_size_px)))))))
    )
    for label, point in zip(labels, points):
        px, py = float(point[0]), float(point[1])
        if radius > 0:
            draw.ellipse(
                [float(px - radius), float(py - radius), float(px + radius), float(py + radius)],
                fill=tuple(int(value) for value in marker_color),
            )
        dx, dy = float(px - centroid_x), float(py - centroid_y)
        center, label_bbox = resolve_text_label_center(
            draw,
            text=str(label),
            anchor=(float(px), float(py)),
            base_direction=(float(dx), float(dy)),
            offset_px=float(offset),
            font=font,
            blocked_segments=segments,
            blocked_points=blocked_point_list,
            occupied_boxes=occupied_boxes,
            stroke_width=int(stroke_width),
            line_clearance_px=3.0,
            point_clearance_px=float(point_clearance),
            canvas_size=(int(canvas_size) if canvas_size is not None else None),
        )
        draw_text_centered(
            draw,
            text=str(label),
            center=(float(center[0]), float(center[1])),
            font=font,
            fill=tuple(int(value) for value in label_color),
            stroke_fill=tuple(int(value) for value in label_stroke_color),
            stroke_width=int(stroke_width),
        )
        occupied_boxes.append(label_bbox)
