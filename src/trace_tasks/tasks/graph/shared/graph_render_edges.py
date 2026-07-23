"""Edge routing, drawing, and label placement for node-link graphs."""

from __future__ import annotations

import math
from typing import Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from ...shared.text_legibility import draw_centered_readable_text, resolve_readable_text_style
from ...shared.text_rendering import load_font
from .graph_render_geometry import _bbox_intersects_bbox, _segment_intersects_bbox
from .graph_render_types import BBox, Point


def _quadratic_bezier_point(start: Point, control: Point, end: Point, t_value: float) -> Tuple[float, float]:
    """Evaluate one quadratic Bezier point."""

    t = min(1.0, max(0.0, float(t_value)))
    inv = 1.0 - t
    x = (inv * inv * float(start[0])) + (2.0 * inv * t * float(control[0])) + (t * t * float(end[0]))
    y = (inv * inv * float(start[1])) + (2.0 * inv * t * float(control[1])) + (t * t * float(end[1]))
    return (float(x), float(y))


def _quadratic_bezier_tangent(start: Point, control: Point, end: Point, t_value: float) -> Tuple[float, float]:
    """Return the normalized tangent for one quadratic Bezier point."""

    t = min(1.0, max(0.0, float(t_value)))
    dx = (2.0 * (1.0 - t) * (float(control[0]) - float(start[0]))) + (2.0 * t * (float(end[0]) - float(control[0])))
    dy = (2.0 * (1.0 - t) * (float(control[1]) - float(start[1]))) + (2.0 * t * (float(end[1]) - float(control[1])))
    norm = float(math.hypot(dx, dy))
    if norm <= 1e-6:
        fallback_dx = float(end[0]) - float(start[0])
        fallback_dy = float(end[1]) - float(start[1])
        fallback_norm = float(math.hypot(fallback_dx, fallback_dy))
        if fallback_norm <= 1e-6:
            return (1.0, 0.0)
        return (float(fallback_dx / fallback_norm), float(fallback_dy / fallback_norm))
    return (float(dx / norm), float(dy / norm))


def _resolve_arc_control_candidates(
    *,
    start: Point,
    end: Point,
    content_bbox: BBox,
    edge_key: Tuple[str, str],
) -> Tuple[Point, ...]:
    """Resolve deterministic arc control points for an edge, outward first."""

    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    norm = float(math.hypot(dx, dy))
    if norm <= 1e-6:
        point = (int(round(x0)), int(round(y0)))
        return (point,)
    mid_x = 0.5 * float(x0 + x1)
    mid_y = 0.5 * float(y0 + y1)
    perp_x = float(-dy / norm)
    perp_y = float(dx / norm)
    content_center_x = 0.5 * float(int(content_bbox[0]) + int(content_bbox[2]))
    content_center_y = 0.5 * float(int(content_bbox[1]) + int(content_bbox[3]))
    outward_x = float(mid_x - content_center_x)
    outward_y = float(mid_y - content_center_y)
    dot = float((perp_x * outward_x) + (perp_y * outward_y))
    if abs(dot) <= 1e-6:
        edge_hash = sum(ord(char) for char in f"{edge_key[0]}-{edge_key[1]}")
        sign = 1.0 if int(edge_hash) % 2 == 0 else -1.0
    else:
        sign = 1.0 if dot > 0.0 else -1.0
    offset = min(76.0, max(28.0, float(norm) * 0.22))
    controls: List[Point] = []
    for candidate_sign in (sign, -sign):
        control_x = float(mid_x + (candidate_sign * perp_x * offset))
        control_y = float(mid_y + (candidate_sign * perp_y * offset))
        control = (int(round(control_x)), int(round(control_y)))
        if control not in controls:
            controls.append(control)
    return tuple(controls)


def _resolve_arc_control_point(
    *,
    start: Point,
    end: Point,
    content_bbox: BBox,
    edge_key: Tuple[str, str],
) -> Point:
    """Resolve one outward-bowed arc control point for an edge."""

    return _resolve_arc_control_candidates(
        start=start,
        end=end,
        content_bbox=content_bbox,
        edge_key=edge_key,
    )[0]


def _curve_clears_non_endpoint_nodes(
    *,
    start: Point,
    control: Point,
    end: Point,
    other_node_centers: Sequence[Point],
    node_radius_px: int,
) -> bool:
    """Return whether an arced edge stays visually clear of unrelated nodes."""

    if not other_node_centers:
        return True

    sample_count = 64
    min_clearance_px = float(max(48, int(node_radius_px) * 2 + 8))
    for index in range(1, int(sample_count)):
        x, y = _quadratic_bezier_point(
            start,
            control,
            end,
            float(index) / float(sample_count),
        )
        for node_x, node_y in other_node_centers:
            if math.hypot(float(x) - float(node_x), float(y) - float(node_y)) < min_clearance_px:
                return False
    return True


def _resolve_edge_route_controls(
    *,
    edge_labels: Sequence[Tuple[str, str]],
    label_to_node: Mapping[str, int],
    positions: Mapping[int, Point],
    content_bbox: BBox,
    edge_routing_variant: str,
    node_radius_px: int,
) -> Dict[Tuple[str, str], Point | None]:
    """Resolve optional arc controls for one graph render."""

    controls: Dict[Tuple[str, str], Point | None] = {
        (str(left), str(right)): None
        for left, right in edge_labels
    }
    if str(edge_routing_variant) != "mixed_arc" or len(edge_labels) < 3:
        return controls

    node_center_by_label = {
        str(label): tuple(int(value) for value in positions[int(node)])
        for label, node in label_to_node.items()
    }
    scored_edges: List[Tuple[float, str, str, Point]] = []
    min_curve_distance = float(max(64, int(node_radius_px) * 4))
    for left_label, right_label in edge_labels:
        start = tuple(int(value) for value in positions[int(label_to_node[str(left_label)])])
        end = tuple(int(value) for value in positions[int(label_to_node[str(right_label)])])
        distance = math.hypot(float(end[0] - start[0]), float(end[1] - start[1]))
        if distance < min_curve_distance:
            continue
        control_candidates = _resolve_arc_control_candidates(
            start=start,
            end=end,
            content_bbox=tuple(int(value) for value in content_bbox),
            edge_key=(str(left_label), str(right_label)),
        )
        other_node_centers = tuple(
            center
            for label, center in node_center_by_label.items()
            if label not in {str(left_label), str(right_label)}
        )
        selected_control = None
        for control in control_candidates:
            if _curve_clears_non_endpoint_nodes(
                start=start,
                control=tuple(int(value) for value in control),
                end=end,
                other_node_centers=other_node_centers,
                node_radius_px=int(node_radius_px),
            ):
                selected_control = tuple(int(value) for value in control)
                break
        if selected_control is None:
            continue
        scored_edges.append((float(distance), str(left_label), str(right_label), selected_control))
    if not scored_edges:
        return controls

    curve_count = min(len(scored_edges), max(1, int(round(float(len(edge_labels)) * 0.30))))
    for _, left_label, right_label, control in sorted(scored_edges, key=lambda item: (-item[0], item[1], item[2]))[
        :curve_count
    ]:
        controls[(str(left_label), str(right_label))] = tuple(int(value) for value in control)
    return controls


def _draw_edge(
    draw: ImageDraw.ImageDraw,
    *,
    start: Point,
    end: Point,
    control: Point | None = None,
    node_radius_px: int,
    edge_width_px: int,
    edge_color_rgb: Sequence[int],
    directed: bool,
    arrow_length_px: int,
    arrow_width_px: int,
) -> Tuple[Point, Point]:
    """Draw one graph edge and return the visible line segment endpoints."""

    if control is not None:
        sample_count = 48
        samples = [
            _quadratic_bezier_point(start, tuple(int(value) for value in control), end, float(index) / float(sample_count))
            for index in range(int(sample_count) + 1)
        ]
        radius = float(max(1, int(node_radius_px)))
        start_index = 0
        for index, point in enumerate(samples):
            if math.hypot(float(point[0]) - float(start[0]), float(point[1]) - float(start[1])) >= radius:
                start_index = int(index)
                break
        tip_index = int(sample_count)
        for index in range(int(sample_count), -1, -1):
            point = samples[int(index)]
            if math.hypot(float(point[0]) - float(end[0]), float(point[1]) - float(end[1])) >= radius:
                tip_index = int(index)
                break
        if tip_index <= start_index:
            return (start, end)

        tip = samples[int(tip_index)]
        tangent = _quadratic_bezier_tangent(
            start,
            tuple(int(value) for value in control),
            end,
            float(tip_index) / float(sample_count),
        )
        line_points = [
            (float(point[0]), float(point[1]))
            for point in samples[int(start_index) : int(tip_index) + 1]
        ]
        if bool(directed):
            line_end = (
                float(tip[0] - (tangent[0] * max(4, int(arrow_length_px) - 1))),
                float(tip[1] - (tangent[1] * max(4, int(arrow_length_px) - 1))),
            )
            if line_points:
                line_points[-1] = line_end
        else:
            line_end = tuple(float(value) for value in tip)
        draw.line(
            line_points,
            fill=tuple(int(v) for v in edge_color_rgb),
            width=max(1, int(edge_width_px)),
        )
        if bool(directed):
            perp_x = float(-tangent[1])
            perp_y = float(tangent[0])
            base = (
                float(tip[0] - (tangent[0] * int(arrow_length_px))),
                float(tip[1] - (tangent[1] * int(arrow_length_px))),
            )
            left = (
                float(base[0] + (perp_x * int(arrow_width_px))),
                float(base[1] + (perp_y * int(arrow_width_px))),
            )
            right = (
                float(base[0] - (perp_x * int(arrow_width_px))),
                float(base[1] - (perp_y * int(arrow_width_px))),
            )
            draw.polygon(
                [
                    (int(round(tip[0])), int(round(tip[1]))),
                    (int(round(left[0])), int(round(left[1]))),
                    (int(round(right[0])), int(round(right[1]))),
                ],
                fill=tuple(int(v) for v in edge_color_rgb),
            )
        line_start = line_points[0]
        return (
            (int(round(line_start[0])), int(round(line_start[1]))),
            (int(round(line_end[0])), int(round(line_end[1]))),
        )

    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    norm = float(math.hypot(dx, dy))
    if norm <= 1e-6:
        return (start, end)
    ux = float(dx / norm)
    uy = float(dy / norm)
    radius = float(max(1, int(node_radius_px)))
    line_start = (float(x0 + (ux * radius)), float(y0 + (uy * radius)))
    tip = (float(x1 - (ux * radius)), float(y1 - (uy * radius)))
    if bool(directed):
        line_end = (
            float(tip[0] - (ux * max(4, int(arrow_length_px) - 1))),
            float(tip[1] - (uy * max(4, int(arrow_length_px) - 1))),
        )
    else:
        line_end = tip
    draw.line(
        (line_start[0], line_start[1], line_end[0], line_end[1]),
        fill=tuple(int(v) for v in edge_color_rgb),
        width=max(1, int(edge_width_px)),
    )
    if bool(directed):
        perp_x = float(-uy)
        perp_y = float(ux)
        base = (
            float(tip[0] - (ux * int(arrow_length_px))),
            float(tip[1] - (uy * int(arrow_length_px))),
        )
        left = (
            float(base[0] + (perp_x * int(arrow_width_px))),
            float(base[1] + (perp_y * int(arrow_width_px))),
        )
        right = (
            float(base[0] - (perp_x * int(arrow_width_px))),
            float(base[1] - (perp_y * int(arrow_width_px))),
        )
        draw.polygon(
            [
                (int(round(tip[0])), int(round(tip[1]))),
                (int(round(left[0])), int(round(left[1]))),
                (int(round(right[0])), int(round(right[1]))),
            ],
            fill=tuple(int(v) for v in edge_color_rgb),
        )
    return (
        (int(round(line_start[0])), int(round(line_start[1]))),
        (int(round(line_end[0])), int(round(line_end[1]))),
    )


def _draw_edge_boxed_label(
    draw: ImageDraw.ImageDraw,
    *,
    box: BBox,
    text: str,
    font_size_px: int,
    font_family: str,
    box_fill_rgb: Sequence[int],
    box_border_rgb: Sequence[int],
    text_rgb: Sequence[int],
    layout_seed: int,
) -> BBox:
    """Draw one boxed edge label inside the provided bbox."""

    font = load_font(max(14, int(font_size_px)), bold=True, font_family=str(font_family or ""))
    resolved_font_size = float(getattr(font, "size", font_size_px))
    stroke_width = max(1, int(round(resolved_font_size * 0.12)))
    center = (0.5 * float(box[0] + box[2]), 0.5 * float(box[1] + box[3]))
    draw.rounded_rectangle(
        box,
        radius=max(6, int(round((min(int(box[2] - box[0]), int(box[3] - box[1])) * 0.22)))),
        fill=tuple(int(value) for value in box_fill_rgb),
        outline=tuple(int(value) for value in box_border_rgb),
        width=max(2, int(round(resolved_font_size * 0.10))),
    )
    label_style = resolve_readable_text_style(
        instance_seed=int(layout_seed),
        namespace="graph.node_link.edge_label_text",
        role="graph_edge_label_text",
        surface_rgbs=(tuple(int(value) for value in box_fill_rgb),),
        preferred_rgbs=(tuple(int(value) for value in text_rgb),),
        min_contrast_ratio=7.0,
        min_lab_distance=38.0,
    )
    draw_centered_readable_text(
        draw,
        text=text,
        center=center,
        font=font,
        style=label_style,
        stroke_width=int(stroke_width),
    )
    return tuple(int(value) for value in box)


def _draw_edge_weight_label(
    draw: ImageDraw.ImageDraw,
    *,
    box: BBox,
    weight: int,
    font_size_px: int,
    font_family: str,
    box_fill_rgb: Sequence[int],
    box_border_rgb: Sequence[int],
    text_rgb: Sequence[int],
    layout_seed: int,
) -> BBox:
    """Draw one boxed edge-weight label inside the provided bbox."""

    return _draw_edge_boxed_label(
        draw,
        box=tuple(int(value) for value in box),
        text=str(int(weight)),
        font_size_px=int(font_size_px),
        font_family=str(font_family or ""),
        box_fill_rgb=tuple(int(value) for value in box_fill_rgb),
        box_border_rgb=tuple(int(value) for value in box_border_rgb),
        text_rgb=tuple(int(value) for value in text_rgb),
        layout_seed=int(layout_seed),
    )


def _resolve_edge_boxed_label_box(
    draw: ImageDraw.ImageDraw,
    *,
    segment: Tuple[Point, Point],
    text: str,
    font_size_px: int,
    font_family: str,
    offset_px: int,
    padding_px: int,
    content_bbox: BBox,
    other_segments: Sequence[Tuple[Point, Point]],
    reserved_boxes: Sequence[BBox],
    node_bboxes: Sequence[BBox],
    side_seed: int,
    require_strict: bool = False,
) -> BBox:
    """Choose one readable edge-label bbox that avoids edge and node collisions."""

    start, end = segment
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    dx = float(x1 - x0)
    dy = float(y1 - y0)
    norm = float(math.hypot(dx, dy))
    mid_x = 0.5 * float(x0 + x1)
    mid_y = 0.5 * float(y0 + y1)
    if float(norm) <= 1e-6:
        norm_x, norm_y = 0.0, -1.0
    else:
        norm_x = float(-dy / norm)
        norm_y = float(dx / norm)

    font = load_font(max(14, int(font_size_px)), bold=True, font_family=str(font_family or ""))
    label_text = str(text)
    stroke_width = max(1, int(round(float(getattr(font, "size", font_size_px)) * 0.10)))
    raw_bbox = draw.textbbox((0, 0), label_text, font=font, stroke_width=int(stroke_width))
    text_width = max(1, int(raw_bbox[2] - raw_bbox[0]))
    text_height = max(1, int(raw_bbox[3] - raw_bbox[1]))
    pad = max(2, int(padding_px))
    half_w = float((text_width / 2.0) + pad)
    half_h = float((text_height / 2.0) + pad)
    required_clearance = float(abs(norm_x) * half_w) + float(abs(norm_y) * half_h) + float(max(4, pad))

    preferred_signs = (1, -1) if int(side_seed) % 2 == 0 else (-1, 1)
    t_positions = (0.50, 0.42, 0.58, 0.34, 0.66, 0.26, 0.74)
    offset_scales = (1.0, 1.35, 1.7, 2.1, 2.5, 3.0)
    content = tuple(int(value) for value in content_bbox)

    best_box: BBox | None = None
    best_score: tuple[float, float, float, float, float] | None = None
    for sign_value in preferred_signs:
        sign = float(sign_value)
        for offset_scale in offset_scales:
            for t_value in t_positions:
                base_x = float(x0 + (float(dx) * float(t_value)))
                base_y = float(y0 + (float(dy) * float(t_value)))
                normal_distance = max(float(offset_px) * float(offset_scale), float(required_clearance))
                center = (
                    float(base_x + (sign * float(norm_x) * float(normal_distance))),
                    float(base_y + (sign * float(norm_y) * float(normal_distance))),
                )
                box = (
                    int(round(center[0] - half_w)),
                    int(round(center[1] - half_h)),
                    int(round(center[0] + half_w)),
                    int(round(center[1] + half_h)),
                )
                out_of_bounds = (
                    max(0, int(content[0] - box[0]))
                    + max(0, int(content[1] - box[1]))
                    + max(0, int(box[2] - content[2]))
                    + max(0, int(box[3] - content[3]))
                )
                own_segment_crossings = int(_segment_intersects_bbox(segment, box))
                segment_crossings = sum(1 for other_segment in other_segments if _segment_intersects_bbox(other_segment, box))
                node_overlaps = sum(1 for node_bbox in node_bboxes if _bbox_intersects_bbox(node_bbox, box))
                label_overlaps = sum(1 for reserved in reserved_boxes if _bbox_intersects_bbox(reserved, box))
                score = (
                    float(own_segment_crossings),
                    float(segment_crossings),
                    float(node_overlaps + label_overlaps),
                    float(out_of_bounds),
                    float(normal_distance + abs(float(t_value) - 0.5)),
                )
                if best_score is None or score < best_score:
                    best_score = score
                    best_box = tuple(int(value) for value in box)
                    if score[:4] == (0.0, 0.0, 0.0, 0.0):
                        return tuple(int(value) for value in best_box)
    if best_box is None:
        raise ValueError("failed to resolve one edge label box")
    if bool(require_strict) and (best_score is None or best_score[:4] != (0.0, 0.0, 0.0, 0.0)):
        raise ValueError("failed to resolve a collision-free edge label box")
    return tuple(int(value) for value in best_box)


def _resolve_edge_weight_label_box(
    draw: ImageDraw.ImageDraw,
    *,
    segment: Tuple[Point, Point],
    weight: int,
    font_size_px: int,
    font_family: str,
    offset_px: int,
    padding_px: int,
    content_bbox: BBox,
    other_segments: Sequence[Tuple[Point, Point]],
    reserved_boxes: Sequence[BBox],
    node_bboxes: Sequence[BBox],
    side_seed: int,
) -> BBox:
    """Choose one readable weight-label bbox that avoids edge and node collisions."""

    return _resolve_edge_boxed_label_box(
        draw,
        segment=tuple(segment),
        text=str(int(weight)),
        font_size_px=int(font_size_px),
        font_family=str(font_family or ""),
        offset_px=int(offset_px),
        padding_px=int(padding_px),
        content_bbox=tuple(int(value) for value in content_bbox),
        other_segments=tuple(other_segments),
        reserved_boxes=tuple(reserved_boxes),
        node_bboxes=tuple(node_bboxes),
        side_seed=int(side_seed),
    )


__all__ = [
    "_draw_edge",
    "_draw_edge_boxed_label",
    "_draw_edge_weight_label",
    "_resolve_edge_boxed_label_box",
    "_resolve_edge_route_controls",
    "_resolve_edge_weight_label_box",
]
