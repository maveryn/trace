"""Rendering primitives for process-flow page scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.pages.shared.diagram.common import round_diagram_bbox
from trace_tasks.tasks.shared.drawing import (
    draw_arrow,
    draw_centered_text,
    draw_dashed_line,
    draw_rounded_rect,
)
from trace_tasks.tasks.shared.text_rendering import fit_font_to_box, load_font

from .state import (
    BADGE_COLORS,
    ROLE_FILL_ADJUST,
    STYLE_PALETTES,
    ProcessFlowRenderParams,
    ProcessFlowSceneCase,
    RenderedProcessFlow,
)


def _round_box(box: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in box]


def _round_point(point: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in point[:2]]


def _point_box(points: Sequence[tuple[float, float]], *, padding: float) -> list[float]:
    if not points:
        return [0.0, 0.0, 0.0, 0.0]
    return [
        round(min(float(point[0]) for point in points) - float(padding), 3),
        round(min(float(point[1]) for point in points) - float(padding), 3),
        round(max(float(point[0]) for point in points) + float(padding), 3),
        round(max(float(point[1]) for point in points) + float(padding), 3),
    ]


def _box_overlaps(a: Sequence[float], b: Sequence[float], *, padding: float = 0.0) -> bool:
    ax0, ay0, ax1, ay1 = [float(value) for value in a]
    bx0, by0, bx1, by1 = [float(value) for value in b]
    return not (
        ax1 + float(padding) <= bx0
        or bx1 + float(padding) <= ax0
        or ay1 + float(padding) <= by0
        or by1 + float(padding) <= ay0
    )


def _shift_box(box: Sequence[float], *, dx: float = 0.0, dy: float = 0.0) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in box]
    return [x0 + float(dx), y0 + float(dy), x1 + float(dx), y1 + float(dy)]


def _clamp_box_to_bounds(box: Sequence[float], bounds: Sequence[float]) -> list[float]:
    x0, y0, x1, y1 = [float(value) for value in box]
    bx0, by0, bx1, by1 = [float(value) for value in bounds]
    width = x1 - x0
    height = y1 - y0
    nx0 = min(max(x0, bx0 + 4.0), bx1 - width - 4.0)
    ny0 = min(max(y0, by0 + 4.0), by1 - height - 4.0)
    return [nx0, ny0, nx0 + width, ny0 + height]


def _avoid_label_collisions(
    box: Sequence[float],
    *,
    occupied_boxes: Sequence[Sequence[float]],
    bounds: Sequence[float],
    vertical: bool,
    edge_index: int,
) -> list[float]:
    """Move edge labels away from nodes and earlier labels when possible."""

    base = _clamp_box_to_bounds(box, bounds)
    if not any(_box_overlaps(base, other, padding=4.0) for other in occupied_boxes):
        return _round_box(base)
    shifts: list[float] = []
    for distance in (18.0, 34.0, 52.0, 72.0):
        sign = 1.0 if (int(edge_index) % 2 == 0) else -1.0
        shifts.extend([sign * distance, -sign * distance])
    for shift in shifts:
        candidate = _shift_box(base, dy=shift) if bool(vertical) else _shift_box(base, dx=shift)
        candidate = _clamp_box_to_bounds(candidate, bounds)
        if not any(_box_overlaps(candidate, other, padding=4.0) for other in occupied_boxes):
            return _round_box(candidate)
    return _round_box(base)


def _lighten(color: Sequence[int], amount: float) -> tuple[int, int, int]:
    factor = max(0.0, min(1.0, float(amount)))
    return tuple(
        int(round(int(channel) + ((255 - int(channel)) * factor)))
        for channel in color[:3]
    )


def _draw_text_in_box(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    text: str,
    max_size_px: int,
    min_size_px: int,
    fill: Sequence[int],
    bold: bool = True,
    stroke_fill: Sequence[int] = (255, 255, 255),
    padding_px: int = 5,
) -> list[float]:
    left, top, right, bottom = [float(value) for value in bbox]
    font = fit_font_to_box(
        draw,
        text=str(text),
        max_width=max(1.0, float(right - left - (2 * int(padding_px)))),
        max_height=max(1.0, float(bottom - top - (2 * int(padding_px)))),
        bold=bool(bold),
        min_size_px=int(min_size_px),
        max_size_px=int(max_size_px),
        fill_ratio=0.94,
    )
    return draw_centered_text(
        draw,
        text=str(text),
        center=(0.5 * (float(left) + float(right)), 0.5 * (float(top) + float(bottom))),
        font=font,
        fill=tuple(int(value) for value in fill),
        stroke_fill=tuple(int(value) for value in stroke_fill),
        stroke_width=1,
    )


def _draw_badge(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    text: str,
    fill: Sequence[int],
    outline: Sequence[int],
    text_fill: Sequence[int],
    font_size_px: int,
) -> list[float]:
    draw.rounded_rectangle(
        tuple(float(value) for value in bbox),
        radius=max(4, int((float(bbox[3]) - float(bbox[1])) * 0.45)),
        fill=tuple(int(value) for value in fill),
        outline=tuple(int(value) for value in outline),
        width=1,
    )
    return _draw_text_in_box(
        draw,
        bbox=bbox,
        text=str(text),
        max_size_px=int(font_size_px),
        min_size_px=8,
        fill=text_fill,
        bold=True,
        padding_px=4,
    )


def _draw_node(
    draw: ImageDraw.ImageDraw,
    *,
    node: Mapping[str, Any],
    palette: Mapping[str, Any],
    render_params: ProcessFlowRenderParams,
) -> tuple[list[float], list[float]]:
    """Draw one process step and preserve the node bbox as its verifier witness.

    The node label/status text boxes are render diagnostics only; node-count
    tasks always annotate the outer rendered step shape, so role-specific shape
    drawing must not mutate the sampled node bbox.
    """

    bbox = [float(value) for value in node["bbox"]]
    left, top, right, bottom = bbox
    role = str(node["role"])
    shape = str(node["shape"])
    fill = tuple(_lighten(ROLE_FILL_ADJUST.get(role, palette["node_fill"]), 0.14))
    if role in {"start", "process", "output"}:
        fill = tuple(int(value) for value in ROLE_FILL_ADJUST.get(role, palette["node_fill"]))
    outline = tuple(int(value) for value in palette["node_border"])
    width = max(2, int(render_params.node_border_width_px))
    radius = max(6, int(render_params.node_corner_radius_px))
    if shape == "diamond":
        points = [
            (0.5 * (left + right), top),
            (right, 0.5 * (top + bottom)),
            (0.5 * (left + right), bottom),
            (left, 0.5 * (top + bottom)),
        ]
        draw.polygon(points, fill=fill, outline=outline)
        for offset in range(1, width):
            inset = float(offset)
            inner = [
                (0.5 * (left + right), top + inset),
                (right - inset, 0.5 * (top + bottom)),
                (0.5 * (left + right), bottom - inset),
                (left + inset, 0.5 * (top + bottom)),
            ]
            draw.line(inner + [inner[0]], fill=outline, width=1)
    elif shape == "ellipse":
        draw.ellipse(tuple(bbox), fill=fill, outline=outline, width=width)
    elif shape == "parallelogram":
        slant = min(18.0, 0.14 * float(right - left))
        points = [(left + slant, top), (right, top), (right - slant, bottom), (left, bottom)]
        draw.polygon(points, fill=fill, outline=outline)
        draw.line(points + [points[0]], fill=outline, width=width)
    else:
        draw_rounded_rect(
            draw,
            tuple(bbox),
            radius=radius,
            fill=fill,
            outline=outline,
            width=width,
        )

    status = str(node["status"])
    has_badge = role not in {"start", "output"}
    if has_badge:
        badge_h = 18.0
        badge_w = min(76.0, max(50.0, 9.0 * len(status)))
        badge_bbox = [
            0.5 * (left + right) - (0.5 * badge_w),
            bottom - badge_h - 5.0,
            0.5 * (left + right) + (0.5 * badge_w),
            bottom - 5.0,
        ]
        label_bbox = [left + 11.0, top + 6.0, right - 11.0, badge_bbox[1] - 2.0]
    else:
        badge_bbox = [0.0, 0.0, 0.0, 0.0]
        label_bbox = [left + 10.0, top + 8.0, right - 10.0, bottom - 8.0]
    label_text_bbox = _draw_text_in_box(
        draw,
        bbox=label_bbox,
        text=str(node["label"]),
        max_size_px=int(render_params.node_label_font_size_px),
        min_size_px=10,
        fill=palette["text"],
        bold=True,
        stroke_fill=(255, 255, 255),
        padding_px=2,
    )
    if has_badge:
        status_bbox = _draw_badge(
            draw,
            bbox=badge_bbox,
            text=status,
            fill=BADGE_COLORS.get(status, (232, 232, 232)),
            outline=palette["node_border"],
            text_fill=palette["text"],
            font_size_px=int(render_params.badge_font_size_px),
        )
    else:
        status_bbox = [0.0, 0.0, 0.0, 0.0]
    return _round_box(label_text_bbox), _round_box(status_bbox)


def _draw_arrowhead(
    draw: ImageDraw.ImageDraw,
    *,
    start: tuple[float, float],
    end: tuple[float, float],
    fill: Sequence[int],
    length: float,
    width: float,
) -> None:
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    dx = ex - sx
    dy = ey - sy
    segment_len = math.hypot(dx, dy)
    if segment_len <= 1e-6:
        return
    ux = dx / segment_len
    uy = dy / segment_len
    head_len = min(float(length), 0.5 * segment_len)
    base = (ex - (ux * head_len), ey - (uy * head_len))
    px = -uy
    py = ux
    half_w = 0.5 * float(width)
    draw.polygon(
        [
            (ex, ey),
            (base[0] + (px * half_w), base[1] + (py * half_w)),
            (base[0] - (px * half_w), base[1] - (py * half_w)),
        ],
        fill=tuple(int(value) for value in fill),
        outline=tuple(int(value) for value in fill),
    )


def _draw_polyline_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    points: Sequence[tuple[float, float]],
    fill: Sequence[int],
    width: int,
    head_length_px: float,
    head_width_px: float,
    dashed: bool,
) -> list[float]:
    """Draw one routed arrow while keeping its endpoint segment stable.

    The verifier segment for handoff tasks uses the first and last routed points,
    independent of dashed styling or intermediate bend points, so this helper
    may vary stroke style without changing arrow endpoint semantics.
    """

    clean = [(float(x), float(y)) for x, y in points]
    if len(clean) < 2:
        return [0.0, 0.0, 0.0, 0.0]
    for index, (start, end) in enumerate(zip(clean, clean[1:])):
        is_last = index == len(clean) - 2
        if dashed:
            draw_dashed_line(
                draw,
                start=start,
                end=end,
                fill=fill,
                width=max(1, int(width)),
                dash_px=10.0,
                gap_px=7.0,
            )
            if is_last:
                _draw_arrowhead(
                    draw,
                    start=start,
                    end=end,
                    fill=fill,
                    length=float(head_length_px),
                    width=float(head_width_px),
                )
        elif is_last:
            draw_arrow(
                draw,
                start=start,
                end=end,
                fill=fill,
                width=max(1, int(width)),
                head_length_px=float(head_length_px),
                head_width_px=float(head_width_px),
            )
        else:
            draw.line([start, end], fill=tuple(int(value) for value in fill), width=max(1, int(width)))
    return _point_box(clean, padding=max(7.0, float(width) + 4.0))


def _edge_points(
    *,
    source: Mapping[str, Any],
    target: Mapping[str, Any],
    layout_variant: str,
    edge_index: int,
    edge_kind: str,
) -> list[tuple[float, float]]:
    """Route one process arrow with separated orthogonal corridors.

    The returned first/last points remain the annotation segment endpoints for
    handoff tasks; intermediate bend points only improve visual separation.
    """

    sx, sy = [float(value) for value in source["center"]]
    tx, ty = [float(value) for value in target["center"]]
    sw, sh = float(source["width"]), float(source["height"])
    tw, th = float(target["width"]), float(target["height"])
    if str(edge_kind) in {"handoff_chain", "branch_clean", "decision_clean"}:
        rightward = tx >= sx
        if rightward:
            return [(sx + (0.5 * sw), sy), (tx - (0.5 * tw), ty)]
        return [(sx - (0.5 * sw), sy), (tx + (0.5 * tw), ty)]
    vertical = str(layout_variant) in {"vertical_swimlane", "staggered_columns"}
    route_offsets = (-32.0, -16.0, 0.0, 16.0, 32.0)
    route_offset = route_offsets[int(edge_index) % len(route_offsets)]
    if str(edge_kind) == "side":
        route_offset *= 1.35
    port_offset = max(-15.0, min(15.0, 0.45 * route_offset))
    if vertical:
        downward = ty >= sy
        start = (sx + port_offset, sy + (0.5 * sh)) if downward else (sx + port_offset, sy - (0.5 * sh))
        end = (tx - port_offset, ty - (0.5 * th)) if downward else (tx - port_offset, ty + (0.5 * th))
        if abs(start[0] - end[0]) < 18.0 and str(edge_kind) != "decision":
            return [start, end]
        mid_y = 0.5 * (start[1] + end[1]) + route_offset
        lower = min(start[1], end[1]) + 20.0
        upper = max(start[1], end[1]) - 20.0
        if lower < upper:
            mid_y = max(lower, min(upper, mid_y))
        return [start, (start[0], mid_y), (end[0], mid_y), end]
    rightward = tx >= sx
    start = (sx + (0.5 * sw), sy + port_offset) if rightward else (sx - (0.5 * sw), sy + port_offset)
    end = (tx - (0.5 * tw), ty - port_offset) if rightward else (tx + (0.5 * tw), ty - port_offset)
    if abs(start[1] - end[1]) < 18.0 and str(edge_kind) != "decision":
        return [start, end]
    mid_x = 0.5 * (start[0] + end[0]) + route_offset
    lower = min(start[0], end[0]) + 20.0
    upper = max(start[0], end[0]) - 20.0
    if lower < upper:
        mid_x = max(lower, min(upper, mid_x))
    return [start, (mid_x, start[1]), (mid_x, end[1]), end]


def _edge_label_bbox(
    *,
    points: Sequence[tuple[float, float]],
    text: str,
    layout_variant: str,
    edge_index: int,
) -> list[float]:
    if len(points) < 2:
        return [0.0, 0.0, 0.0, 0.0]
    mid_segment = max(0, min(len(points) - 2, (len(points) - 1) // 2))
    a = points[mid_segment]
    b = points[mid_segment + 1]
    cx = 0.5 * (float(a[0]) + float(b[0]))
    cy = 0.5 * (float(a[1]) + float(b[1]))
    width = min(84.0, max(46.0, 9.5 * len(str(text)) + 16.0))
    height = 24.0
    offset = 26.0 if int(edge_index) % 2 == 0 else -26.0
    if str(layout_variant) in {"vertical_swimlane", "staggered_columns"}:
        cy += offset
    else:
        cx += offset
    return [cx - (0.5 * width), cy - (0.5 * height), cx + (0.5 * width), cy + (0.5 * height)]


def _draw_edge_label(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    text: str,
    palette: Mapping[str, Any],
    render_params: ProcessFlowRenderParams,
) -> list[float]:
    draw.rounded_rectangle(
        tuple(float(value) for value in bbox),
        radius=9,
        fill=(255, 255, 255),
        outline=tuple(int(value) for value in palette["edge"]),
        width=1,
    )
    return _draw_text_in_box(
        draw,
        bbox=bbox,
        text=str(text),
        max_size_px=int(render_params.edge_label_font_size_px),
        min_size_px=9,
        fill=palette["text"],
        bold=True,
        stroke_fill=(255, 255, 255),
        padding_px=4,
    )


def render_process_flow_scene(
    background: Image.Image,
    *,
    case: ProcessFlowSceneCase,
    instance_seed: int,
    namespace: str,
) -> RenderedProcessFlow:
    """Draw one resolved process-flow scene and return projection maps."""

    image = background.convert("RGB")
    draw = ImageDraw.Draw(image)
    render_params = case.render_params
    palette = STYLE_PALETTES[str(case.style_variant)]
    draw.rounded_rectangle(
        tuple(float(value) for value in case.panel_bbox),
        radius=max(4, int(render_params.panel_corner_radius_px)),
        fill=tuple(int(value) for value in palette["panel_fill"]),
        outline=tuple(int(value) for value in palette["panel_border"]),
        width=3,
    )
    title_font = load_font(int(render_params.title_font_size_px), bold=True)
    draw_centered_text(
        draw,
        text=str(case.scene_title),
        center=(
            0.5 * (float(case.title_bbox[0]) + float(case.title_bbox[2])),
            0.5 * (float(case.title_bbox[1]) + float(case.title_bbox[3])),
        ),
        font=title_font,
        fill=tuple(int(value) for value in palette["title"]),
        stroke_fill=(255, 255, 255),
        stroke_width=1,
    )
    vertical = str(case.layout_variant) in {"vertical_swimlane", "staggered_columns"}
    lane_fills = list(palette["lane_fills"])
    lane_header_fill = tuple(int(value) for value in palette["lane_header"])
    lane_font = load_font(int(render_params.lane_font_size_px), bold=True)
    for index, lane in enumerate(case.lanes):
        bbox = [float(value) for value in case.lane_bboxes[str(lane)]]
        fill = tuple(int(value) for value in lane_fills[index % len(lane_fills)])
        draw.rectangle(tuple(bbox), fill=fill)
        draw.rectangle(tuple(bbox), outline=tuple(int(value) for value in palette["panel_border"]), width=1)
        if vertical:
            header_bbox = [bbox[0], bbox[1], bbox[2], bbox[1] + float(render_params.lane_header_height_px)]
        else:
            header_width = max(
                112.0 if str(case.flow_family) == "handoff_chain" else 88.0,
                float(render_params.lane_header_height_px),
            )
            header_bbox = [bbox[0], bbox[1], bbox[0] + header_width, bbox[3]]
        draw.rectangle(
            tuple(header_bbox),
            fill=lane_header_fill,
            outline=tuple(int(value) for value in palette["panel_border"]),
            width=2 if str(case.flow_family) == "handoff_chain" else 1,
        )
        if not vertical and str(case.flow_family) == "handoff_chain":
            label_pad = 8.0
            label_bbox = [
                header_bbox[0] + label_pad,
                header_bbox[1] + 0.5 * (header_bbox[3] - header_bbox[1]) - 18.0,
                header_bbox[2] - label_pad,
                header_bbox[1] + 0.5 * (header_bbox[3] - header_bbox[1]) + 18.0,
            ]
            draw.rounded_rectangle(
                tuple(label_bbox),
                radius=8,
                fill=(255, 255, 255),
                outline=tuple(int(value) for value in palette["panel_border"]),
                width=1,
            )
            _draw_text_in_box(
                draw,
                bbox=label_bbox,
                text=str(lane),
                max_size_px=int(render_params.lane_font_size_px),
                min_size_px=8,
                fill=palette["title"],
                bold=True,
                stroke_fill=(255, 255, 255),
                padding_px=3,
            )
        else:
            draw_centered_text(
                draw,
                text=str(lane),
                center=(0.5 * (header_bbox[0] + header_bbox[2]), 0.5 * (header_bbox[1] + header_bbox[3])),
                font=lane_font,
                fill=tuple(int(value) for value in palette["title"]),
                stroke_fill=(255, 255, 255),
                stroke_width=1,
            )

    nodes = [dict(node) for node in case.nodes]
    edges = [dict(edge) for edge in case.edges]
    node_by_ref = {str(node["node_id"]): node for node in nodes}
    edge_bbox_map: Dict[str, list[float]] = {}
    edge_segment_map: Dict[str, list[list[float]]] = {}
    edge_polyline_map: Dict[str, list[list[float]]] = {}
    edge_label_bbox_map: Dict[str, list[float]] = {}
    edge_label_text_bbox_map: Dict[str, list[float]] = {}
    edge_style_rng = spawn_rng(int(instance_seed), f"{namespace}.edge_style")
    dashed_side_edges = bool(edge_style_rng.randrange(0, 2))
    occupied_label_boxes: list[list[float]] = [list(node["bbox"]) for node in nodes]
    label_bounds = [
        float(case.content_bbox[0]) + 6.0,
        float(case.content_bbox[1]) + 6.0,
        float(case.content_bbox[2]) - 6.0,
        float(case.content_bbox[3]) - 6.0,
    ]
    for edge_index, edge in enumerate(edges):
        source = node_by_ref[str(edge["source"])]
        target = node_by_ref[str(edge["target"])]
        points = _edge_points(
            source=source,
            target=target,
            layout_variant=str(case.layout_variant),
            edge_index=int(edge_index),
            edge_kind=str(edge.get("kind", "")),
        )
        dashed = dashed_side_edges and str(edge.get("kind", "")) == "side"
        edge_bbox_map[str(edge["edge_id"])] = _draw_polyline_arrow(
            draw,
            points=points,
            fill=palette["edge"],
            width=int(render_params.edge_width_px),
            head_length_px=float(render_params.arrow_head_length_px),
            head_width_px=float(render_params.arrow_head_width_px),
            dashed=bool(dashed),
        )
        rounded_points = [_round_point((x, y)) for x, y in points]
        edge_segment_map[str(edge["edge_id"])] = [list(rounded_points[0]), list(rounded_points[-1])]
        edge_polyline_map[str(edge["edge_id"])] = [list(point) for point in rounded_points]
        if str(edge.get("label", "")).strip():
            label_bbox = _edge_label_bbox(
                points=points,
                text=str(edge["label"]),
                layout_variant=str(case.layout_variant),
                edge_index=int(edge_index),
            )
            label_bbox = _avoid_label_collisions(
                label_bbox,
                occupied_boxes=occupied_label_boxes,
                bounds=label_bounds,
                vertical=bool(vertical),
                edge_index=int(edge_index),
            )
            edge_label_bbox_map[str(edge["edge_id"])] = _round_box(label_bbox)
            edge_label_text_bbox_map[str(edge["edge_id"])] = _draw_edge_label(
                draw,
                bbox=label_bbox,
                text=str(edge["label"]),
                palette=palette,
                render_params=render_params,
            )
            occupied_label_boxes.append(_round_box(label_bbox))

    node_bbox_map: Dict[str, list[float]] = {}
    node_label_bbox_map: Dict[str, list[float]] = {}
    node_status_bbox_map: Dict[str, list[float]] = {}
    for node in nodes:
        label_bbox, status_bbox = _draw_node(
            draw,
            node=node,
            palette=palette,
            render_params=render_params,
        )
        node_bbox_map[str(node["bbox_id"])] = list(node["bbox"])
        node_label_bbox_map[str(node["bbox_id"])] = list(label_bbox)
        if any(float(value) for value in status_bbox):
            node_status_bbox_map[str(node["bbox_id"])] = list(status_bbox)

    render_map = {
        "image_id": "img0",
        "panel_bbox_px": round_diagram_bbox(case.panel_bbox),
        "title_bbox_px": round_diagram_bbox(case.title_bbox),
        "content_bbox_px": round_diagram_bbox(case.content_bbox),
        "lane_bboxes_px": {str(key): _round_box(value) for key, value in case.lane_bboxes.items()},
        "node_bboxes_px": node_bbox_map,
        "node_label_bboxes_px": node_label_bbox_map,
        "node_status_bboxes_px": node_status_bbox_map,
        "edge_bboxes_px": edge_bbox_map,
        "edge_point_pairs_px": edge_segment_map,
        "edge_segments_px": edge_segment_map,
        "edge_polylines_px": edge_polyline_map,
        "edge_label_bboxes_px": edge_label_bbox_map,
        "edge_label_text_bboxes_px": edge_label_text_bbox_map,
    }
    return RenderedProcessFlow(image=image, render_map=render_map)
