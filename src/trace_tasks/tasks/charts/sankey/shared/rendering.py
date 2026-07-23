"""Rendering helpers for Sankey flow chart tasks."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.tasks.charts.shared.information_style import prepare_chart_information_scene
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.shared.bbox_projection import round_bbox as _round_bbox
from trace_tasks.tasks.shared.drawing import draw_centered_text
from trace_tasks.tasks.shared.text_rendering import fit_font_to_box, load_font, temporary_default_font_family

from .defaults import (
    FLOW_PALETTE_RGB,
    POST_IMAGE_NOISE_DEFAULTS,
    resolve_render_params,
    sample_font_family,
)
from .sampling import node_dict, path_dict
from .state import BBox, Point, FlowRenderParams, RenderedSankey, SankeyDataset, SankeyRenderResult


def clamp_unit_interval(value: float) -> float:
    return max(0.0, min(1.0, float(value)))

def _clamp_bbox(bbox: Sequence[float], *, width: int, height: int) -> List[float]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    x0 = max(0.0, min(float(width), x0))
    y0 = max(0.0, min(float(height), y0))
    x1 = max(0.0, min(float(width), x1))
    y1 = max(0.0, min(float(height), y1))
    if x1 <= x0:
        x1 = min(float(width), x0 + 1.0)
    if y1 <= y0:
        y1 = min(float(height), y0 + 1.0)
    return _round_bbox((x0, y0, x1, y1))


def _cubic_point(p0: Point, p1: Point, p2: Point, p3: Point, t: float) -> Point:
    inv = 1.0 - float(t)
    x = (inv**3 * p0[0]) + (3 * inv * inv * t * p1[0]) + (3 * inv * t * t * p2[0]) + (t**3 * p3[0])
    y = (inv**3 * p0[1]) + (3 * inv * inv * t * p1[1]) + (3 * inv * t * t * p2[1]) + (t**3 * p3[1])
    return (float(x), float(y))


def _curve_points(start: Point, end: Point, *, steps: int = 36, lane_offset_y: float = 0.0) -> List[Point]:
    dx = float(end[0] - start[0])
    c1 = (float(start[0] + (0.42 * dx)), float(start[1] + float(lane_offset_y)))
    c2 = (float(end[0] - (0.42 * dx)), float(end[1] + float(lane_offset_y)))
    return [
        _cubic_point(start, c1, c2, end, float(index) / float(max(1, int(steps) - 1)))
        for index in range(int(steps))
    ]


def _curve_bbox(points: Sequence[Point], *, stroke_width: int, canvas_width: int, canvas_height: int) -> List[float]:
    pad = max(2.0, 0.5 * float(stroke_width) + 2.0)
    return _clamp_bbox(
        (
            min(point[0] for point in points) - pad,
            min(point[1] for point in points) - pad,
            max(point[0] for point in points) + pad,
            max(point[1] for point in points) + pad,
        ),
        width=int(canvas_width),
        height=int(canvas_height),
    )


def _node_bboxes(
    *,
    nodes: Sequence[Mapping[str, Any]],
    x_center: float,
    top: float,
    bottom: float,
    render_params: FlowRenderParams,
) -> Dict[str, BBox]:
    count = len(nodes)
    if int(count) <= 0:
        return {}
    usable_top = float(top + 28.0)
    usable_bottom = float(bottom - 26.0)
    if int(count) == 1:
        centers = [0.5 * (usable_top + usable_bottom)]
    else:
        centers = [
            float(usable_top + ((usable_bottom - usable_top) * float(index) / float(count - 1)))
            for index in range(int(count))
        ]
    half_w = 0.5 * float(render_params.node_width_px)
    half_h = 0.5 * float(render_params.node_height_px)
    return {
        str(node["node_id"]): (
            float(x_center - half_w),
            float(centers[index] - half_h),
            float(x_center + half_w),
            float(centers[index] + half_h),
        )
        for index, node in enumerate(nodes)
    }


def _port_y(
    *,
    node_bbox: Sequence[float],
    ordered_path_ids: Sequence[str],
    path_id: str,
    render_params: FlowRenderParams,
) -> float:
    center_y = 0.5 * (float(node_bbox[1]) + float(node_bbox[3]))
    total = max(1, len(ordered_path_ids))
    if int(total) == 1:
        return float(center_y)
    index = [str(item) for item in ordered_path_ids].index(str(path_id))
    node_height = float(node_bbox[3] - node_bbox[1])
    available_span = max(0.0, node_height - 12.0)
    min_separation = max(
        14.0,
        float(render_params.port_separation_px),
        float(render_params.max_flow_width_px) + 10.0,
    )
    span = min(float(available_span), max(14.0, float(total - 1) * float(min_separation)))
    offset = (-0.5 * span) + (span * float(index) / float(total - 1))
    return float(center_y + offset)


def _flow_width(value: int, *, render_params: FlowRenderParams, value_min: int, value_max: int) -> int:
    if int(value_max) <= int(value_min):
        return int(render_params.min_flow_width_px)
    norm = (float(value) - float(value_min)) / float(value_max - value_min)
    width = float(render_params.min_flow_width_px) + (clamp_unit_interval(norm) * float(render_params.max_flow_width_px - render_params.min_flow_width_px))
    return max(1, int(round(width)))


def _value_label_size(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    render_params: FlowRenderParams,
) -> Tuple[float, float]:
    font = load_font(int(render_params.value_label_font_size_px), bold=False)
    text_bbox = draw.textbbox((0, 0), str(text), font=font)
    text_width = float(text_bbox[2] - text_bbox[0])
    text_height = float(text_bbox[3] - text_bbox[1])
    return (max(36.0, float(text_width + 18.0)), max(26.0, float(text_height + 12.0)))


def _value_label_bbox(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Point,
    render_params: FlowRenderParams,
) -> BBox:
    label_width, label_height = _value_label_size(draw, text=str(text), render_params=render_params)
    cx, cy = float(center[0]), float(center[1])
    return (
        float(cx - (0.5 * label_width)),
        float(cy - (0.5 * label_height)),
        float(cx + (0.5 * label_width)),
        float(cy + (0.5 * label_height)),
    )


def _draw_value_label(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Point,
    render_params: FlowRenderParams,
) -> List[float]:
    font = load_font(int(render_params.value_label_font_size_px), bold=False)
    bbox = _value_label_bbox(draw, text=str(text), center=center, render_params=render_params)
    cx, cy = float(center[0]), float(center[1])
    draw.rounded_rectangle(
        bbox,
        radius=8,
        fill=tuple(int(channel) for channel in render_params.value_label_fill_rgb),
        outline=tuple(int(channel) for channel in render_params.value_label_border_rgb),
        width=1,
    )
    draw_centered_text(
        draw,
        text=str(text),
        center=(float(cx), float(cy)),
        font=font,
        fill=render_params.value_label_text_rgb,
        stroke_fill=render_params.value_label_fill_rgb,
        stroke_width=0,
    )
    return _round_bbox(bbox)


def _resolve_value_label_centers(
    draw: ImageDraw.ImageDraw,
    *,
    label_specs: Sequence[Mapping[str, Any]],
    plot_bbox: Sequence[float],
    render_params: FlowRenderParams,
) -> Dict[str, Point]:
    """Place value labels in each link column while preserving non-overlap vertically."""

    resolved: Dict[str, Point] = {}
    gap = float(render_params.value_label_gap_px)
    top_limit = float(plot_bbox[1]) + max(4.0, gap)
    bottom_limit = float(plot_bbox[3]) - max(4.0, gap)
    available_height = max(1.0, float(bottom_limit - top_limit))

    for segment_kind in ("source_middle", "middle_target"):
        group = [
            dict(spec)
            for spec in label_specs
            if str(spec["segment_kind"]) == str(segment_kind)
        ]
        if not group:
            continue
        group = sorted(
            group,
            key=lambda spec: (
                float(spec["desired_center"][1]),
                float(spec["desired_center"][0]),
                str(spec["segment_id"]),
            ),
        )
        heights = [
            _value_label_size(draw, text=str(spec["text"]), render_params=render_params)[1]
            for spec in group
        ]
        if len(group) > 1:
            total_label_height = sum(float(height) for height in heights)
            effective_gap = min(float(gap), max(0.0, (available_height - total_label_height) / float(len(group) - 1)))
        else:
            effective_gap = 0.0

        placed: List[Tuple[Dict[str, Any], float, float]] = []
        previous_bottom = float("-inf")
        for spec, height in zip(group, heights):
            half_height = 0.5 * float(height)
            desired_x, desired_y = [float(value) for value in spec["desired_center"]]
            min_center_y = float(top_limit + half_height)
            max_center_y = float(bottom_limit - half_height)
            desired_y = max(min_center_y, min(max_center_y, float(desired_y)))
            if placed:
                desired_y = max(float(desired_y), float(previous_bottom + effective_gap + half_height))
            placed.append((spec, float(desired_y), float(height)))
            previous_bottom = float(desired_y + half_height)

        overflow = max(0.0, float(previous_bottom - bottom_limit))
        if overflow > 0.0:
            placed = [(spec, float(center_y - overflow), height) for spec, center_y, height in placed]

        first_top = float(placed[0][1] - (0.5 * placed[0][2]))
        underflow = max(0.0, float(top_limit - first_top))
        if underflow > 0.0:
            placed = [(spec, float(center_y + underflow), height) for spec, center_y, height in placed]

        for spec, center_y, _height in placed:
            desired_x = float(spec["desired_center"][0])
            resolved[str(spec["segment_id"])] = (float(desired_x), float(center_y))

    return resolved


def _sorted_paths_for_port(
    paths: Sequence[Mapping[str, Any]],
    *,
    key_fields: Sequence[str],
) -> List[str]:
    return [
        str(path["path_id"])
        for path in sorted(
            paths,
            key=lambda path: tuple(str(path[field]) for field in key_fields),
        )
    ]


def _segment_lane_offsets(
    paths: Sequence[Mapping[str, Any]],
    *,
    segment_kind: str,
    render_params: FlowRenderParams,
) -> Dict[str, float]:
    """Fan out parallel bands that connect the same pair of Sankey nodes."""

    if int(render_params.shared_pair_lane_gap_px) <= 0:
        return {str(path["path_id"]): 0.0 for path in paths}

    grouped: Dict[Tuple[str, str], List[Mapping[str, Any]]] = {}
    for path in paths:
        if str(segment_kind) == "source_middle":
            key = (str(path["source_id"]), str(path["middle_id"]))
        else:
            key = (str(path["middle_id"]), str(path["target_id"]))
        grouped.setdefault(key, []).append(path)

    offsets: Dict[str, float] = {}
    for group in grouped.values():
        if len(group) <= 1:
            offsets[str(group[0]["path_id"])] = 0.0
            continue
        if str(segment_kind) == "source_middle":
            ordered = sorted(group, key=lambda path: (str(path["target_label"]), str(path["path_id"])))
        else:
            ordered = sorted(group, key=lambda path: (str(path["source_label"]), str(path["path_id"])))
        center = 0.5 * float(len(ordered) - 1)
        for index, path in enumerate(ordered):
            offsets[str(path["path_id"])] = (float(index) - center) * float(render_params.shared_pair_lane_gap_px)
    return offsets


def _render_sankey(
    background: Image.Image,
    *,
    scene_title: str,
    sources: Sequence[Mapping[str, Any]],
    middles: Sequence[Mapping[str, Any]],
    targets: Sequence[Mapping[str, Any]],
    paths: Sequence[Mapping[str, Any]],
    render_params: FlowRenderParams,
    value_min: int,
    value_max: int,
) -> RenderedSankey:
    """Render the three-column Sankey grammar and record node, segment, and label geometry."""

    base = background.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    flow_draw = ImageDraw.Draw(overlay)

    outer = float(render_params.outer_margin_px)
    offset_x = float(render_params.layout_offset_x_px)
    offset_y = float(render_params.layout_offset_y_px)
    panel_bbox: BBox = (
        outer + offset_x,
        outer + offset_y,
        float(render_params.canvas_width) - outer + offset_x,
        float(render_params.canvas_height) - outer + offset_y,
    )
    title_bbox: BBox = (
        panel_bbox[0] + float(render_params.panel_padding_px),
        panel_bbox[1] + 10.0,
        panel_bbox[2] - float(render_params.panel_padding_px),
        panel_bbox[1] + float(render_params.title_band_height_px),
    )
    plot_bbox: BBox = (
        panel_bbox[0] + float(render_params.panel_padding_px),
        title_bbox[3] + 18.0,
        panel_bbox[2] - float(render_params.panel_padding_px),
        panel_bbox[3] - float(render_params.panel_padding_px),
    )
    x_left = float(plot_bbox[0] + 84.0)
    x_middle = float(0.5 * (plot_bbox[0] + plot_bbox[2]))
    x_right = float(plot_bbox[2] - 84.0)
    node_bbox_map_raw: Dict[str, BBox] = {}
    node_bbox_map_raw.update(
        _node_bboxes(
            nodes=sources,
            x_center=float(x_left),
            top=float(plot_bbox[1]),
            bottom=float(plot_bbox[3]),
            render_params=render_params,
        )
    )
    node_bbox_map_raw.update(
        _node_bboxes(
            nodes=middles,
            x_center=float(x_middle),
            top=float(plot_bbox[1]),
            bottom=float(plot_bbox[3]),
            render_params=render_params,
        )
    )
    node_bbox_map_raw.update(
        _node_bboxes(
            nodes=targets,
            x_center=float(x_right),
            top=float(plot_bbox[1]),
            bottom=float(plot_bbox[3]),
            render_params=render_params,
        )
    )

    paths_by_id = {str(path["path_id"]): dict(path) for path in paths}
    source_right: Dict[str, List[str]] = {}
    middle_left: Dict[str, List[str]] = {}
    middle_right: Dict[str, List[str]] = {}
    target_left: Dict[str, List[str]] = {}
    for source in sources:
        source_id = str(source["node_id"])
        source_paths = [dict(path) for path in paths if str(path["source_id"]) == source_id]
        source_right[source_id] = _sorted_paths_for_port(source_paths, key_fields=("middle_label", "target_label", "path_id"))
    for middle in middles:
        middle_id = str(middle["node_id"])
        in_paths = [dict(path) for path in paths if str(path["middle_id"]) == middle_id]
        out_paths = list(in_paths)
        middle_left[middle_id] = _sorted_paths_for_port(in_paths, key_fields=("source_label", "target_label", "path_id"))
        middle_right[middle_id] = _sorted_paths_for_port(out_paths, key_fields=("target_label", "source_label", "path_id"))
    for target in targets:
        target_id = str(target["node_id"])
        target_paths = [dict(path) for path in paths if str(path["target_id"]) == target_id]
        target_left[target_id] = _sorted_paths_for_port(target_paths, key_fields=("middle_label", "source_label", "path_id"))

    source_middle_lane_offsets = _segment_lane_offsets(
        paths,
        segment_kind="source_middle",
        render_params=render_params,
    )
    middle_target_lane_offsets = _segment_lane_offsets(
        paths,
        segment_kind="middle_target",
        render_params=render_params,
    )

    segment_bbox_map: Dict[str, List[float]] = {}
    segment_label_bbox_map: Dict[str, List[float]] = {}
    segment_center_map: Dict[str, List[float]] = {}
    entities: List[Dict[str, Any]] = []

    for index, path in enumerate(paths):
        path_id = str(path["path_id"])
        color = FLOW_PALETTE_RGB[int(index) % len(FLOW_PALETTE_RGB)]
        source_bbox = node_bbox_map_raw[str(path["source_id"])]
        middle_bbox = node_bbox_map_raw[str(path["middle_id"])]
        target_bbox = node_bbox_map_raw[str(path["target_id"])]
        first_start = (
            float(source_bbox[2]),
            _port_y(
                node_bbox=source_bbox,
                ordered_path_ids=source_right[str(path["source_id"])],
                path_id=path_id,
                render_params=render_params,
            ),
        )
        first_end = (
            float(middle_bbox[0]),
            _port_y(
                node_bbox=middle_bbox,
                ordered_path_ids=middle_left[str(path["middle_id"])],
                path_id=path_id,
                render_params=render_params,
            ),
        )
        second_start = (
            float(middle_bbox[2]),
            _port_y(
                node_bbox=middle_bbox,
                ordered_path_ids=middle_right[str(path["middle_id"])],
                path_id=path_id,
                render_params=render_params,
            ),
        )
        second_end = (
            float(target_bbox[0]),
            _port_y(
                node_bbox=target_bbox,
                ordered_path_ids=target_left[str(path["target_id"])],
                path_id=path_id,
                render_params=render_params,
            ),
        )
        for segment_kind, start, end, value, label_t in (
            ("source_middle", first_start, first_end, int(path["first_value"]), float(render_params.source_middle_label_t)),
            ("middle_target", second_start, second_end, int(path["second_value"]), float(render_params.middle_target_label_t)),
        ):
            segment_id = f"{path_id}:{segment_kind}"
            lane_offset = (
                float(source_middle_lane_offsets.get(str(path_id), 0.0))
                if str(segment_kind) == "source_middle"
                else float(middle_target_lane_offsets.get(str(path_id), 0.0))
            )
            points = _curve_points(start, end, lane_offset_y=float(lane_offset))
            stroke_width = _flow_width(
                int(value),
                render_params=render_params,
                value_min=int(value_min),
                value_max=int(value_max),
            )
            flow_draw.line(
                points,
                fill=(int(color[0]), int(color[1]), int(color[2]), int(render_params.flow_alpha)),
                width=int(stroke_width),
                joint="curve",
            )
            bbox = _curve_bbox(
                points,
                stroke_width=int(stroke_width),
                canvas_width=int(render_params.canvas_width),
                canvas_height=int(render_params.canvas_height),
            )
            label_center = _cubic_point(
                points[0],
                points[max(1, len(points) // 3)],
                points[min(len(points) - 2, (2 * len(points)) // 3)],
                points[-1],
                float(label_t),
            )
            segment_bbox_map[str(segment_id)] = list(bbox)
            segment_center_map[str(segment_id)] = [round(float(label_center[0]), 3), round(float(label_center[1]), 3)]

    image = Image.alpha_composite(base, overlay).convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(panel_bbox, radius=16, fill=render_params.panel_fill_rgb, outline=render_params.panel_border_rgb, width=2)
    draw.rounded_rectangle(plot_bbox, radius=12, fill=render_params.plot_fill_rgb, outline=render_params.panel_border_rgb, width=1)
    image_with_flows = Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")
    draw = ImageDraw.Draw(image_with_flows)
    title_text_bbox = draw_centered_text(
        draw,
        text=str(scene_title),
        center=(0.5 * (title_bbox[0] + title_bbox[2]), 0.5 * (title_bbox[1] + title_bbox[3])),
        font=load_font(int(render_params.title_font_size_px), bold=False),
        fill=render_params.title_color_rgb,
        stroke_fill=render_params.panel_fill_rgb,
        stroke_width=0,
    )

    value_label_specs: List[Dict[str, Any]] = []
    for path in paths:
        path_id = str(path["path_id"])
        for segment_kind, value in (
            ("source_middle", int(path["first_value"])),
            ("middle_target", int(path["second_value"])),
        ):
            segment_id = f"{path_id}:{segment_kind}"
            value_label_specs.append(
                {
                    "segment_id": str(segment_id),
                    "segment_kind": str(segment_kind),
                    "text": str(value),
                    "desired_center": tuple(float(value) for value in segment_center_map[str(segment_id)]),
                }
            )

    resolved_label_centers = _resolve_value_label_centers(
        draw,
        label_specs=value_label_specs,
        plot_bbox=plot_bbox,
        render_params=render_params,
    )

    for path in paths:
        path_id = str(path["path_id"])
        for segment_kind, value in (
            ("source_middle", int(path["first_value"])),
            ("middle_target", int(path["second_value"])),
        ):
            segment_id = f"{path_id}:{segment_kind}"
            label_center = tuple(float(value) for value in resolved_label_centers[str(segment_id)])
            segment_center_map[str(segment_id)] = [round(float(label_center[0]), 3), round(float(label_center[1]), 3)]
            segment_label_bbox_map[str(segment_id)] = _draw_value_label(
                draw,
                text=str(value),
                center=label_center,
                render_params=render_params,
            )
            entities.append(
                {
                    "entity_id": str(segment_id),
                    "entity_type": "sankey_segment",
                    "bbox_xyxy": list(segment_bbox_map[str(segment_id)]),
                    "attrs": {
                        "path_id": str(path_id),
                        "segment_kind": str(segment_kind),
                        "value": int(value),
                        "label_bbox_xyxy": list(segment_label_bbox_map[str(segment_id)]),
                    },
                }
            )

    node_bbox_map: Dict[str, List[float]] = {}
    node_label_bbox_map: Dict[str, List[float]] = {}
    all_nodes = [*sources, *middles, *targets]
    for node in all_nodes:
        node_id = str(node["node_id"])
        bbox = node_bbox_map_raw[node_id]
        node_bbox_map[node_id] = _round_bbox(bbox)
        draw.rounded_rectangle(
            bbox,
            radius=10,
            fill=tuple(int(channel) for channel in render_params.node_fill_rgb),
            outline=tuple(int(channel) for channel in render_params.node_border_rgb),
            width=max(1, int(render_params.node_border_width_px)),
        )
        label_font = fit_font_to_box(
            draw,
            text=str(node["label"]),
            max_width=float(bbox[2] - bbox[0] - 12.0),
            max_height=float(bbox[3] - bbox[1] - 8.0),
            bold=False,
            min_size_px=12,
            max_size_px=int(render_params.node_label_font_size_px),
            fill_ratio=0.9,
        )
        label_bbox = draw_centered_text(
            draw,
            text=str(node["label"]),
            center=(0.5 * (bbox[0] + bbox[2]), 0.5 * (bbox[1] + bbox[3])),
            font=label_font,
            fill=render_params.node_text_rgb,
            stroke_fill=render_params.node_fill_rgb,
            stroke_width=0,
        )
        node_label_bbox_map[node_id] = list(label_bbox)
        entities.append(
            {
                "entity_id": str(node_id),
                "entity_type": "sankey_node",
                "bbox_xyxy": list(node_bbox_map[node_id]),
                "attrs": {
                    "label": str(node["label"]),
                    "column": str(node["column"]),
                },
            }
        )

    entities.insert(0, {"entity_id": "flow_panel", "entity_type": "flow_panel", "bbox_xyxy": _round_bbox(panel_bbox)})
    entities.insert(
        1,
        {
            "entity_id": "flow_title",
            "entity_type": "flow_title",
            "bbox_xyxy": list(title_text_bbox),
            "attrs": {"title": str(scene_title)},
        },
    )
    return RenderedSankey(
        image=image_with_flows,
        entities=tuple(dict(item) for item in entities),
        panel_bbox_px=_round_bbox(panel_bbox),
        title_bbox_px=list(title_text_bbox),
        plot_bbox_px=_round_bbox(plot_bbox),
        node_bbox_map=dict(node_bbox_map),
        node_label_bbox_map=dict(node_label_bbox_map),
        segment_bbox_map=dict(segment_bbox_map),
        segment_label_bbox_map=dict(segment_label_bbox_map),
        segment_center_map=dict(segment_center_map),
    )


def render_sankey_dataset(
    *,
    dataset: SankeyDataset,
    params: Mapping[str, Any],
    instance_seed: int,
) -> SankeyRenderResult:
    """Render a task-bound Sankey dataset with sampled chart font, background, and noise."""

    frame = dataset.frame
    render_style_params = {**dict(params), "_render_style_seed": int(instance_seed)}
    resolved_params = resolve_render_params(render_style_params)
    render_params, background, background_meta, information_style_meta = prepare_chart_information_scene(
        instance_seed=int(instance_seed),
        params=params,
        scene_id="sankey",
        render_params=resolved_params,
        protected_colors=FLOW_PALETTE_RGB,
    )
    chart_font_family = sample_font_family(int(instance_seed), params)
    with temporary_default_font_family(str(chart_font_family)):
        rendered_scene = _render_sankey(
            background,
            scene_title=str(frame.scene_title),
            sources=[node_dict(node) for node in frame.sources],
            middles=[node_dict(node) for node in frame.middles],
            targets=[node_dict(node) for node in frame.targets],
            paths=[path_dict(path) for path in frame.paths],
            render_params=render_params,
            value_min=int(frame.value_min),
            value_max=int(frame.value_max),
        )
    image, post_noise_meta = apply_post_image_noise(
        rendered_scene.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return SankeyRenderResult(
        image=image,
        rendered_scene=rendered_scene,
        render_params=render_params,
        background_meta={**dict(background_meta), "information_scene_style": dict(information_style_meta)},
        post_noise_meta=dict(post_noise_meta),
        chart_font_family=str(chart_font_family),
    )


__all__ = ["render_sankey_dataset"]
