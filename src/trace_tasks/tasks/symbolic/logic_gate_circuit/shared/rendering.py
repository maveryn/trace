"""Renderer for symbolic Boolean logic-gate circuit scenes."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from PIL import Image, ImageDraw

from .....core.visual.noise import apply_post_image_noise
from ....shared.text_rendering import load_font
from ...shared.drawing import draw_centered_text, draw_rounded_rect
from ...shared.scene_style import SymbolicSceneStyle

from .state import CandidateAssignmentSpec, LogicCircuitSpec, LogicGateRenderBundle, LogicGateRenderParams, RenderedLogicGateScene, SCENE_ID
from .styles import resolve_background, resolve_render_params


def _rounded_bbox(values: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in values]


def _rounded_point(x: float, y: float) -> list[float]:
    return [round(float(x), 3), round(float(y), 3)]


def _wire_polyline_points(
    *,
    start: Sequence[float],
    end: Sequence[float],
    lane_x: float | None = None,
) -> tuple[tuple[float, float], ...]:
    sx, sy = float(start[0]), float(start[1])
    ex, ey = float(end[0]), float(end[1])
    if abs(sy - ey) <= 0.5 or abs(ex - sx) <= 0.5:
        return ((sx, sy), (ex, ey))

    if lane_x is None:
        lane_x = sx + (0.5 * (ex - sx))
    if ex > sx:
        min_lane = sx + 10.0
        max_lane = ex - 10.0
        if max_lane > min_lane:
            lane_x = max(min_lane, min(max_lane, float(lane_x)))
        else:
            lane_x = sx + (0.5 * (ex - sx))
    else:
        min_lane = ex + 10.0
        max_lane = sx - 10.0
        if max_lane > min_lane:
            lane_x = max(min_lane, min(max_lane, float(lane_x)))
        else:
            lane_x = sx + (0.5 * (ex - sx))

    points = ((sx, sy), (float(lane_x), sy), (float(lane_x), ey), (ex, ey))
    deduped: list[tuple[float, float]] = []
    for point in points:
        if not deduped or abs(deduped[-1][0] - point[0]) > 0.5 or abs(deduped[-1][1] - point[1]) > 0.5:
            deduped.append(point)
    return tuple(deduped)


def _draw_wire(
    draw: ImageDraw.ImageDraw,
    *,
    points: Sequence[Sequence[float]],
    rgb: Sequence[int],
    width: int,
) -> None:
    color = tuple(int(value) for value in rgb)
    line_width = int(width)
    flattened: list[float] = []
    for point in points:
        flattened.extend((float(point[0]), float(point[1])))
    draw.line(tuple(flattened), fill=color, width=line_width, joint="curve")


def _wire_lane_x(
    *,
    start: Sequence[float],
    end: Sequence[float],
    input_index: int,
    input_count: int,
) -> float:
    """Choose a per-input vertical routing lane inside the available gap."""

    sx = float(start[0])
    ex = float(end[0])
    span = ex - sx
    if span <= 24.0:
        return sx + (0.5 * span)
    inner_left = sx + 12.0
    inner_right = ex - 10.0
    if inner_right <= inner_left:
        return sx + (0.5 * span)
    fraction = float(int(input_index) + 1) / float(max(1, int(input_count)) + 1)
    return inner_left + (fraction * (inner_right - inner_left))


def _spread_y_positions(
    proposed: Sequence[tuple[str, float]],
    *,
    lower: float,
    upper: float,
    min_gap: float,
) -> dict[str, float]:
    """Spread same-level gates vertically while preserving their order."""

    ordered = sorted(((str(key), float(value)) for key, value in proposed), key=lambda item: (item[1], item[0]))
    if not ordered:
        return {}
    if len(ordered) == 1:
        key, value = ordered[0]
        return {key: max(float(lower), min(float(upper), float(value)))}

    available_span = max(1.0, float(upper) - float(lower))
    gap = min(float(min_gap), available_span / float(len(ordered) - 1))
    values = [max(float(lower), min(float(upper), float(value))) for _key, value in ordered]
    for index in range(1, len(values)):
        values[index] = max(values[index], values[index - 1] + gap)
    overflow = values[-1] - float(upper)
    if overflow > 0:
        values = [value - overflow for value in values]
    for index in range(len(values) - 2, -1, -1):
        values[index] = min(values[index], values[index + 1] - gap)
    underflow = float(lower) - values[0]
    if underflow > 0:
        values = [value + underflow for value in values]
    return {key: round(float(value), 6) for (key, _proposed_y), value in zip(ordered, values)}


def _gate_input_ports(gate_bbox: Sequence[float], count: int) -> tuple[tuple[float, float], ...]:
    left, top, _right, bottom = [float(value) for value in gate_bbox]
    if int(count) == 1:
        return ((left, 0.5 * (top + bottom)),)
    return ((left, top + 0.25 * (bottom - top)), (left, top + 0.75 * (bottom - top)))


def _draw_output_node(
    draw: ImageDraw.ImageDraw,
    *,
    center: Sequence[float],
    params: LogicGateRenderParams,
    style: SymbolicSceneStyle,
) -> None:
    cx, cy = float(center[0]), float(center[1])
    radius = int(params.node_radius_px)
    draw.ellipse(
        (cx - radius, cy - radius, cx + radius, cy + radius),
        fill=tuple(int(value) for value in style.panel_accent_rgb),
        outline=tuple(int(value) for value in style.text_rgb),
        width=1,
    )


def _quadratic_curve_points(
    start: Sequence[float],
    control: Sequence[float],
    end: Sequence[float],
    *,
    steps: int = 16,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for index in range(int(steps) + 1):
        t = float(index) / float(max(1, int(steps)))
        inv = 1.0 - t
        x = (inv * inv * float(start[0])) + (2.0 * inv * t * float(control[0])) + (t * t * float(end[0]))
        y = (inv * inv * float(start[1])) + (2.0 * inv * t * float(control[1])) + (t * t * float(end[1]))
        points.append((float(x), float(y)))
    return points


def _draw_polyline(
    draw: ImageDraw.ImageDraw,
    points: Sequence[Sequence[float]],
    *,
    fill: Sequence[int],
    width: int,
) -> None:
    flattened: list[float] = []
    for point in points:
        flattened.extend((float(point[0]), float(point[1])))
    draw.line(tuple(flattened), fill=tuple(int(value) for value in fill), width=int(width), joint="curve")


def _draw_and_symbol(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    fill_rgb: Sequence[int],
    stroke_rgb: Sequence[int],
    width: int,
) -> None:
    left, top, right, bottom = [float(value) for value in bbox]
    mid_x = left + (0.52 * (right - left))
    center_y = 0.5 * (top + bottom)
    radius_x = right - mid_x
    radius_y = 0.5 * (bottom - top)
    arc_points = [
        (
            mid_x + (radius_x * math.cos(-0.5 * math.pi + (math.pi * float(index) / 24.0))),
            center_y + (radius_y * math.sin(-0.5 * math.pi + (math.pi * float(index) / 24.0))),
        )
        for index in range(25)
    ]
    fill_points = [(left, top), (mid_x, top), *arc_points, (mid_x, bottom), (left, bottom)]
    draw.polygon(fill_points, fill=tuple(int(value) for value in fill_rgb))
    _draw_polyline(draw, [(left, top), (mid_x, top), *arc_points, (mid_x, bottom), (left, bottom), (left, top)], fill=stroke_rgb, width=width)


def _draw_or_symbol(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    fill_rgb: Sequence[int],
    stroke_rgb: Sequence[int],
    width: int,
    xor: bool,
) -> None:
    left, top, right, bottom = [float(value) for value in bbox]
    w = right - left
    h = bottom - top
    cy = 0.5 * (top + bottom)
    top_start = (left + 0.19 * w, top + 0.05 * h)
    bottom_start = (left + 0.19 * w, bottom - 0.05 * h)
    output = (right, cy)
    top_curve = _quadratic_curve_points(top_start, (left + 0.78 * w, top - 0.02 * h), output, steps=18)
    bottom_curve = _quadratic_curve_points(output, (left + 0.78 * w, bottom + 0.02 * h), bottom_start, steps=18)
    back_curve = _quadratic_curve_points(bottom_start, (left + 0.42 * w, cy), top_start, steps=18)
    fill_points = [*top_curve, *bottom_curve[1:], *back_curve[1:]]
    draw.polygon(fill_points, fill=tuple(int(value) for value in fill_rgb))
    _draw_polyline(draw, top_curve, fill=stroke_rgb, width=width)
    _draw_polyline(draw, bottom_curve, fill=stroke_rgb, width=width)
    _draw_polyline(draw, back_curve, fill=stroke_rgb, width=width)
    input_y_values = (top + 0.25 * h, top + 0.75 * h)
    for input_y in input_y_values:
        _draw_polyline(draw, [(left, input_y), (left + 0.23 * w, input_y)], fill=stroke_rgb, width=width)
    if bool(xor):
        xor_curve = _quadratic_curve_points(
            (left + 0.05 * w, bottom - 0.06 * h),
            (left + 0.28 * w, cy),
            (left + 0.05 * w, top + 0.06 * h),
            steps=18,
        )
        _draw_polyline(draw, xor_curve, fill=stroke_rgb, width=width)


def _draw_not_symbol(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: Sequence[float],
    fill_rgb: Sequence[int],
    stroke_rgb: Sequence[int],
    width: int,
) -> None:
    left, top, right, bottom = [float(value) for value in bbox]
    cy = 0.5 * (top + bottom)
    points = [(left, top), (left, bottom), (right, cy)]
    draw.polygon(points, fill=tuple(int(value) for value in fill_rgb))
    _draw_polyline(draw, [points[0], points[1], points[2], points[0]], fill=stroke_rgb, width=width)


def _draw_gate_symbol(
    draw: ImageDraw.ImageDraw,
    *,
    gate_type: str,
    bbox: Sequence[float],
    style: SymbolicSceneStyle,
    width: int,
) -> None:
    normalized = str(gate_type).upper()
    fill_rgb = tuple(int(value) for value in style.panel_fill_rgb)
    stroke_rgb = tuple(int(value) for value in style.text_rgb)
    if normalized in {"AND", "NAND"}:
        _draw_and_symbol(draw, bbox=bbox, fill_rgb=fill_rgb, stroke_rgb=stroke_rgb, width=int(width))
    elif normalized in {"OR", "NOR", "XOR"}:
        _draw_or_symbol(
            draw,
            bbox=bbox,
            fill_rgb=fill_rgb,
            stroke_rgb=stroke_rgb,
            width=int(width),
            xor=normalized == "XOR",
        )
    elif normalized == "NOT":
        _draw_not_symbol(draw, bbox=bbox, fill_rgb=fill_rgb, stroke_rgb=stroke_rgb, width=int(width))
    else:
        raise ValueError(f"unsupported logic-gate symbol: {gate_type}")


def _draw_logic_circuit(
    draw: ImageDraw.ImageDraw,
    *,
    circuit: LogicCircuitSpec,
    bbox: Sequence[float],
    params: LogicGateRenderParams,
    style: SymbolicSceneStyle,
    show_fixed_input_values: bool,
    title: str | None = None,
) -> tuple[dict[str, Any], dict[str, list[float]], dict[str, list[float]], dict[str, list[float]], tuple[dict[str, Any], ...]]:
    """Draw one circuit panel and expose reusable geometry maps.

    The invariant is that every returned output point and item bbox is already
    in final panel pixel coordinates, so public tasks can bind annotation from
    these maps without reinterpreting circuit layout internals.
    """

    left, top, right, bottom = [float(value) for value in bbox]
    item_bboxes: dict[str, list[float]] = {str(circuit.item_id): _rounded_bbox(bbox)}
    output_points: dict[str, list[float]] = {}
    signal_points: dict[str, list[float]] = {}

    draw_rounded_rect(
        draw,
        (left, top, right, bottom),
        radius=int(params.card_corner_radius_px),
        fill=tuple(int(value) for value in style.panel_fill_rgb),
        outline=tuple(int(value) for value in style.panel_border_rgb),
        width=int(params.card_border_width_px),
    )

    title_font = load_font(int(params.label_font_size_px), bold=True)
    small_font = load_font(int(params.small_font_size_px), bold=True)
    if str(circuit.label).strip():
        label_bbox = draw_centered_text(
            draw,
            text=str(circuit.label),
            center=(left + 22.0, top + 24.0),
            font=title_font,
            fill=style.text_rgb,
            stroke_fill=style.panel_fill_rgb,
            stroke_width=2,
        )
        item_bboxes[f"{circuit.item_id}_label"] = list(label_bbox)
    if title:
        draw_centered_text(
            draw,
            text=str(title),
            center=(0.5 * (left + right), top + 24.0),
            font=small_font,
            fill=style.text_rgb,
            stroke_fill=style.panel_fill_rgb,
            stroke_width=2,
        )

    input_x = left + 48.0
    input_top = top + 62.0
    input_bottom = bottom - 58.0
    input_count = max(1, len(circuit.inputs))
    input_gap = 0.0 if input_count == 1 else (input_bottom - input_top) / float(input_count - 1)
    signal_source_points: dict[str, tuple[float, float]] = {}
    input_y_by_id: dict[str, float] = {
        str(input_spec.item_id): input_top + (index * input_gap)
        for index, input_spec in enumerate(circuit.inputs)
    }

    signal_level: dict[str, int] = {str(input_spec.item_id): 0 for input_spec in circuit.inputs}
    gate_levels: dict[str, int] = {}
    gates_by_level: dict[int, list[Any]] = {}
    for gate in circuit.gates:
        level = 1 + max(signal_level[str(signal_id)] for signal_id in gate.input_signal_ids)
        gate_levels[str(gate.item_id)] = int(level)
        gates_by_level.setdefault(int(level), []).append(gate)
        signal_level[str(gate.output_signal_id)] = int(level)

    max_level = max(gate_levels.values(), default=1)
    gate_area_left = left + 178.0
    gate_area_right = right - 136.0
    level_gap = 0.0 if max_level <= 1 else (gate_area_right - gate_area_left) / float(max_level - 1)
    y_min = top + 54.0
    y_max = bottom - 48.0
    min_gate_gap = float(params.gate_height_px) + max(22.0, float(params.wire_width_px) * 7.0)
    signal_y: dict[str, float] = {str(input_id): float(y) for input_id, y in input_y_by_id.items()}
    gate_centers: dict[str, tuple[float, float]] = {}
    for level in range(1, int(max_level) + 1):
        level_gates = gates_by_level.get(int(level), [])
        proposed_y: list[tuple[str, float]] = []
        for gate in level_gates:
            source_ys = [float(signal_y[str(signal_id)]) for signal_id in gate.input_signal_ids]
            proposed_y.append((str(gate.item_id), sum(source_ys) / float(len(source_ys))))
        spread_y = _spread_y_positions(
            proposed_y,
            lower=float(y_min),
            upper=float(y_max),
            min_gap=float(min_gate_gap),
        )
        x = gate_area_left + ((int(level) - 1) * level_gap)
        for gate in level_gates:
            y = float(spread_y[str(gate.item_id)])
            gate_centers[str(gate.item_id)] = (float(x), float(y))
            signal_y[str(gate.output_signal_id)] = float(y)

    for index, input_spec in enumerate(circuit.inputs):
        y = float(input_y_by_id[str(input_spec.item_id)])
        text = str(input_spec.label)
        if bool(show_fixed_input_values) and input_spec.value is not None:
            text = f"{input_spec.label}={int(input_spec.value)}"
        input_bbox = draw_centered_text(
            draw,
            text=text,
            center=(input_x, y),
            font=small_font,
            fill=style.text_rgb,
            stroke_fill=style.panel_fill_rgb,
            stroke_width=2,
        )
        port = (input_x + 36.0, y)
        signal_source_points[str(input_spec.item_id)] = port
        signal_points[str(input_spec.item_id)] = _rounded_point(*port)
        item_bboxes[str(input_spec.item_id)] = _rounded_bbox(input_bbox)
        draw.ellipse((port[0] - 3, port[1] - 3, port[0] + 3, port[1] + 3), fill=tuple(int(value) for value in style.text_rgb))

    gate_draw_specs: list[dict[str, Any]] = []
    wire_specs: list[dict[str, Any]] = []
    for gate in circuit.gates:
        cx, cy = gate_centers[str(gate.item_id)]
        gate_bbox = (
            cx - 0.5 * int(params.gate_width_px),
            cy - 0.5 * int(params.gate_height_px),
            cx + 0.5 * int(params.gate_width_px),
            cy + 0.5 * int(params.gate_height_px),
        )
        input_ports = _gate_input_ports(gate_bbox, len(gate.input_signal_ids))
        visual_input_signal_ids = sorted(
            (str(signal_id) for signal_id in gate.input_signal_ids),
            key=lambda signal_id: (signal_source_points[str(signal_id)][1], str(signal_id)),
        )
        for input_index, signal_id in enumerate(visual_input_signal_ids):
            start = signal_source_points[str(signal_id)]
            end = input_ports[int(input_index)]
            wire_specs.append(
                {
                    "wire_id": f"{signal_id}->{gate.item_id}:{input_index}",
                    "source_signal_id": str(signal_id),
                    "target_item_id": str(gate.item_id),
                    "target_port_index": int(input_index),
                    "start": start,
                    "end": end,
                    "lane_x": _wire_lane_x(
                        start=start,
                        end=end,
                        input_index=int(input_index),
                        input_count=len(gate.input_signal_ids),
                    ),
                }
            )
        if str(gate.gate_type).upper() in {"NOT", "NAND", "NOR"}:
            bubble_r = 4.0
            bubble_cx = float(gate_bbox[2]) + bubble_r
            bubble_cy = cy
            out_port = (bubble_cx + bubble_r, bubble_cy)
        else:
            bubble_r = 0.0
            bubble_cx = 0.0
            bubble_cy = 0.0
            out_port = (float(gate_bbox[2]), cy)
        signal_source_points[str(gate.output_signal_id)] = out_port
        signal_points[str(gate.output_signal_id)] = _rounded_point(*out_port)
        item_bboxes[str(gate.item_id)] = _rounded_bbox(gate_bbox)
        gate_draw_specs.append(
            {
                "gate_type": str(gate.gate_type),
                "bbox": gate_bbox,
                "center": (cx, cy),
                "has_bubble": str(gate.gate_type).upper() in {"NOT", "NAND", "NOR"},
                "bubble": (bubble_cx, bubble_cy, bubble_r),
            }
        )

    final_source = signal_source_points[str(circuit.output_signal_id)]
    output_center = (right - 54.0, final_source[1])
    output_id = f"{circuit.item_id}_output"
    wire_specs.append(
        {
            "wire_id": f"{circuit.output_signal_id}->{output_id}",
            "source_signal_id": str(circuit.output_signal_id),
            "target_item_id": str(output_id),
            "target_port_index": 0,
            "start": final_source,
            "end": output_center,
            "lane_x": None,
        }
    )
    rendered_wire_segments: list[dict[str, Any]] = []
    for segment in wire_specs:
        points = _wire_polyline_points(
            start=segment["start"],
            end=segment["end"],
            lane_x=segment["lane_x"],
        )
        _draw_wire(
            draw,
            points=points,
            rgb=style.text_rgb,
            width=int(params.wire_width_px),
        )
        rendered_wire_segments.append(
            {
                "wire_id": str(segment["wire_id"]),
                "source_signal_id": str(segment["source_signal_id"]),
                "target_item_id": str(segment["target_item_id"]),
                "target_port_index": int(segment["target_port_index"]),
                "points": [_rounded_point(float(point[0]), float(point[1])) for point in points],
            }
        )
    for gate_spec in gate_draw_specs:
        gate_bbox = gate_spec["bbox"]
        _draw_gate_symbol(
            draw,
            gate_type=str(gate_spec["gate_type"]),
            bbox=gate_bbox,
            style=style,
            width=2,
        )
        if bool(gate_spec["has_bubble"]):
            bubble_cx, bubble_cy, bubble_r = gate_spec["bubble"]
            draw.ellipse(
                (
                    float(bubble_cx) - float(bubble_r),
                    float(bubble_cy) - float(bubble_r),
                    float(bubble_cx) + float(bubble_r),
                    float(bubble_cy) + float(bubble_r),
                ),
                fill=tuple(int(value) for value in style.panel_fill_rgb),
                outline=tuple(int(value) for value in style.text_rgb),
                width=2,
            )
    _draw_output_node(draw, center=output_center, params=params, style=style)
    draw_centered_text(
        draw,
        text="OUT",
        center=(right - 54.0, output_center[1] + 23.0),
        font=small_font,
        fill=style.text_rgb,
        stroke_fill=style.panel_fill_rgb,
        stroke_width=2,
    )
    output_points[output_id] = _rounded_point(*output_center)
    signal_points[output_id] = _rounded_point(*output_center)
    item_bboxes[output_id] = _rounded_bbox(
        (
            output_center[0] - int(params.node_radius_px),
            output_center[1] - int(params.node_radius_px),
            output_center[0] + int(params.node_radius_px),
            output_center[1] + int(params.node_radius_px),
        )
    )
    entity = {
        "item_id": str(circuit.item_id),
        "entity_type": "logic_circuit",
        "role": str(circuit.role),
        "label": str(circuit.label),
        "bbox_px": _rounded_bbox(bbox),
        "input_ids": [str(item.item_id) for item in circuit.inputs],
        "gate_ids": [str(item.item_id) for item in circuit.gates],
        "output_point_id": str(output_id),
        "output_value": None if circuit.output_value is None else int(circuit.output_value),
    }
    return entity, item_bboxes, output_points, signal_points, tuple(rendered_wire_segments)


def render_logic_gate_option_scene(
    image: Image.Image,
    *,
    circuits: Sequence[LogicCircuitSpec],
    params: LogicGateRenderParams,
    style: SymbolicSceneStyle,
) -> RenderedLogicGateScene:
    """Render four independent circuit-option panels for output-value selection."""

    if len(circuits) != 4:
        raise ValueError("logic-gate output-label scenes require exactly four circuit options")
    draw = ImageDraw.Draw(image)
    width, height = int(params.canvas_width), int(params.canvas_height)
    margin_x = 48
    margin_y = 50
    gap_x = 34
    gap_y = 32
    card_w = (width - (2 * margin_x) - gap_x) / 2.0
    card_h = (height - (2 * margin_y) - gap_y) / 2.0

    entities: list[dict[str, Any]] = []
    item_bboxes: dict[str, list[float]] = {}
    output_points: dict[str, list[float]] = {}
    signal_points: dict[str, list[float]] = {}
    wire_segments: list[dict[str, Any]] = []
    for index, circuit in enumerate(circuits):
        row = index // 2
        col = index % 2
        left = margin_x + col * (card_w + gap_x)
        top = margin_y + row * (card_h + gap_y)
        bbox = (left, top, left + card_w, top + card_h)
        entity, boxes, points, signals, wires = _draw_logic_circuit(
            draw,
            circuit=circuit,
            bbox=bbox,
            params=params,
            style=style,
            show_fixed_input_values=True,
        )
        entities.append(entity)
        item_bboxes.update(boxes)
        output_points.update(points)
        signal_points.update(signals)
        wire_segments.extend(dict(wire) for wire in wires)

    return RenderedLogicGateScene(
        image=image,
        entities=tuple(entities),
        item_bboxes=item_bboxes,
        output_points=output_points,
        signal_points=signal_points,
        wire_segments=tuple(wire_segments),
        scene_bbox_px=_rounded_bbox((30, 32, width - 30, height - 32)),
        style_metadata={
            "renderer": "logic_gate_circuit_v1",
            "layout": "four_circuit_option_grid",
            "gate_rendering": "standard_logic_symbols_no_gate_text",
            "wire_routing": "planar_expression_tree_orthogonal_v4",
        },
    )


def render_logic_single_circuit_scene(
    image: Image.Image,
    *,
    circuit: LogicCircuitSpec,
    params: LogicGateRenderParams,
    style: SymbolicSceneStyle,
    show_fixed_input_values: bool,
) -> RenderedLogicGateScene:
    """Render one circuit panel for source-only circuit objectives."""

    draw = ImageDraw.Draw(image)
    width, height = int(params.canvas_width), int(params.canvas_height)
    circuit_bbox = (72.0, 86.0, float(width - 72), float(height - 78))
    entity, item_bboxes, output_points, signal_points, wire_segments = _draw_logic_circuit(
        draw,
        circuit=circuit,
        bbox=circuit_bbox,
        params=params,
        style=style,
        show_fixed_input_values=bool(show_fixed_input_values),
        title="Circuit",
    )
    return RenderedLogicGateScene(
        image=image,
        entities=(dict(entity),),
        item_bboxes=item_bboxes,
        output_points=output_points,
        signal_points=signal_points,
        wire_segments=tuple(wire_segments),
        scene_bbox_px=_rounded_bbox((44, 56, width - 44, height - 44)),
        style_metadata={
            "renderer": "logic_gate_circuit_v1",
            "layout": "single_circuit_panel",
            "gate_rendering": "standard_logic_symbols_no_gate_text",
            "wire_routing": "planar_expression_tree_orthogonal_v4",
        },
    )


def render_logic_assignment_scene(
    image: Image.Image,
    *,
    circuit: LogicCircuitSpec,
    candidates: Sequence[CandidateAssignmentSpec],
    params: LogicGateRenderParams,
    style: SymbolicSceneStyle,
) -> RenderedLogicGateScene:
    """Render one source circuit and four labeled candidate assignment rows."""

    if len(candidates) != 4:
        raise ValueError("logic-gate assignment scenes require exactly four candidates")
    draw = ImageDraw.Draw(image)
    width, height = int(params.canvas_width), int(params.canvas_height)
    entities: list[dict[str, Any]] = []
    item_bboxes: dict[str, list[float]] = {}
    output_points: dict[str, list[float]] = {}
    signal_points: dict[str, list[float]] = {}
    wire_segments: list[dict[str, Any]] = []

    circuit_bbox = (48.0, 120.0, 730.0, 690.0)
    entity, boxes, points, signals, wires = _draw_logic_circuit(
        draw,
        circuit=circuit,
        bbox=circuit_bbox,
        params=params,
        style=style,
        show_fixed_input_values=False,
        title="Source circuit",
    )
    entities.append(entity)
    item_bboxes.update(boxes)
    output_points.update(points)
    signal_points.update(signals)
    wire_segments.extend(dict(wire) for wire in wires)

    table_bbox = (775.0, 126.0, float(width - 48), 690.0)
    draw_rounded_rect(
        draw,
        table_bbox,
        radius=int(params.card_corner_radius_px),
        fill=tuple(int(value) for value in style.panel_fill_rgb),
        outline=tuple(int(value) for value in style.panel_border_rgb),
        width=int(params.card_border_width_px),
    )
    table_font = load_font(int(params.table_font_size_px), bold=True)
    small_font = load_font(int(params.small_font_size_px), bold=True)
    title_font = load_font(int(params.label_font_size_px), bold=True)
    draw_centered_text(
        draw,
        text="Assignments",
        center=(0.5 * (table_bbox[0] + table_bbox[2]), table_bbox[1] + 28.0),
        font=title_font,
        fill=style.text_rgb,
        stroke_fill=style.panel_fill_rgb,
        stroke_width=2,
    )

    header_y = table_bbox[1] + 70.0
    col_x = {
        "label": table_bbox[0] + 38.0,
        "x": table_bbox[0] + 122.0,
        "y": table_bbox[0] + 206.0,
        "z": table_bbox[0] + 290.0,
    }
    for key, label in (("label", ""), ("x", "x"), ("y", "y"), ("z", "z")):
        draw_centered_text(
            draw,
            text=str(label),
            center=(col_x[str(key)], header_y),
            font=small_font,
            fill=style.text_rgb,
            stroke_fill=style.panel_fill_rgb,
            stroke_width=2,
        )
    draw.line(
        (table_bbox[0] + 18, header_y + 24, table_bbox[2] - 18, header_y + 24),
        fill=tuple(int(value) for value in style.grid_rgb),
        width=2,
    )

    row_top = header_y + 48.0
    available_row_h = max(72.0, (table_bbox[3] - row_top - 24.0) / float(len(candidates)))
    for index, candidate in enumerate(candidates):
        top = row_top + (index * available_row_h)
        bbox = (table_bbox[0] + 18.0, top, table_bbox[2] - 18.0, top + available_row_h - 12.0)
        if index % 2 == 0:
            draw.rounded_rectangle(
                bbox,
                radius=8,
                fill=tuple(
                    int(0.5 * int(a) + 0.5 * int(b))
                    for a, b in zip(style.panel_fill_rgb, style.background_rgb)
                ),
                outline=None,
            )
        y_center = 0.5 * (bbox[1] + bbox[3])
        draw_centered_text(
            draw,
            text=str(candidate.label),
            center=(col_x["label"], y_center),
            font=table_font,
            fill=style.text_rgb,
            stroke_fill=style.panel_fill_rgb,
            stroke_width=2,
        )
        for key in ("x", "y", "z"):
            draw_centered_text(
                draw,
                text=str(int(candidate.values[str(key)])),
                center=(col_x[str(key)], y_center),
                font=table_font,
                fill=style.text_rgb,
                stroke_fill=style.panel_fill_rgb,
                stroke_width=2,
            )
        item_bboxes[str(candidate.item_id)] = _rounded_bbox(bbox)
        entities.append(
            {
                "item_id": str(candidate.item_id),
                "entity_type": "logic_assignment_option",
                "role": "correct_option" if bool(candidate.is_correct) else "distractor_option",
                "label": str(candidate.label),
                "values": {str(key): int(value) for key, value in candidate.values.items()},
                "output_value": int(candidate.output_value),
                "is_correct": bool(candidate.is_correct),
                "bbox_px": _rounded_bbox(bbox),
            }
        )

    item_bboxes["assignment_table"] = _rounded_bbox(table_bbox)
    entities.append(
        {
            "item_id": "assignment_table",
            "entity_type": "logic_assignment_table",
            "role": "candidate_options",
            "bbox_px": _rounded_bbox(table_bbox),
            "option_ids": [str(candidate.item_id) for candidate in candidates],
        }
    )
    return RenderedLogicGateScene(
        image=image,
        entities=tuple(entities),
        item_bboxes=item_bboxes,
        output_points=output_points,
        signal_points=signal_points,
        wire_segments=tuple(wire_segments),
        scene_bbox_px=_rounded_bbox((34, 60, width - 34, height - 52)),
        style_metadata={
            "renderer": "logic_gate_circuit_v1",
            "layout": "source_circuit_and_assignment_options",
            "gate_rendering": "standard_logic_symbols_no_gate_text",
            "wire_routing": "planar_expression_tree_orthogonal_v4",
        },
    )


def render_option_bundle(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    circuits: Sequence[LogicCircuitSpec],
) -> LogicGateRenderBundle:
    """Finalize the four-panel circuit-option rendering with scene-wide style/noise."""

    render_params = resolve_render_params(render_defaults)
    background, background_meta, scene_style, scene_style_meta = resolve_background(
        instance_seed=int(instance_seed),
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
    )
    rendered = render_logic_gate_option_scene(
        background,
        circuits=circuits,
        params=render_params,
        style=scene_style,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=noise_defaults,
    )
    return LogicGateRenderBundle(
        image=image,
        rendered=rendered,
        render_params=render_params,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        scene_style_meta=dict(scene_style_meta),
    )


def render_single_circuit_bundle(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    circuit: LogicCircuitSpec,
    show_fixed_input_values: bool = False,
) -> LogicGateRenderBundle:
    """Finalize a single-circuit rendering with scene-wide style/noise."""

    render_params = resolve_render_params(render_defaults)
    background, background_meta, scene_style, scene_style_meta = resolve_background(
        instance_seed=int(instance_seed),
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
    )
    rendered = render_logic_single_circuit_scene(
        background,
        circuit=circuit,
        params=render_params,
        style=scene_style,
        show_fixed_input_values=bool(show_fixed_input_values),
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=noise_defaults,
    )
    return LogicGateRenderBundle(
        image=image,
        rendered=rendered,
        render_params=render_params,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        scene_style_meta=dict(scene_style_meta),
    )


def render_assignment_bundle(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    noise_defaults: Mapping[str, Any],
    circuit: LogicCircuitSpec,
    candidates: Sequence[CandidateAssignmentSpec],
) -> LogicGateRenderBundle:
    """Finalize the source-circuit plus assignment-options rendering."""

    render_params = resolve_render_params(render_defaults)
    background, background_meta, scene_style, scene_style_meta = resolve_background(
        instance_seed=int(instance_seed),
        canvas_width=int(render_params.canvas_width),
        canvas_height=int(render_params.canvas_height),
    )
    rendered = render_logic_assignment_scene(
        background,
        circuit=circuit,
        candidates=candidates,
        params=render_params,
        style=scene_style,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=noise_defaults,
    )
    return LogicGateRenderBundle(
        image=image,
        rendered=rendered,
        render_params=render_params,
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
        scene_style_meta=dict(scene_style_meta),
    )


def rounded_render_maps(bundle: LogicGateRenderBundle) -> tuple[dict[str, list[float]], dict[str, list[float]], dict[str, list[float]]]:
    """Return rounded item, output, and signal geometry maps from a rendered bundle."""

    rendered = bundle.rendered
    item_bboxes = {
        str(key): [round(float(value), 3) for value in bbox]
        for key, bbox in rendered.item_bboxes.items()
    }
    output_points = {
        str(key): [round(float(value[0]), 3), round(float(value[1]), 3)]
        for key, value in rendered.output_points.items()
    }
    signal_points = {
        str(key): [round(float(value[0]), 3), round(float(value[1]), 3)]
        for key, value in rendered.signal_points.items()
    }
    return item_bboxes, output_points, signal_points


def _rounded_wire_segments(rendered: RenderedLogicGateScene) -> list[dict[str, Any]]:
    """Return JSON-friendly wire polylines from the rendered scene."""

    rounded: list[dict[str, Any]] = []
    for segment in rendered.wire_segments:
        rounded.append(
            {
                "wire_id": str(segment["wire_id"]),
                "source_signal_id": str(segment["source_signal_id"]),
                "target_item_id": str(segment["target_item_id"]),
                "target_port_index": int(segment["target_port_index"]),
                "points": [
                    [round(float(point[0]), 3), round(float(point[1]), 3)]
                    for point in segment.get("points", ())
                ],
            }
        )
    return rounded


def render_payload_sections(
    bundle: LogicGateRenderBundle,
    *,
    item_bboxes: Mapping[str, Sequence[float]],
    output_points: Mapping[str, Sequence[float]],
    signal_points: Mapping[str, Sequence[float]],
    annotation_source: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build render-spec and render-map sections that are independent of objective logic."""

    rendered = bundle.rendered
    render_spec = {
        "scene_id": SCENE_ID,
        "canvas_width": int(bundle.render_params.canvas_width),
        "canvas_height": int(bundle.render_params.canvas_height),
        "coord_space": "pixel",
        "scene_style": dict(bundle.scene_style_meta),
        "logic_gate_style": dict(rendered.style_metadata),
        "background_style": dict(bundle.background_meta),
        "post_image_noise": dict(bundle.post_noise_meta),
        "scene_bbox_px": list(rendered.scene_bbox_px),
    }
    render_map = {
        "image_id": "img0",
        "scene_bbox_px": list(rendered.scene_bbox_px),
        "item_bboxes_px": {str(key): list(value) for key, value in item_bboxes.items()},
        "output_points_px": {str(key): list(value) for key, value in output_points.items()},
        "signal_points_px": {str(key): list(value) for key, value in signal_points.items()},
        "wire_segments_px": _rounded_wire_segments(rendered),
        "annotation_source": str(annotation_source),
    }
    return render_spec, render_map
