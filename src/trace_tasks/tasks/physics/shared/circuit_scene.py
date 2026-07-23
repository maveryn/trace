"""Shared equivalent-circuit rendering helpers for physics circuits tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from ...shared.drawing import draw_centered_text_with_auto_stroke as _draw_centered_text
from ...shared.text_rendering import load_font
from ...shared.text_legibility import draw_text_traced
from .style import build_physics_circuit_theme


@dataclass(frozen=True)
class CircuitComponentSpec:
    """One visible labeled circuit component in a rendered scene."""

    component_id: str
    label: str
    kind: str
    value: int
    unit: str
    bbox_px: List[float]
    symbol_bbox_px: List[float]
    label_bbox_px: List[float]


@dataclass(frozen=True)
class RenderedCircuitScene:
    """Rendered equivalent-circuit scene plus prompt-facing annotation metadata."""

    image: Image.Image
    component_specs: List[CircuitComponentSpec]
    annotation_bbox_map: Dict[str, List[float]]
    annotation_entity_id_map: Dict[str, str]
    render_map: Dict[str, Any]
    scene_entities: List[Dict[str, Any]]


def _rgb(value: Sequence[int]) -> Tuple[int, int, int]:
    return tuple(int(channel) for channel in value[:3])


def _round_bbox(bbox_px: Sequence[float]) -> List[float]:
    return [round(float(value), 3) for value in bbox_px]


def _union_bbox(*bboxes: Sequence[float]) -> List[float]:
    usable = [tuple(float(v) for v in bbox) for bbox in bboxes if len(bbox) == 4]
    if not usable:
        return [0.0, 0.0, 0.0, 0.0]
    return _round_bbox(
        [
            min(bbox[0] for bbox in usable),
            min(bbox[1] for bbox in usable),
            max(bbox[2] for bbox in usable),
            max(bbox[3] for bbox in usable),
        ]
    )


def _draw_terminal(
    draw: ImageDraw.ImageDraw,
    *,
    center_xy: Tuple[float, float],
    label: str,
    radius_px: float,
    font,
    theme,
    stroke_width_px: int,
) -> tuple[List[float], List[float]]:
    """Draw one labeled terminal and return circle/text bboxes."""

    center_x, center_y = float(center_xy[0]), float(center_xy[1])
    circle_bbox = _round_bbox(
        [
            center_x - radius_px,
            center_y - radius_px,
            center_x + radius_px,
            center_y + radius_px,
        ]
    )
    draw.ellipse(
        tuple(float(value) for value in circle_bbox),
        fill=_rgb(theme.terminal_fill_rgb),
        outline=_rgb(theme.terminal_outline_rgb),
        width=3,
    )
    text_bbox = _draw_centered_text(
        draw,
        text=str(label),
        center_xy=(float(center_x), float(center_y - (radius_px + 22.0))),
        font=font,
        fill=_rgb(theme.terminal_text_rgb),
        stroke_width_px=int(stroke_width_px),
    )
    return circle_bbox, text_bbox


def _draw_label_tag(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center_xy: Tuple[float, float],
    font,
    theme,
    diagram_style: Any | None,
    stroke_width_px: int,
) -> List[float]:
    """Draw a compact label tag and return its bbox."""

    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=0)
    text_width = float(text_bbox[2] - text_bbox[0])
    text_height = float(text_bbox[3] - text_bbox[1])
    pad_x = 9.0
    pad_y = 5.0
    cx, cy = float(center_xy[0]), float(center_xy[1])
    tag_bbox = _round_bbox(
        [
            cx - (0.5 * text_width) - pad_x,
            cy - (0.5 * text_height) - pad_y,
            cx + (0.5 * text_width) + pad_x,
            cy + (0.5 * text_height) + pad_y,
        ]
    )
    label_fill = tuple(int(value) for value in getattr(diagram_style, "label_fill_rgb", theme.resistor_fill_rgb))
    label_outline = tuple(int(value) for value in getattr(diagram_style, "label_border_rgb", theme.resistor_outline_rgb))
    label_text = tuple(int(value) for value in getattr(diagram_style, "label_rgb", theme.resistor_text_rgb))
    draw.rounded_rectangle(
        tuple(float(value) for value in tag_bbox),
        radius=6,
        fill=label_fill,
        outline=label_outline,
        width=2,
    )
    text_xy = (
        float(cx - (0.5 * text_width) - text_bbox[0]),
        float(cy - (0.5 * text_height) - text_bbox[1]),
    )
    draw_text_traced(draw,
        text_xy,
        str(text),
        font=font,
        fill=label_text,
        stroke_width=max(0, int(stroke_width_px) - 2),
        stroke_fill=label_fill,
     role="readout", required=False,)
    return list(tag_bbox)


def _draw_resistor_symbol(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Sequence[float],
    theme,
    wire_width_px: int,
) -> List[float]:
    """Draw an ANSI-style zigzag resistor in one horizontal component slot."""

    left, top, right, bottom = [float(value) for value in bbox_px]
    center_y = 0.5 * (top + bottom)
    lead = min(18.0, max(8.0, 0.16 * (right - left)))
    zig_left = left + lead
    zig_right = right - lead
    amplitude = min(20.0, max(10.0, 0.38 * (bottom - top)))
    points: List[Tuple[float, float]] = [(left, center_y), (zig_left, center_y)]
    segment_count = 8
    for index in range(segment_count + 1):
        x_value = zig_left + ((zig_right - zig_left) * index / float(segment_count))
        if index == 0 or index == segment_count:
            y_value = center_y
        else:
            y_value = center_y + (amplitude if index % 2 else -amplitude)
        points.append((float(x_value), float(y_value)))
    points.append((right, center_y))
    draw.line(points, fill=_rgb(theme.wire_rgb), width=max(2, int(wire_width_px)), joint="curve")
    return _round_bbox([left, center_y - amplitude - wire_width_px, right, center_y + amplitude + wire_width_px])


def _draw_capacitor_symbol(
    draw: ImageDraw.ImageDraw,
    *,
    bbox_px: Sequence[float],
    theme,
    wire_width_px: int,
) -> List[float]:
    """Draw a parallel-plate capacitor in one horizontal component slot."""

    left, top, right, bottom = [float(value) for value in bbox_px]
    center_y = 0.5 * (top + bottom)
    center_x = 0.5 * (left + right)
    gap = min(18.0, max(10.0, 0.15 * (right - left)))
    plate_height = min(54.0, max(32.0, 0.88 * (bottom - top)))
    plate_left_x = center_x - (0.5 * gap)
    plate_right_x = center_x + (0.5 * gap)
    plate_top = center_y - (0.5 * plate_height)
    plate_bottom = center_y + (0.5 * plate_height)
    ink = _rgb(theme.wire_rgb)
    width = max(2, int(wire_width_px))
    draw.line([(left, center_y), (plate_left_x, center_y)], fill=ink, width=width)
    draw.line([(plate_right_x, center_y), (right, center_y)], fill=ink, width=width)
    draw.line([(plate_left_x, plate_top), (plate_left_x, plate_bottom)], fill=ink, width=width + 1)
    draw.line([(plate_right_x, plate_top), (plate_right_x, plate_bottom)], fill=ink, width=width + 1)
    return _round_bbox([plate_left_x - width, plate_top - width, plate_right_x + width, plate_bottom + width])


def render_component_network_scene(
    *,
    scene_variant: str,
    component_kind: str,
    series_values: Sequence[int],
    parallel_values: Sequence[int],
    parallel_blocks: Sequence[Sequence[int]] | None = None,
    inter_block_series_values: Sequence[int] | None = None,
    outer_series_values: Sequence[int] | None = None,
    background: Image.Image,
    render_defaults: Mapping[str, Any],
    accent_color_name: str,
    origin_offset_px: Tuple[float, float] = (0.0, 0.0),
    diagram_style: Any | None = None,
    font_family: str | None = None,
) -> RenderedCircuitScene:
    """Render one resistor or capacitor network between labeled terminals `A` and `B`."""

    kind = str(component_kind)
    if kind not in {"resistor", "capacitor"}:
        raise ValueError(f"unsupported circuit component kind: {component_kind}")

    canvas = background.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    theme = build_physics_circuit_theme(str(accent_color_name), diagram_style=diagram_style)
    canvas_width = int(render_defaults["canvas_width"])
    canvas_height = int(render_defaults["canvas_height"])
    mid_y = float(0.5 * canvas_height)
    wire_width = int(render_defaults["wire_width_px"])
    terminal_radius = float(render_defaults["terminal_radius_px"])
    symbol_width = float(
        render_defaults.get(
            "component_symbol_width_px",
            render_defaults.get("resistor_box_width_px", 96),
        )
    )
    symbol_height = float(
        render_defaults.get(
            "component_symbol_height_px",
            render_defaults.get("resistor_box_height_px", 48),
        )
    )
    terminal_label_font = load_font(
        int(render_defaults["terminal_font_size_px"]),
        bold=True,
        font_family=font_family,
    )
    component_font = load_font(
        int(render_defaults.get("component_label_font_size_px", render_defaults.get("resistor_font_size_px", 22))),
        bold=False,
        font_family=font_family,
    )
    label_stroke_width = int(render_defaults["label_stroke_width_px"])
    origin_x = float(origin_offset_px[0])
    origin_y = float(origin_offset_px[1])
    prefix = "R" if kind == "resistor" else "C"
    unit = "ohm" if kind == "resistor" else "uF"

    terminal_left = (float(origin_x + float(render_defaults["terminal_left_x_px"])), float(origin_y + mid_y))
    terminal_right = (
        float(origin_x + float(canvas_width - int(render_defaults["terminal_left_x_px"]))),
        float(origin_y + mid_y),
    )
    left_circle_bbox, left_label_bbox = _draw_terminal(
        draw,
        center_xy=terminal_left,
        label="A",
        radius_px=float(terminal_radius),
        font=terminal_label_font,
        theme=theme,
        stroke_width_px=label_stroke_width,
    )
    right_circle_bbox, right_label_bbox = _draw_terminal(
        draw,
        center_xy=terminal_right,
        label="B",
        radius_px=float(terminal_radius),
        font=terminal_label_font,
        theme=theme,
        stroke_width_px=label_stroke_width,
    )

    component_specs: List[CircuitComponentSpec] = []
    scene_entities: List[Dict[str, Any]] = [
        {
            "entity_id": "terminal_A",
            "entity_type": "physics_circuit_terminal",
            "bbox_px": list(left_circle_bbox),
            "meta": {"terminal_label": "A"},
        },
        {
            "entity_id": "terminal_B",
            "entity_type": "physics_circuit_terminal",
            "bbox_px": list(right_circle_bbox),
            "meta": {"terminal_label": "B"},
        },
    ]
    wire_segments: List[List[List[float]]] = []

    def line(start_xy: Tuple[float, float], end_xy: Tuple[float, float]) -> None:
        draw.line(
            [tuple(float(v) for v in start_xy), tuple(float(v) for v in end_xy)],
            fill=_rgb(theme.wire_rgb),
            width=max(1, int(wire_width)),
        )
        wire_segments.append(
            [
                [round(float(start_xy[0]), 3), round(float(start_xy[1]), 3)],
                [round(float(end_xy[0]), 3), round(float(end_xy[1]), 3)],
            ]
        )

    next_component_index = 1

    def add_component(value: int, slot_bbox_px: Sequence[float]) -> CircuitComponentSpec:
        nonlocal next_component_index
        label = f"{prefix}{int(next_component_index)}"
        component_id = f"{kind}_{int(next_component_index)}"
        slot = [float(v) for v in slot_bbox_px]
        if kind == "resistor":
            symbol_bbox = _draw_resistor_symbol(
                draw,
                bbox_px=slot,
                theme=theme,
                wire_width_px=wire_width,
            )
        else:
            symbol_bbox = _draw_capacitor_symbol(
                draw,
                bbox_px=slot,
                theme=theme,
                wire_width_px=wire_width,
            )
        label_bbox = _draw_label_tag(
            draw,
            text=f"{label}={int(value)} {unit}",
            center_xy=(0.5 * (slot[0] + slot[2]), slot[1] - 22.0),
            font=component_font,
            theme=theme,
            diagram_style=diagram_style,
            stroke_width_px=label_stroke_width,
        )
        component_bbox = _union_bbox(symbol_bbox, label_bbox)
        spec = CircuitComponentSpec(
            component_id=str(component_id),
            label=str(label),
            kind=str(kind),
            value=int(value),
            unit=str(unit),
            bbox_px=list(component_bbox),
            symbol_bbox_px=list(symbol_bbox),
            label_bbox_px=list(label_bbox),
        )
        component_specs.append(spec)
        scene_entities.append(
            {
                "entity_id": str(component_id),
                "entity_type": f"physics_{kind}",
                "bbox_px": list(component_bbox),
                "meta": {
                    "component_label": str(label),
                    "component_kind": str(kind),
                    "value": int(value),
                    "unit": str(unit),
                    "symbol_bbox_px": list(symbol_bbox),
                    "label_bbox_px": list(label_bbox),
                },
            }
        )
        next_component_index += 1
        return spec

    def add_series_chain(start_x: float, end_x: float, values: Sequence[int]) -> None:
        if not values:
            line((float(start_x), float(origin_y + mid_y)), (float(end_x), float(origin_y + mid_y)))
            return
        count = len(values)
        usable_width = float(end_x - start_x)
        step = usable_width / float(count)
        previous_x = float(start_x)
        for value_index, value in enumerate(values):
            center_x = float(start_x + ((value_index + 0.5) * step))
            slot = [
                round(float(center_x - (0.5 * symbol_width)), 3),
                round(float(origin_y + mid_y - (0.5 * symbol_height)), 3),
                round(float(center_x + (0.5 * symbol_width)), 3),
                round(float(origin_y + mid_y + (0.5 * symbol_height)), 3),
            ]
            line((float(previous_x), float(origin_y + mid_y)), (float(slot[0]), float(origin_y + mid_y)))
            add_component(int(value), slot)
            previous_x = float(slot[2])
        line((float(previous_x), float(origin_y + mid_y)), (float(end_x), float(origin_y + mid_y)))

    def add_parallel_bank(left_x: float, right_x: float, values: Sequence[int]) -> None:
        if not values:
            return
        branch_top_y = float(origin_y + float(render_defaults["parallel_branch_top_y_px"]))
        branch_bottom_y = float(origin_y + float(render_defaults["parallel_branch_bottom_y_px"]))
        if len(values) == 1:
            branch_ys = [float(0.5 * (branch_top_y + branch_bottom_y))]
        else:
            step_y = float(branch_bottom_y - branch_top_y) / float(len(values) - 1)
            branch_ys = [float(branch_top_y + (index * step_y)) for index in range(len(values))]
        line((left_x, float(min(branch_ys))), (left_x, float(max(branch_ys))))
        line((right_x, float(min(branch_ys))), (right_x, float(max(branch_ys))))
        component_center_x = 0.5 * float(left_x + right_x)
        for value, branch_y in zip(values, branch_ys, strict=True):
            slot = [
                round(float(component_center_x - (0.5 * symbol_width)), 3),
                round(float(branch_y - (0.5 * symbol_height)), 3),
                round(float(component_center_x + (0.5 * symbol_width)), 3),
                round(float(branch_y + (0.5 * symbol_height)), 3),
            ]
            line((left_x, float(branch_y)), (float(slot[0]), float(branch_y)))
            line((float(slot[2]), float(branch_y)), (right_x, float(branch_y)))
            add_component(int(value), slot)

    def add_gap_series_component(start_x: float, end_x: float, value: int) -> None:
        if int(value) <= 0:
            line((float(start_x), float(origin_y + mid_y)), (float(end_x), float(origin_y + mid_y)))
            return
        center_x = float(0.5 * (float(start_x) + float(end_x)))
        slot = [
            round(float(center_x - (0.5 * symbol_width)), 3),
            round(float(origin_y + mid_y - (0.5 * symbol_height)), 3),
            round(float(center_x + (0.5 * symbol_width)), 3),
            round(float(origin_y + mid_y + (0.5 * symbol_height)), 3),
        ]
        line((float(start_x), float(origin_y + mid_y)), (float(slot[0]), float(origin_y + mid_y)))
        add_component(int(value), slot)
        line((float(slot[2]), float(origin_y + mid_y)), (float(end_x), float(origin_y + mid_y)))

    def add_compound_parallel_chain(
        start_x: float,
        end_x: float,
        blocks: Sequence[Sequence[int]],
        gaps: Sequence[int],
    ) -> None:
        if not blocks:
            line((float(start_x), float(origin_y + mid_y)), (float(end_x), float(origin_y + mid_y)))
            return
        if len(gaps) != max(0, len(blocks) - 1):
            raise ValueError("compound circuit gaps must have length len(blocks) - 1")
        slot_count = (2 * len(blocks)) - 1
        slot_width = float(end_x - start_x) / float(slot_count)
        for block_index, block_values in enumerate(blocks):
            block_left_x = float(start_x + ((2 * block_index) * slot_width))
            block_right_x = float(start_x + (((2 * block_index) + 1) * slot_width))
            add_parallel_bank(block_left_x, block_right_x, block_values)
            if block_index < len(gaps):
                gap_left_x = block_right_x
                gap_right_x = float(start_x + (((2 * block_index) + 2) * slot_width))
                add_gap_series_component(gap_left_x, gap_right_x, int(gaps[block_index]))

    compound_blocks = (
        tuple(tuple(int(value) for value in block) for block in parallel_blocks)
        if parallel_blocks is not None
        else tuple()
    )
    compound_gaps = (
        tuple(int(value) for value in inter_block_series_values)
        if inter_block_series_values is not None
        else tuple()
    )
    compound_outer = (
        tuple(int(value) for value in outer_series_values)
        if outer_series_values is not None
        else tuple()
    )

    left_anchor_x = float(terminal_left[0] + terminal_radius)
    right_anchor_x = float(terminal_right[0] - terminal_radius)
    if compound_blocks:
        if compound_outer:
            if len(compound_outer) != 2:
                raise ValueError("compound outer series values must contain left and right slots")
            slot_count = (2 * len(compound_blocks)) + 1
            slot_width = float(right_anchor_x - left_anchor_x) / float(slot_count)
            chain_left_x = float(left_anchor_x + slot_width)
            chain_right_x = float(right_anchor_x - slot_width)
            add_gap_series_component(left_anchor_x, chain_left_x, int(compound_outer[0]))
            add_compound_parallel_chain(chain_left_x, chain_right_x, compound_blocks, compound_gaps)
            add_gap_series_component(chain_right_x, right_anchor_x, int(compound_outer[1]))
        else:
            add_compound_parallel_chain(left_anchor_x, right_anchor_x, compound_blocks, compound_gaps)
    elif str(scene_variant) == "parallel":
        rail_left_x = float(origin_x + float(render_defaults["parallel_rail_left_x_px"]))
        rail_right_x = float(origin_x + float(canvas_width - int(render_defaults["parallel_rail_left_x_px"])))
        line((left_anchor_x, float(origin_y + mid_y)), (rail_left_x, float(origin_y + mid_y)))
        line((rail_right_x, float(origin_y + mid_y)), (right_anchor_x, float(origin_y + mid_y)))
        add_parallel_bank(rail_left_x, rail_right_x, parallel_values)
    else:
        add_series_chain(left_anchor_x, right_anchor_x, series_values)

    annotation_bbox_map = {str(spec.label): list(spec.bbox_px) for spec in component_specs}
    annotation_entity_id_map = {str(spec.label): str(spec.component_id) for spec in component_specs}
    render_map = {
        "accent_color_name": str(accent_color_name),
        "component_kind": str(kind),
        "component_symbol_style": "ansi_zigzag" if kind == "resistor" else "parallel_plates",
        "component_bboxes_px": {str(spec.label): list(spec.bbox_px) for spec in component_specs},
        "component_symbol_bboxes_px": {str(spec.label): list(spec.symbol_bbox_px) for spec in component_specs},
        "component_label_bboxes_px": {str(spec.label): list(spec.label_bbox_px) for spec in component_specs},
        "component_entity_ids": dict(annotation_entity_id_map),
        "component_values": {str(spec.label): int(spec.value) for spec in component_specs},
        "component_units": {str(spec.label): str(spec.unit) for spec in component_specs},
        "wire_segments_px": list(wire_segments),
        "terminal_bboxes_px": {
            "A": list(left_circle_bbox),
            "B": list(right_circle_bbox),
        },
        "terminal_label_bboxes_px": {
            "A": list(left_label_bbox),
            "B": list(right_label_bbox),
        },
        "annotation_bbox_map_px": dict(annotation_bbox_map),
        "annotation_entity_id_map": dict(annotation_entity_id_map),
    }
    if kind == "resistor":
        render_map["resistor_bboxes_px"] = dict(annotation_bbox_map)
    if diagram_style is not None:
        render_map["technical_diagram_frame_mode"] = str(getattr(diagram_style, "frame_mode", "none"))
    return RenderedCircuitScene(
        image=canvas,
        component_specs=list(component_specs),
        annotation_bbox_map=dict(annotation_bbox_map),
        annotation_entity_id_map=dict(annotation_entity_id_map),
        render_map=render_map,
        scene_entities=list(scene_entities),
    )

__all__ = [
    "CircuitComponentSpec",
    "RenderedCircuitScene",
    "render_component_network_scene",
]
