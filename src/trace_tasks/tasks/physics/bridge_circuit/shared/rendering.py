"""Rendering primitives for bridge-circuit diagrams."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.drawing import draw_centered_text
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.named_colors import named_color
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .annotations import bbox, clip_bbox, normalize_annotation_bbox_map, union_bbox
from .state import (
    BRIDGE_EQUATION,
    DEFAULT_RENDERING,
    SCENE_ID,
    SCENE_NAMESPACE,
    BridgeScenario,
    RenderedBridgeScene,
)


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def draw_label_tag(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font: Any,
    style: Any,
    fill_rgb: Tuple[int, int, int] | None = None,
    outline_rgb: Tuple[int, int, int] | None = None,
    text_rgb: Tuple[int, int, int] | None = None,
) -> List[float]:
    """Draw a rounded text tag and return the unioned visual bbox."""

    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=0)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    cx, cy = float(center[0]), float(center[1])
    pad_x = 10.0
    pad_y = 6.0
    tag = bbox(
        (
            cx - (text_w * 0.5) - pad_x,
            cy - (text_h * 0.5) - pad_y,
            cx + (text_w * 0.5) + pad_x,
            cy + (text_h * 0.5) + pad_y,
        )
    )
    fill = tuple(int(value) for value in (fill_rgb or style.label_fill_rgb))
    outline = tuple(int(value) for value in (outline_rgb or style.label_border_rgb))
    text_color = tuple(int(value) for value in (text_rgb or style.stroke_rgb))
    draw.rounded_rectangle(tuple(float(value) for value in tag), radius=7, fill=fill, outline=outline, width=2)
    text_box = draw_centered_text(
        draw,
        text=str(text),
        center=(cx, cy),
        font=font,
        fill=text_color,
        stroke_fill=resolve_text_stroke_fill(text_color),
        stroke_width=1,
    )
    return union_bbox(tag, text_box)


def draw_vertical_resistor(
    draw: ImageDraw.ImageDraw,
    *,
    center_x: float,
    y0: float,
    y1: float,
    style: Any,
    wire_width: int,
) -> List[float]:
    """Draw a vertical zigzag resistor symbol."""

    stroke = tuple(int(value) for value in style.stroke_rgb)
    top = float(min(y0, y1))
    bottom = float(max(y0, y1))
    lead = 20.0
    zig_top = top + lead
    zig_bottom = bottom - lead
    amplitude = 20.0
    points: List[Tuple[float, float]] = [(float(center_x), top), (float(center_x), zig_top)]
    segment_count = 8
    for index in range(segment_count + 1):
        y_value = zig_top + ((zig_bottom - zig_top) * index / float(segment_count))
        if index == 0 or index == segment_count:
            x_value = float(center_x)
        else:
            x_value = float(center_x) + (amplitude if index % 2 else -amplitude)
        points.append((float(x_value), float(y_value)))
    points.append((float(center_x), bottom))
    draw.line(points, fill=stroke, width=max(2, int(wire_width)), joint="curve")
    return bbox(
        (
            float(center_x) - amplitude - wire_width,
            top,
            float(center_x) + amplitude + wire_width,
            bottom,
        )
    )


def draw_battery(
    draw: ImageDraw.ImageDraw,
    *,
    x: float,
    y_top: float,
    y_bottom: float,
    style: Any,
    font: Any,
    wire_width: int,
) -> List[float]:
    """Draw the bridge supply battery."""

    stroke = tuple(int(value) for value in style.stroke_rgb)
    guide = tuple(int(value) for value in style.guide_rgb)
    mid_y = (float(y_top) + float(y_bottom)) * 0.5
    plate_top = mid_y - 25.0
    plate_bottom = mid_y + 25.0
    draw.line((x, y_top, x, plate_top - 18.0), fill=stroke, width=int(wire_width))
    draw.line((x, plate_bottom + 18.0, x, y_bottom), fill=stroke, width=int(wire_width))
    draw.line((x - 42.0, plate_top, x + 42.0, plate_top), fill=stroke, width=max(3, int(wire_width) + 2))
    draw.line((x - 28.0, plate_bottom, x + 28.0, plate_bottom), fill=stroke, width=max(3, int(wire_width) + 2))
    plus_box = draw_centered_text(
        draw,
        text="+",
        center=(x + 62.0, plate_top - 2.0),
        font=font,
        fill=stroke,
        stroke_fill=resolve_text_stroke_fill(stroke),
        stroke_width=1,
    )
    minus_box = draw_centered_text(
        draw,
        text="-",
        center=(x + 62.0, plate_bottom),
        font=font,
        fill=guide,
        stroke_fill=resolve_text_stroke_fill(guide),
        stroke_width=1,
    )
    return union_bbox((x - 46.0, plate_top - 24.0, x + 70.0, plate_bottom + 24.0), plus_box, minus_box)


def draw_zero_meter(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    style: Any,
    accent_rgb: Tuple[int, int, int],
    label_font: Any,
    wire_width: int,
) -> List[float]:
    """Draw the center galvanometer reading zero."""

    cx, cy = float(center[0]), float(center[1])
    stroke = tuple(int(value) for value in style.stroke_rgb)
    fill = tuple(int(value) for value in style.panel_alt_fill_rgb)
    radius = 34.0
    circle = bbox((cx - radius, cy - radius, cx + radius, cy + radius))
    draw.ellipse(tuple(float(value) for value in circle), fill=fill, outline=stroke, width=3)
    draw.arc((cx - 21.0, cy - 18.0, cx + 21.0, cy + 24.0), start=205, end=335, fill=tuple(int(value) for value in style.guide_rgb), width=2)
    draw.line((cx, cy + 10.0, cx, cy - 14.0), fill=accent_rgb, width=max(2, int(wire_width) - 1))
    letter_box = draw_centered_text(
        draw,
        text="G",
        center=(cx, cy + 14.0),
        font=label_font,
        fill=stroke,
        stroke_fill=resolve_text_stroke_fill(stroke),
        stroke_width=1,
    )
    readout_box = draw_label_tag(
        draw,
        text="G=0",
        center=(cx, cy + 60.0),
        font=label_font,
        style=style,
    )
    return union_bbox(circle, letter_box, readout_box)


def draw_bridge_circuit(
    *,
    image: Image.Image,
    scenario: BridgeScenario,
    font_family: str,
    style: Any,
    render_defaults: Mapping[str, Any],
) -> tuple[Image.Image, Dict[str, List[float]], Dict[str, Any], List[Dict[str, Any]]]:
    """Draw the bridge circuit and return projected visual witness boxes."""

    draw = ImageDraw.Draw(image)
    canvas_width, canvas_height = image.size
    title_font = load_font(
        int(render_defaults.get("title_font_size_px", DEFAULT_RENDERING.title_font_size_px)),
        bold=True,
        font_family=font_family,
    )
    label_font = load_font(
        int(
            render_defaults.get(
                "component_label_font_size_px",
                DEFAULT_RENDERING.component_label_font_size_px,
            )
        ),
        bold=True,
        font_family=font_family,
    )
    small_font = load_font(22, bold=True, font_family=font_family)
    wire_width = int(render_defaults.get("wire_width_px", DEFAULT_RENDERING.wire_width_px))
    panel = (
        float(render_defaults.get("panel_left_px", DEFAULT_RENDERING.panel_left_px)),
        float(render_defaults.get("panel_top_px", DEFAULT_RENDERING.panel_top_px)),
        float(render_defaults.get("panel_right_px", DEFAULT_RENDERING.panel_right_px)),
        float(render_defaults.get("panel_bottom_px", DEFAULT_RENDERING.panel_bottom_px)),
    )
    draw.rounded_rectangle(
        panel,
        radius=20,
        fill=tuple(int(value) for value in style.panel_fill_rgb),
        outline=tuple(int(value) for value in style.panel_border_rgb),
        width=3,
    )
    text_rgb = tuple(int(value) for value in style.stroke_rgb)
    draw_centered_text(
        draw,
        text="Balanced bridge circuit",
        center=((panel[0] + panel[2]) * 0.5, panel[1] + 34.0),
        font=title_font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=1,
    )

    stroke = tuple(int(value) for value in style.stroke_rgb)
    accent_rgb = named_color(str(scenario.accent_color_name))
    target_rgb = named_color("red")
    target_fill = (255, 236, 236)

    top_y = 152.0
    bottom_y = 560.0
    mid_y = (top_y + bottom_y) * 0.5
    source_x = 238.0
    left_branch_x = 520.0
    right_branch_x = 850.0
    right_bus_x = 1006.0
    meter_center = ((left_branch_x + right_branch_x) * 0.5, mid_y)

    draw_battery(
        draw,
        x=source_x,
        y_top=top_y,
        y_bottom=bottom_y,
        style=style,
        font=small_font,
        wire_width=wire_width,
    )
    draw.line((source_x, top_y, right_bus_x, top_y), fill=stroke, width=wire_width)
    draw.line((source_x, bottom_y, right_bus_x, bottom_y), fill=stroke, width=wire_width)
    draw.line((right_bus_x, top_y, right_bus_x, bottom_y), fill=stroke, width=wire_width)

    positions = {
        "R1": (left_branch_x, top_y + 42.0, mid_y - 34.0, left_branch_x - 112.0, (top_y + mid_y) * 0.5),
        "R2": (left_branch_x, mid_y + 34.0, bottom_y - 42.0, left_branch_x - 112.0, (mid_y + bottom_y) * 0.5),
        "R3": (right_branch_x, top_y + 42.0, mid_y - 34.0, right_branch_x + 112.0, (top_y + mid_y) * 0.5),
        "R4": (right_branch_x, mid_y + 34.0, bottom_y - 42.0, right_branch_x + 112.0, (mid_y + bottom_y) * 0.5),
    }
    raw_annotation: Dict[str, List[float]] = {}
    resistor_bboxes: Dict[str, List[float]] = {}
    resistor_label_bboxes: Dict[str, List[float]] = {}
    resistor_symbol_bboxes: Dict[str, List[float]] = {}
    scene_entities: List[Dict[str, Any]] = []
    for spec in scenario.resistors:
        x, y0, y1, label_x, label_y = positions[str(spec.label)]
        if str(spec.label) in {"R1", "R3"}:
            draw.line((float(x), top_y, float(x), float(y0)), fill=stroke, width=wire_width)
            draw.line((float(x), float(y1), float(x), mid_y), fill=stroke, width=wire_width)
        else:
            draw.line((float(x), mid_y, float(x), float(y0)), fill=stroke, width=wire_width)
            draw.line((float(x), float(y1), float(x), bottom_y), fill=stroke, width=wire_width)
        symbol_box = draw_vertical_resistor(
            draw,
            center_x=float(x),
            y0=float(y0),
            y1=float(y1),
            style=style,
            wire_width=wire_width,
        )
        label_text = f"{spec.label}=?" if bool(spec.is_missing) else f"{spec.label}={int(spec.value_ohm)} ohm"
        label_box = draw_label_tag(
            draw,
            text=label_text,
            center=(float(label_x), float(label_y)),
            font=label_font,
            style=style,
            fill_rgb=target_fill if bool(spec.is_missing) else None,
            outline_rgb=target_rgb if bool(spec.is_missing) else None,
            text_rgb=target_rgb if bool(spec.is_missing) else None,
        )
        full_box = union_bbox(symbol_box, label_box)
        resistor_bboxes[str(spec.label)] = list(full_box)
        resistor_label_bboxes[str(spec.label)] = list(label_box)
        resistor_symbol_bboxes[str(spec.label)] = list(symbol_box)
        if bool(spec.is_missing):
            raw_annotation["target_resistor"] = list(full_box)
        else:
            raw_annotation[str(spec.label)] = list(full_box)
        scene_entities.append(
            {
                "entity_id": str(spec.label),
                "entity_type": "bridge_resistor",
                "label": str(spec.label),
                "value_ohm": int(spec.value_ohm),
                "is_missing": bool(spec.is_missing),
                "bbox_px": clip_bbox(full_box, width=int(canvas_width), height=int(canvas_height)),
            }
        )

    draw.line((left_branch_x, mid_y, meter_center[0] - 38.0, mid_y), fill=stroke, width=wire_width)
    draw.line((meter_center[0] + 38.0, mid_y, right_branch_x, mid_y), fill=stroke, width=wire_width)
    zero_meter_bbox = draw_zero_meter(
        draw,
        center=meter_center,
        style=style,
        accent_rgb=accent_rgb,
        label_font=small_font,
        wire_width=wire_width,
    )
    raw_annotation["zero_meter"] = list(zero_meter_bbox)
    for node in (
        (left_branch_x, mid_y),
        (right_branch_x, mid_y),
        (left_branch_x, top_y),
        (right_branch_x, top_y),
        (left_branch_x, bottom_y),
        (right_branch_x, bottom_y),
    ):
        cx, cy = float(node[0]), float(node[1])
        draw.ellipse((cx - 5.0, cy - 5.0, cx + 5.0, cy + 5.0), fill=stroke)

    annotation = normalize_annotation_bbox_map(
        raw_annotation,
        missing_resistor=str(scenario.missing_resistor),
        width=int(canvas_width),
        height=int(canvas_height),
    )
    render_map = {
        "panel_bbox": bbox(panel),
        "scene_variant": str(scenario.scene_variant),
        "missing_resistor": str(scenario.missing_resistor),
        "target_answer": int(scenario.target_answer),
        "resistor_values": {
            str(spec.label): int(spec.value_ohm)
            for spec in scenario.resistors
        },
        "resistor_bboxes": {
            str(key): clip_bbox(value, width=int(canvas_width), height=int(canvas_height))
            for key, value in resistor_bboxes.items()
        },
        "resistor_symbol_bboxes": {
            str(key): clip_bbox(value, width=int(canvas_width), height=int(canvas_height))
            for key, value in resistor_symbol_bboxes.items()
        },
        "resistor_label_bboxes": {
            str(key): clip_bbox(value, width=int(canvas_width), height=int(canvas_height))
            for key, value in resistor_label_bboxes.items()
        },
        "zero_meter_bbox": clip_bbox(
            zero_meter_bbox,
            width=int(canvas_width),
            height=int(canvas_height),
        ),
        "annotation_bbox_map": dict(annotation),
        "bridge_equation": BRIDGE_EQUATION,
        "wire_segments": {
            "top_bus": bbox((source_x, top_y, right_bus_x, top_y)),
            "bottom_bus": bbox((source_x, bottom_y, right_bus_x, bottom_y)),
            "right_bus": bbox((right_bus_x, top_y, right_bus_x, bottom_y)),
            "meter_branch": bbox((left_branch_x, mid_y, right_branch_x, mid_y)),
        },
    }
    return image, annotation, render_map, scene_entities


def render_bridge_circuit(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: BridgeScenario,
    render_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> RenderedBridgeScene:
    """Render a complete balanced bridge circuit."""

    canvas_width = int(render_defaults.get("canvas_width", DEFAULT_RENDERING.canvas_width))
    canvas_height = int(render_defaults.get("canvas_height", DEFAULT_RENDERING.canvas_height))
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
    rendered, annotation_map, render_map, scene_entities = draw_bridge_circuit(
        image=background,
        scenario=scenario,
        font_family=str(font_family),
        style=diagram_style,
        render_defaults=render_defaults,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    render_map.update(
        {
            "technical_diagram_style": dict(diagram_style_meta),
            "background_style": background_meta,
            "post_image_noise": post_noise_meta,
        }
    )
    return RenderedBridgeScene(
        image=image,
        annotation_bbox_map={str(key): list(value) for key, value in annotation_map.items()},
        scene_entities=scene_entities,
        render_map=render_map,
        font_family=str(font_family),
    )


__all__ = [
    "draw_battery",
    "draw_bridge_circuit",
    "draw_label_tag",
    "draw_vertical_resistor",
    "draw_zero_meter",
    "render_bridge_circuit",
]
