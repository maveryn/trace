"""Rendering primitives for circuit state-change diagrams."""

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

from .annotations import bbox, clip_bbox, union_bbox
from .state import (
    DEFAULT_RENDERING,
    SCENE_ID,
    SCENE_NAMESPACE,
    BulbStateChangeSpec,
    CircuitStateChangeScenario,
    RenderedCircuitStateChangeScene,
)


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def draw_label_tag(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font: Any,
    style: Any,
) -> List[float]:
    """Draw a rounded label tag and return its visual bbox."""

    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=0)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    cx, cy = float(center[0]), float(center[1])
    pad_x = 10.0
    pad_y = 6.0
    tag = bbox(
        (
            cx - text_w * 0.5 - pad_x,
            cy - text_h * 0.5 - pad_y,
            cx + text_w * 0.5 + pad_x,
            cy + text_h * 0.5 + pad_y,
        )
    )
    draw.rounded_rectangle(
        tuple(float(value) for value in tag),
        radius=7,
        fill=tuple(int(value) for value in style.label_fill_rgb),
        outline=tuple(int(value) for value in style.label_border_rgb),
        width=2,
    )
    text_box = draw_centered_text(
        draw,
        text=str(text),
        center=(cx, cy),
        font=font,
        fill=tuple(int(value) for value in style.stroke_rgb),
        stroke_fill=resolve_text_stroke_fill(style.stroke_rgb),
        stroke_width=1,
    )
    return union_bbox(tag, text_box)


def draw_bulb(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    spec: BulbStateChangeSpec,
    style: Any,
    accent_rgb: Tuple[int, int, int],
    label_font: Any,
    label_offset: Tuple[float, float],
) -> List[float]:
    """Draw one bulb symbol and resistance tag."""

    cx, cy = float(center[0]), float(center[1])
    radius = 34.0
    stroke = tuple(int(value) for value in style.stroke_rgb)
    fill = tuple(int(value) for value in style.panel_alt_fill_rgb)
    glass = (
        min(255, int(fill[0]) + 12),
        min(255, int(fill[1]) + 10),
        min(255, int(fill[2]) + 4),
    )
    bulb_bbox = bbox((cx - radius, cy - radius, cx + radius, cy + radius))
    draw.ellipse(
        tuple(float(value) for value in bulb_bbox),
        fill=glass,
        outline=stroke,
        width=4,
    )
    base = bbox((cx - 17.0, cy + 21.0, cx + 17.0, cy + 38.0))
    draw.rounded_rectangle(
        tuple(float(value) for value in base),
        radius=4,
        fill=tuple(int(value) for value in style.muted_fill_rgb),
        outline=stroke,
        width=3,
    )
    filament_rgb = tuple(
        max(0, min(255, int((0.35 * accent_rgb[idx]) + (0.65 * stroke[idx]))))
        for idx in range(3)
    )
    filament_y = cy - 1.0
    filament_points = [
        (cx - 19.0, filament_y + 7.0),
        (cx - 11.0, filament_y - 7.0),
        (cx - 3.5, filament_y + 7.0),
        (cx + 4.0, filament_y - 7.0),
        (cx + 12.0, filament_y + 7.0),
        (cx + 19.0, filament_y - 7.0),
    ]
    draw.line(filament_points, fill=filament_rgb, width=4, joint="curve")
    draw.line((cx - 15.0, cy + 20.0, cx - 19.0, filament_y + 7.0), fill=stroke, width=3)
    draw.line((cx + 15.0, cy + 20.0, cx + 19.0, filament_y - 7.0), fill=stroke, width=3)
    label_box = draw_label_tag(
        draw,
        text=f"{spec.label}={int(spec.resistance_ohm)} ohm",
        center=(cx + float(label_offset[0]), cy + float(label_offset[1])),
        font=label_font,
        style=style,
    )
    return union_bbox(bulb_bbox, base, label_box)


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
    """Draw the ideal battery symbol that anchors the circuit loop."""

    stroke = tuple(int(value) for value in style.stroke_rgb)
    guide = tuple(int(value) for value in style.guide_rgb)
    mid_y = (float(y_top) + float(y_bottom)) * 0.5
    plate_top = mid_y - 22.0
    plate_bottom = mid_y + 24.0
    draw.line((x, y_top, x, plate_top - 16.0), fill=stroke, width=int(wire_width))
    draw.line((x, plate_bottom + 16.0, x, y_bottom), fill=stroke, width=int(wire_width))
    draw.line(
        (x - 38.0, plate_top, x + 38.0, plate_top),
        fill=stroke,
        width=max(3, int(wire_width) + 2),
    )
    draw.line(
        (x - 25.0, plate_bottom, x + 25.0, plate_bottom),
        fill=stroke,
        width=max(3, int(wire_width) + 2),
    )
    plus_box = draw_centered_text(
        draw,
        text="+",
        center=(x + 58.0, plate_top - 2.0),
        font=font,
        fill=stroke,
        stroke_fill=resolve_text_stroke_fill(stroke),
        stroke_width=1,
    )
    minus_box = draw_centered_text(
        draw,
        text="-",
        center=(x + 58.0, plate_bottom),
        font=font,
        fill=guide,
        stroke_fill=resolve_text_stroke_fill(guide),
        stroke_width=1,
    )
    return union_bbox((x - 42.0, plate_top - 20.0, x + 66.0, plate_bottom + 20.0), plus_box, minus_box)


def draw_switch(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    closed: bool,
    action_text: str,
    style: Any,
    font: Any,
    wire_width: int,
) -> List[float]:
    """Draw the switch and red action cue."""

    cx, cy = float(center[0]), float(center[1])
    stroke = tuple(int(value) for value in style.stroke_rgb)
    red = (204, 48, 48)
    left_contact = (cx - 38.0, cy)
    right_contact = (cx + 38.0, cy)
    draw.line((cx - 82.0, cy, left_contact[0], cy), fill=stroke, width=int(wire_width))
    draw.line((right_contact[0], cy, cx + 82.0, cy), fill=stroke, width=int(wire_width))
    for contact in (left_contact, right_contact):
        draw.ellipse(
            (
                contact[0] - 5.5,
                contact[1] - 5.5,
                contact[0] + 5.5,
                contact[1] + 5.5,
            ),
            fill=stroke,
        )
    if bool(closed):
        draw.line(
            (left_contact[0], left_contact[1], right_contact[0], right_contact[1]),
            fill=tuple(int(value) for value in style.secondary_accent_rgb),
            width=5,
        )
    else:
        draw.line(
            (left_contact[0], left_contact[1], right_contact[0] - 8.0, right_contact[1] - 32.0),
            fill=tuple(int(value) for value in style.secondary_accent_rgb),
            width=5,
        )
    switch_box = bbox((cx - 48.0, cy - 38.0, cx + 48.0, cy + 16.0))
    action_box = draw_label_tag(
        draw,
        text=str(action_text),
        center=(cx, cy - 68.0),
        font=font,
        style=style,
    )
    outline_box = union_bbox(switch_box, action_box)
    draw.rounded_rectangle(tuple(float(value) for value in outline_box), radius=10, outline=red, width=4)
    return union_bbox(outline_box, (cx - 82.0, cy - 8.0, cx + 82.0, cy + 8.0))


def render_scene_core(
    *,
    image: Image.Image,
    scenario: CircuitStateChangeScenario,
    font_family: str,
    style: Any,
    render_defaults: Mapping[str, Any],
) -> Tuple[Image.Image, Dict[str, List[float]], List[Dict[str, Any]], Dict[str, Any]]:
    """Render the state-change circuit before post-image noise."""

    draw = ImageDraw.Draw(image)
    canvas_width, canvas_height = image.size
    title_font = load_font(
        int(render_defaults.get("title_font_size_px", DEFAULT_RENDERING.title_font_size_px)),
        bold=True,
        font_family=font_family,
    )
    label_font = load_font(
        int(render_defaults.get("component_label_font_size_px", DEFAULT_RENDERING.component_label_font_size_px)),
        bold=True,
        font_family=font_family,
    )
    battery_font = load_font(24, bold=True, font_family=font_family)
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
        text="Switch action in an ideal bulb circuit",
        center=((panel[0] + panel[2]) * 0.5, panel[1] + 34.0),
        font=title_font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=1,
    )

    stroke = tuple(int(value) for value in style.stroke_rgb)
    accent_rgb = named_color(str(scenario.accent_color_name))
    y_top = 188.0
    y_return = 628.0
    x_battery = 172.0
    x_series = 386.0
    x_split = 560.0
    x_right = 1064.0
    y_main = 324.0
    y_switched = 456.0
    y_reference = 586.0
    x_main_bulb = 804.0
    x_switch = 654.0
    x_switched_bulb = 884.0
    x_reference_left = 252.0
    x_reference_bulb_1 = 420.0
    x_reference_bulb_2 = 760.0

    draw_battery(
        draw,
        x=x_battery,
        y_top=y_top,
        y_bottom=y_return,
        style=style,
        font=battery_font,
        wire_width=wire_width,
    )
    draw.line((x_battery, y_top, x_split, y_top), fill=stroke, width=wire_width)
    draw.line((x_split, y_top, x_split, y_switched), fill=stroke, width=wire_width)
    draw.line((x_split, y_main, x_right, y_main), fill=stroke, width=wire_width)
    draw.line((x_split, y_switched, x_right, y_switched), fill=stroke, width=wire_width)
    draw.line((x_right, y_main, x_right, y_return), fill=stroke, width=wire_width)
    draw.line((x_battery, y_return, x_right, y_return), fill=stroke, width=wire_width)
    draw.line((x_reference_left, y_top, x_reference_left, y_reference), fill=stroke, width=wire_width)
    draw.line((x_reference_left, y_reference, x_right, y_reference), fill=stroke, width=wire_width)

    spec_by_role = {str(spec.role): spec for spec in scenario.bulbs}
    bulb_bboxes = {
        str(spec_by_role["series_bulb"].label): draw_bulb(
            draw,
            center=(x_series, y_top),
            spec=spec_by_role["series_bulb"],
            style=style,
            accent_rgb=accent_rgb,
            label_font=label_font,
            label_offset=(0.0, -58.0),
        ),
        str(spec_by_role["main_branch_bulb"].label): draw_bulb(
            draw,
            center=(x_main_bulb, y_main),
            spec=spec_by_role["main_branch_bulb"],
            style=style,
            accent_rgb=accent_rgb,
            label_font=label_font,
            label_offset=(0.0, -62.0),
        ),
        str(spec_by_role["switched_branch_bulb"].label): draw_bulb(
            draw,
            center=(x_switched_bulb, y_switched),
            spec=spec_by_role["switched_branch_bulb"],
            style=style,
            accent_rgb=accent_rgb,
            label_font=label_font,
            label_offset=(0.0, -62.0),
        ),
        str(spec_by_role["reference_branch_bulb_1"].label): draw_bulb(
            draw,
            center=(x_reference_bulb_1, y_reference),
            spec=spec_by_role["reference_branch_bulb_1"],
            style=style,
            accent_rgb=accent_rgb,
            label_font=label_font,
            label_offset=(0.0, -58.0),
        ),
        str(spec_by_role["reference_branch_bulb_2"].label): draw_bulb(
            draw,
            center=(x_reference_bulb_2, y_reference),
            spec=spec_by_role["reference_branch_bulb_2"],
            style=style,
            accent_rgb=accent_rgb,
            label_font=label_font,
            label_offset=(0.0, -58.0),
        ),
    }
    before_switch_closed = str(scenario.switch_action) == "opens"
    action_text = "S opens" if str(scenario.switch_action) == "opens" else "S closes"
    changed_switch_bbox = draw_switch(
        draw,
        center=(x_switch, y_switched),
        closed=bool(before_switch_closed),
        action_text=str(action_text),
        style=style,
        font=label_font,
        wire_width=wire_width,
    )

    annotation_map = {
        "changed_switch": clip_bbox(changed_switch_bbox, width=canvas_width, height=canvas_height)
    }
    for label, box in sorted(bulb_bboxes.items()):
        annotation_map[str(label)] = clip_bbox(box, width=canvas_width, height=canvas_height)

    scene_entities: List[Dict[str, Any]] = [
        {
            "entity_id": "changed_switch",
            "entity_type": "switch",
            "bbox_px": list(annotation_map["changed_switch"]),
            "meta": {
                "switch_action": str(scenario.switch_action),
                "state_before": "closed" if bool(before_switch_closed) else "open",
                "state_after": "open" if bool(before_switch_closed) else "closed",
            },
        }
    ]
    for spec in scenario.bulbs:
        scene_entities.append(
            {
                "entity_id": str(spec.label),
                "entity_type": "bulb",
                "bbox_px": list(annotation_map[str(spec.label)]),
                "meta": {
                    "role": str(spec.role),
                    "resistance_ohm": int(spec.resistance_ohm),
                    "power_before": float(spec.power_before),
                    "power_after": float(spec.power_after),
                    "change_class": str(spec.change_class),
                },
            }
        )

    render_map = {
        "panel_bbox": bbox(panel),
        "bulb_bboxes": {str(label): list(box) for label, box in sorted(bulb_bboxes.items())},
        "changed_switch_bbox": list(annotation_map["changed_switch"]),
        "correct_label": str(scenario.correct_label),
        "switch_action": str(scenario.switch_action),
        "state_before": "closed" if bool(before_switch_closed) else "open",
        "state_after": "open" if bool(before_switch_closed) else "closed",
        "bulb_specs": [
            {
                "role": str(spec.role),
                "label": str(spec.label),
                "resistance_ohm": int(spec.resistance_ohm),
                "power_before": round(float(spec.power_before), 8),
                "power_after": round(float(spec.power_after), 8),
                "change_class": str(spec.change_class),
            }
            for spec in scenario.bulbs
        ],
    }
    return image, dict(annotation_map), [dict(entity) for entity in scene_entities], dict(render_map)


def render_circuit_state_change(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: CircuitStateChangeScenario,
    render_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> RenderedCircuitStateChangeScene:
    """Render the complete state-change scene and preserve projected annotations."""

    canvas_width = int(params.get("canvas_width", render_defaults.get("canvas_width", DEFAULT_RENDERING.canvas_width)))
    canvas_height = int(
        params.get("canvas_height", render_defaults.get("canvas_height", DEFAULT_RENDERING.canvas_height))
    )
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
    rendered, annotation_map, scene_entities, render_map = render_scene_core(
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
    render_map = dict(render_map)
    render_map["technical_diagram_style"] = dict(diagram_style_meta)
    render_map["background_style"] = dict(background_meta)
    render_map["post_image_noise"] = dict(post_noise_meta)
    return RenderedCircuitStateChangeScene(
        image=image,
        annotation_bbox_map={str(key): list(value) for key, value in annotation_map.items()},
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
        font_family=str(font_family),
    )


__all__ = ["render_circuit_state_change"]
