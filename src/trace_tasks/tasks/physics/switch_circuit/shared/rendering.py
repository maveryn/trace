"""Rendering primitives for switch-circuit diagrams."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.diagram_style import (
    prepare_physics_diagram_style_and_background,
)
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.drawing import draw_centered_text
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.named_colors import named_color
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .annotations import bbox, clip_bbox, union_bbox
from .state import (
    BULB_LABELS,
    NEG_NODE,
    POS_NODE,
    SCENE_ID,
    SCENE_NAMESPACE,
    SWITCH_LABELS,
    RenderedSwitchCircuitScene,
    SwitchCircuitDefaults,
    SwitchCircuitScenario,
)


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def draw_label_tag(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font: Any,
    style: Any,
    stroke_width: int,
) -> List[float]:
    """Draw a rounded component label tag and return its visual bbox."""

    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=0)
    text_width = float(text_bbox[2] - text_bbox[0])
    text_height = float(text_bbox[3] - text_bbox[1])
    pad_x = 8.0
    pad_y = 5.0
    cx, cy = float(center[0]), float(center[1])
    tag_bbox = bbox(
        (
            cx - text_width / 2.0 - pad_x,
            cy - text_height / 2.0 - pad_y,
            cx + text_width / 2.0 + pad_x,
            cy + text_height / 2.0 + pad_y,
        )
    )
    draw.rounded_rectangle(
        tuple(float(value) for value in tag_bbox),
        radius=7,
        fill=tuple(int(value) for value in style.label_fill_rgb),
        outline=tuple(int(value) for value in style.label_border_rgb),
        width=2,
    )
    text_rgb = tuple(int(value) for value in style.label_rgb)
    text_box = draw_centered_text(
        draw,
        text=str(text),
        center=(cx, cy),
        font=font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=max(0, int(stroke_width) - 1),
    )
    return union_bbox(tag_bbox, text_box)


def draw_battery(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    font: Any,
    style: Any,
    wire_width: int,
) -> List[float]:
    """Draw an ideal battery symbol."""

    cx, cy = float(center[0]), float(center[1])
    stroke = tuple(int(value) for value in style.stroke_rgb)
    label_rgb = tuple(int(value) for value in style.label_rgb)
    draw.line((cx, cy - 82.0, cx, cy - 42.0), fill=stroke, width=wire_width)
    draw.line((cx, cy + 18.0, cx, cy + 58.0), fill=stroke, width=wire_width)
    draw.line((cx - 22.0, cy - 42.0, cx + 22.0, cy - 42.0), fill=stroke, width=wire_width + 2)
    draw.line((cx - 13.0, cy + 18.0, cx + 13.0, cy + 18.0), fill=stroke, width=wire_width + 2)
    plus_bbox = draw_centered_text(
        draw,
        text="+",
        center=(cx + 42.0, cy - 42.0),
        font=font,
        fill=label_rgb,
        stroke_fill=resolve_text_stroke_fill(label_rgb),
        stroke_width=1,
    )
    minus_bbox = draw_centered_text(
        draw,
        text="-",
        center=(cx + 42.0, cy + 18.0),
        font=font,
        fill=label_rgb,
        stroke_fill=resolve_text_stroke_fill(label_rgb),
        stroke_width=1,
    )
    return union_bbox((cx - 24.0, cy - 84.0, cx + 24.0, cy + 60.0), plus_bbox, minus_bbox)


def draw_switch(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    label: str,
    closed: bool,
    font: Any,
    style: Any,
    render_defaults: Mapping[str, int],
    label_side: str = "above",
) -> List[float]:
    """Draw an open or closed switch."""

    cx, cy = float(center[0]), float(center[1])
    width = float(render_defaults["switch_width_px"])
    height = float(render_defaults["switch_height_px"])
    half_w = width / 2.0
    contact_r = 4.8
    left_contact = (cx - half_w * 0.42, cy)
    right_contact = (cx + half_w * 0.42, cy)
    stroke = tuple(int(value) for value in style.stroke_rgb)
    switch_rgb = tuple(int(value) for value in style.secondary_accent_rgb)
    contact_fill = tuple(int(value) for value in style.panel_fill_rgb)
    erase_bbox = (cx - half_w * 0.56, cy - height * 0.50, cx + half_w * 0.56, cy + height * 0.30)
    draw.rounded_rectangle(
        erase_bbox,
        radius=8,
        fill=tuple(int(value) for value in style.panel_fill_rgb),
    )
    draw.ellipse(
        (
            left_contact[0] - contact_r,
            left_contact[1] - contact_r,
            left_contact[0] + contact_r,
            left_contact[1] + contact_r,
        ),
        fill=contact_fill,
        outline=stroke,
        width=2,
    )
    draw.ellipse(
        (
            right_contact[0] - contact_r,
            right_contact[1] - contact_r,
            right_contact[0] + contact_r,
            right_contact[1] + contact_r,
        ),
        fill=contact_fill,
        outline=stroke,
        width=2,
    )
    if closed:
        draw.line((left_contact[0], left_contact[1], right_contact[0], right_contact[1]), fill=switch_rgb, width=5)
    else:
        draw.line(
            (
                left_contact[0],
                left_contact[1],
                right_contact[0] - 7.0,
                right_contact[1] - height * 0.34,
            ),
            fill=switch_rgb,
            width=5,
        )
    symbol_bbox = bbox((cx - half_w * 0.50, cy - height * 0.45, cx + half_w * 0.50, cy + height * 0.22))
    label_dy = -height * 0.70 if str(label_side) == "above" else height * 0.72
    label_bbox = draw_label_tag(
        draw,
        text=str(label),
        center=(cx, cy + label_dy),
        font=font,
        style=style,
        stroke_width=int(render_defaults["label_stroke_width_px"]),
    )
    return union_bbox(symbol_bbox, label_bbox)


def draw_bulb(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    label: str,
    font: Any,
    style: Any,
    accent_rgb: Tuple[int, int, int],
    render_defaults: Mapping[str, int],
    label_side: str = "below",
) -> List[float]:
    """Draw a bulb symbol and return only the physical bulb bbox."""

    cx, cy = float(center[0]), float(center[1])
    radius = float(render_defaults["bulb_radius_px"])
    glass = tuple(int(value) for value in style.panel_alt_fill_rgb)
    outline = tuple(max(0, int(value) - 45) for value in accent_rgb)
    filament = tuple(int(value) for value in style.stroke_rgb)
    bulb_bbox = bbox((cx - radius, cy - radius, cx + radius, cy + radius))
    draw.ellipse(tuple(float(value) for value in bulb_bbox), fill=glass, outline=outline, width=3)
    draw.arc(
        (cx - radius * 0.45, cy - radius * 0.18, cx + radius * 0.45, cy + radius * 0.48),
        205,
        335,
        fill=filament,
        width=3,
    )
    draw.line(
        (cx - radius * 0.18, cy + radius * 0.48, cx + radius * 0.18, cy + radius * 0.48),
        fill=filament,
        width=3,
    )
    label_dy = radius + 25.0 if str(label_side) == "below" else -(radius + 25.0)
    draw_label_tag(
        draw,
        text=str(label),
        center=(cx, cy + label_dy),
        font=font,
        style=style,
        stroke_width=int(render_defaults["label_stroke_width_px"]),
    )
    return list(bulb_bbox)


def draw_wire(
    draw: ImageDraw.ImageDraw,
    points: Tuple[Tuple[float, float], ...],
    *,
    style: Any,
    wire_width: int,
) -> None:
    """Draw one polyline wire."""

    draw.line(
        [(float(x), float(y)) for x, y in points],
        fill=tuple(int(value) for value in style.stroke_rgb),
        width=int(wire_width),
    )


def render_switch_circuit_body(
    *,
    image: Image.Image,
    scenario: SwitchCircuitScenario,
    font_family: str,
    style: Any,
    render_defaults: Mapping[str, int],
) -> RenderedSwitchCircuitScene:
    """Render the switch-circuit body onto an existing image."""

    draw = ImageDraw.Draw(image)
    canvas_width, canvas_height = image.size
    panel = (
        float(render_defaults["panel_left_px"]),
        float(render_defaults["panel_top_px"]),
        float(render_defaults["panel_right_px"]),
        float(render_defaults["panel_bottom_px"]),
    )
    draw.rounded_rectangle(
        panel,
        radius=20,
        fill=tuple(int(value) for value in style.panel_fill_rgb),
        outline=tuple(int(value) for value in style.panel_border_rgb),
        width=3,
    )
    title_font = load_font(
        int(render_defaults["title_font_size_px"]),
        bold=True,
        font_family=font_family,
    )
    label_font = load_font(
        int(render_defaults["component_label_font_size_px"]),
        bold=True,
        font_family=font_family,
    )
    text_rgb = tuple(int(value) for value in style.stroke_rgb)
    draw_centered_text(
        draw,
        text="mixed switch circuit",
        center=((panel[0] + panel[2]) * 0.5, panel[1] + 34.0),
        font=title_font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=1,
    )

    accent_rgb = named_color(str(scenario.accent_color_name))
    wire_width = int(render_defaults["wire_width_px"])
    left_x = 220.0
    right_x = 1080.0
    top_y = 205.0
    upper_mid_y = 318.0
    lower_mid_y = 418.0
    bottom_y = 545.0
    split_left_x = 390.0
    split_right_x = 910.0
    battery_x = left_x - 78.0
    battery_y = (top_y + bottom_y) * 0.5
    battery_positive_y = battery_y - 82.0
    battery_negative_y = battery_y + 58.0
    return_y = float(panel[3] - 42.0)

    draw_wire(draw, ((left_x, top_y), (left_x, bottom_y)), style=style, wire_width=wire_width)
    draw_wire(draw, ((right_x, top_y), (right_x, return_y)), style=style, wire_width=wire_width)
    battery_bbox = draw_battery(
        draw,
        center=(battery_x, battery_y),
        font=label_font,
        style=style,
        wire_width=wire_width,
    )
    draw_wire(draw, ((battery_x, battery_positive_y), (left_x, battery_positive_y)), style=style, wire_width=wire_width)
    draw_wire(draw, ((battery_x, battery_negative_y), (battery_x, return_y)), style=style, wire_width=wire_width)
    draw_wire(draw, ((battery_x, return_y), (right_x, return_y)), style=style, wire_width=wire_width)

    draw_wire(draw, ((left_x, top_y), (right_x, top_y)), style=style, wire_width=wire_width)
    mid_y = (upper_mid_y + lower_mid_y) * 0.5
    draw_wire(draw, ((left_x, mid_y), (split_left_x, mid_y)), style=style, wire_width=wire_width)
    draw_wire(draw, ((split_left_x, upper_mid_y), (split_right_x, upper_mid_y)), style=style, wire_width=wire_width)
    draw_wire(draw, ((split_left_x, lower_mid_y), (split_right_x, lower_mid_y)), style=style, wire_width=wire_width)
    draw_wire(draw, ((split_left_x, upper_mid_y), (split_left_x, lower_mid_y)), style=style, wire_width=wire_width)
    draw_wire(draw, ((split_right_x, upper_mid_y), (split_right_x, lower_mid_y)), style=style, wire_width=wire_width)
    draw_wire(draw, ((split_right_x, mid_y), (right_x, mid_y)), style=style, wire_width=wire_width)
    draw_wire(draw, ((left_x, bottom_y), (right_x, bottom_y)), style=style, wire_width=wire_width)

    switch_bboxes = {
        "S1": draw_switch(
            draw,
            center=(310.0, top_y),
            label="S1",
            closed=bool(scenario.switch_states["S1"]),
            font=label_font,
            style=style,
            render_defaults=render_defaults,
            label_side="above",
        ),
        "S2": draw_switch(
            draw,
            center=(300.0, mid_y),
            label="S2",
            closed=bool(scenario.switch_states["S2"]),
            font=label_font,
            style=style,
            render_defaults=render_defaults,
            label_side="below",
        ),
        "S3": draw_switch(
            draw,
            center=(760.0, upper_mid_y),
            label="S3",
            closed=bool(scenario.switch_states["S3"]),
            font=label_font,
            style=style,
            render_defaults=render_defaults,
            label_side="above",
        ),
        "S4": draw_switch(
            draw,
            center=(550.0, lower_mid_y),
            label="S4",
            closed=bool(scenario.switch_states["S4"]),
            font=label_font,
            style=style,
            render_defaults=render_defaults,
            label_side="below",
        ),
        "S5": draw_switch(
            draw,
            center=(310.0, bottom_y),
            label="S5",
            closed=bool(scenario.switch_states["S5"]),
            font=label_font,
            style=style,
            render_defaults=render_defaults,
            label_side="above",
        ),
    }
    bulb_bboxes = {
        "B1": draw_bulb(
            draw,
            center=(555.0, top_y),
            label="B1",
            font=label_font,
            style=style,
            accent_rgb=accent_rgb,
            render_defaults=render_defaults,
            label_side="above",
        ),
        "B2": draw_bulb(
            draw,
            center=(785.0, top_y),
            label="B2",
            font=label_font,
            style=style,
            accent_rgb=accent_rgb,
            render_defaults=render_defaults,
            label_side="above",
        ),
        "B3": draw_bulb(
            draw,
            center=(555.0, upper_mid_y),
            label="B3",
            font=label_font,
            style=style,
            accent_rgb=accent_rgb,
            render_defaults=render_defaults,
            label_side="below",
        ),
        "B4": draw_bulb(
            draw,
            center=(760.0, lower_mid_y),
            label="B4",
            font=label_font,
            style=style,
            accent_rgb=accent_rgb,
            render_defaults=render_defaults,
            label_side="below",
        ),
        "B5": draw_bulb(
            draw,
            center=(655.0, bottom_y),
            label="B5",
            font=label_font,
            style=style,
            accent_rgb=accent_rgb,
            render_defaults=render_defaults,
            label_side="below",
        ),
    }
    bulb_bboxes = {
        key: clip_bbox(value, width=canvas_width, height=canvas_height)
        for key, value in sorted(bulb_bboxes.items())
    }
    switch_bboxes = {
        key: clip_bbox(value, width=canvas_width, height=canvas_height)
        for key, value in sorted(switch_bboxes.items())
    }
    lit_set = set(scenario.lit_bulbs)
    annotation_bboxes = [list(bulb_bboxes[label]) for label in BULB_LABELS if label in lit_set]

    entities: List[Dict[str, Any]] = [
        {
            "entity_id": "battery",
            "entity_type": "ideal_battery",
            "bbox_px": clip_bbox(battery_bbox, width=canvas_width, height=canvas_height),
            "meta": {"positive_node": POS_NODE, "negative_node": NEG_NODE},
        }
    ]
    for label in BULB_LABELS:
        entities.append(
            {
                "entity_id": str(label),
                "entity_type": "bulb",
                "bbox_px": list(bulb_bboxes[str(label)]),
                "meta": {"label": str(label), "is_lit": str(label) in lit_set},
            }
        )
    for label in SWITCH_LABELS:
        entities.append(
            {
                "entity_id": str(label),
                "entity_type": "switch",
                "bbox_px": list(switch_bboxes[str(label)]),
                "meta": {
                    "label": str(label),
                    "state": "closed" if bool(scenario.switch_states[str(label)]) else "open",
                    "conductive": bool(scenario.switch_states[str(label)]),
                },
            }
        )

    render_map = {
        "panel_bbox": bbox(panel),
        "scene_variant": str(scenario.scene_variant),
        "battery_bbox": clip_bbox(battery_bbox, width=canvas_width, height=canvas_height),
        "bulb_bboxes": dict(bulb_bboxes),
        "switch_bboxes": dict(switch_bboxes),
        "switch_states": {
            label: ("closed" if bool(scenario.switch_states[label]) else "open")
            for label in SWITCH_LABELS
        },
        "lit_bulbs": list(scenario.lit_bulbs),
        "annotation_bbox_set": [list(item) for item in annotation_bboxes],
    }
    return RenderedSwitchCircuitScene(
        image=image,
        annotation_bboxes=[list(item) for item in annotation_bboxes],
        scene_entities=[dict(entity) for entity in entities],
        render_map=dict(render_map),
        font_family=str(font_family),
    )


def render_switch_circuit(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: SwitchCircuitScenario,
    rendering_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, int],
    defaults: SwitchCircuitDefaults,
) -> RenderedSwitchCircuitScene:
    """Render a complete switch-circuit scene, including background and noise."""

    canvas_width = int(
        params.get(
            "canvas_width",
            group_default(rendering_defaults, "canvas_width", defaults.canvas_width),
        )
    )
    canvas_height = int(
        params.get(
            "canvas_height",
            group_default(rendering_defaults, "canvas_height", defaults.canvas_height),
        )
    )
    background, background_meta, diagram_style, diagram_style_meta = (
        prepare_physics_diagram_style_and_background(
            instance_seed=int(instance_seed),
            params=params,
            scene_id=SCENE_ID,
            canvas_width=canvas_width,
            canvas_height=canvas_height,
            require_grid=True,
        )
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.font",
        params=params,
    )
    rendered = render_switch_circuit_body(
        image=background,
        scenario=scenario,
        font_family=str(font_family),
        style=diagram_style,
        render_defaults=render_defaults,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    render_map = dict(rendered.render_map)
    render_map["technical_diagram_style"] = dict(diagram_style_meta)
    render_map["background_style"] = dict(background_meta)
    render_map["post_image_noise"] = dict(post_noise_meta)
    return RenderedSwitchCircuitScene(
        image=image,
        annotation_bboxes=list(rendered.annotation_bboxes),
        scene_entities=list(rendered.scene_entities),
        render_map=render_map,
        font_family=str(font_family),
    )


__all__ = [
    "POST_IMAGE_NOISE_DEFAULTS",
    "draw_battery",
    "draw_bulb",
    "draw_label_tag",
    "draw_switch",
    "draw_wire",
    "render_switch_circuit",
    "render_switch_circuit_body",
]
