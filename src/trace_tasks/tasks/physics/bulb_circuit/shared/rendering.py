"""Rendering primitives for bulb-circuit diagrams."""

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
    DEFAULT_RENDERING,
    SCENE_ID,
    SCENE_NAMESPACE,
    BulbScenario,
    BulbSpec,
    RenderedBulbScene,
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
    spec: BulbSpec,
    style: Any,
    accent_rgb: Tuple[int, int, int],
    label_font: Any,
    label_offset: Tuple[float, float] | None = None,
) -> List[float]:
    """Draw one bulb symbol and its resistance tag."""

    cx, cy = float(center[0]), float(center[1])
    radius = 36.0
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
    base = bbox((cx - 18.0, cy + 22.0, cx + 18.0, cy + 39.0))
    draw.rounded_rectangle(
        tuple(float(value) for value in base),
        radius=4,
        fill=tuple(int(value) for value in style.muted_fill_rgb),
        outline=stroke,
        width=3,
    )
    filament_y = cy - 1.0
    filament_rgb = tuple(
        max(0, min(255, int((0.35 * accent_rgb[idx]) + (0.65 * stroke[idx]))))
        for idx in range(3)
    )
    filament_points = [
        (cx - 20.0, filament_y + 8.0),
        (cx - 12.0, filament_y - 8.0),
        (cx - 4.0, filament_y + 8.0),
        (cx + 4.0, filament_y - 8.0),
        (cx + 12.0, filament_y + 8.0),
        (cx + 20.0, filament_y - 8.0),
    ]
    draw.line(filament_points, fill=filament_rgb, width=4, joint="curve")
    draw.line((cx - 15.0, cy + 20.0, cx - 20.0, filament_y + 8.0), fill=stroke, width=3)
    draw.line((cx + 15.0, cy + 20.0, cx + 20.0, filament_y - 8.0), fill=stroke, width=3)
    label_text = f"{spec.label}={int(spec.resistance_ohm)} ohm"
    if label_offset is None:
        label_offset = (0.0, radius + 28.0)
    label_bbox = draw_label_tag(
        draw,
        text=label_text,
        center=(cx + float(label_offset[0]), cy + float(label_offset[1])),
        font=label_font,
        style=style,
    )
    return union_bbox(bulb_bbox, base, label_bbox)


def draw_battery(
    draw: ImageDraw.ImageDraw,
    *,
    x: float,
    y_top: float,
    y_bottom: float,
    style: Any,
    font: Any,
    wire_width: int,
) -> Dict[str, List[float]]:
    """Draw the ideal battery symbol."""

    stroke = tuple(int(value) for value in style.stroke_rgb)
    guide = tuple(int(value) for value in style.guide_rgb)
    mid_y = (float(y_top) + float(y_bottom)) * 0.5
    plate_top = mid_y - 22.0
    plate_bottom = mid_y + 24.0
    draw.line((x, y_top, x, plate_top - 16.0), fill=stroke, width=int(wire_width))
    draw.line((x, plate_bottom + 16.0, x, y_bottom), fill=stroke, width=int(wire_width))
    draw.line((x - 38.0, plate_top, x + 38.0, plate_top), fill=stroke, width=max(3, int(wire_width) + 2))
    draw.line((x - 25.0, plate_bottom, x + 25.0, plate_bottom), fill=stroke, width=max(3, int(wire_width) + 2))
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
    return {"battery": union_bbox((x - 42.0, plate_top - 20.0, x + 66.0, plate_bottom + 20.0), plus_box, minus_box)}


def draw_series_circuit(
    draw: ImageDraw.ImageDraw,
    *,
    scenario: BulbScenario,
    style: Any,
    accent_rgb: Tuple[int, int, int],
    label_font: Any,
    battery_font: Any,
    render_defaults: Mapping[str, Any],
) -> Dict[str, List[float]]:
    """Draw the five bulbs in one series loop."""

    wire_width = int(render_defaults.get("wire_width_px", DEFAULT_RENDERING.wire_width_px))
    stroke = tuple(int(value) for value in style.stroke_rgb)
    left_x = 166.0
    right_x = 1108.0
    top_y = 250.0
    bottom_y = 510.0
    centers = [
        (304.0, top_y),
        (482.0, top_y),
        (660.0, top_y),
        (838.0, top_y),
        (1016.0, top_y),
    ]
    draw_battery(
        draw,
        x=left_x,
        y_top=top_y,
        y_bottom=bottom_y,
        style=style,
        font=battery_font,
        wire_width=wire_width,
    )
    draw.line((left_x, bottom_y, right_x, bottom_y, right_x, top_y), fill=stroke, width=wire_width)
    draw.line((left_x, top_y, right_x, top_y), fill=stroke, width=wire_width)
    annotation: Dict[str, List[float]] = {}
    for spec, center in zip(scenario.bulbs, centers):
        annotation[str(spec.label)] = draw_bulb(
            draw,
            center=center,
            spec=spec,
            style=style,
            accent_rgb=accent_rgb,
            label_font=label_font,
        )
    return annotation


def draw_parallel_circuit(
    draw: ImageDraw.ImageDraw,
    *,
    scenario: BulbScenario,
    style: Any,
    accent_rgb: Tuple[int, int, int],
    label_font: Any,
    battery_font: Any,
    render_defaults: Mapping[str, Any],
) -> Dict[str, List[float]]:
    """Draw five unequal bulbs in parallel branches."""

    wire_width = int(render_defaults.get("wire_width_px", DEFAULT_RENDERING.wire_width_px))
    stroke = tuple(int(value) for value in style.stroke_rgb)
    left_x = 244.0
    right_x = 1080.0
    branch_ys = (156.0, 252.0, 348.0, 444.0, 540.0)
    draw_battery(
        draw,
        x=left_x,
        y_top=branch_ys[0],
        y_bottom=branch_ys[-1],
        style=style,
        font=battery_font,
        wire_width=wire_width,
    )
    draw.line((right_x, branch_ys[0], right_x, branch_ys[-1]), fill=stroke, width=wire_width)
    annotation: Dict[str, List[float]] = {}
    for spec, y in zip(scenario.bulbs, branch_ys):
        draw.line((left_x, y, right_x, y), fill=stroke, width=wire_width)
        annotation[str(spec.label)] = draw_bulb(
            draw,
            center=((left_x + right_x) * 0.5, y),
            spec=spec,
            style=style,
            accent_rgb=accent_rgb,
            label_font=label_font,
            label_offset=(122.0, 0.0),
        )
    return annotation


def draw_mixed_circuit(
    draw: ImageDraw.ImageDraw,
    *,
    scenario: BulbScenario,
    style: Any,
    accent_rgb: Tuple[int, int, int],
    label_font: Any,
    battery_font: Any,
    render_defaults: Mapping[str, Any],
) -> Dict[str, List[float]]:
    """Draw two parallel branches with unequal bulb counts."""

    wire_width = int(render_defaults.get("wire_width_px", DEFAULT_RENDERING.wire_width_px))
    stroke = tuple(int(value) for value in style.stroke_rgb)
    left_x = 244.0
    right_x = 1080.0
    top_y = 238.0
    bottom_y = 480.0
    single_y = top_y if scenario.branch_single_position == "top" else bottom_y
    pair_y = bottom_y if scenario.branch_single_position == "top" else top_y
    short_specs = scenario.bulbs[:2]
    long_specs = scenario.bulbs[2:]
    draw_battery(
        draw,
        x=left_x,
        y_top=top_y,
        y_bottom=bottom_y,
        style=style,
        font=battery_font,
        wire_width=wire_width,
    )
    draw.line((right_x, top_y, right_x, bottom_y), fill=stroke, width=wire_width)
    draw.line((left_x, single_y, right_x, single_y), fill=stroke, width=wire_width)
    draw.line((left_x, pair_y, right_x, pair_y), fill=stroke, width=wire_width)
    annotation: Dict[str, List[float]] = {}
    for spec, x in zip(short_specs, (560.0, 764.0)):
        annotation[str(spec.label)] = draw_bulb(
            draw,
            center=(x, single_y),
            spec=spec,
            style=style,
            accent_rgb=accent_rgb,
            label_font=label_font,
        )
    for spec, x in zip(long_specs, (438.0, 662.0, 886.0)):
        annotation[str(spec.label)] = draw_bulb(
            draw,
            center=(x, pair_y),
            spec=spec,
            style=style,
            accent_rgb=accent_rgb,
            label_font=label_font,
        )
    return annotation


def draw_bulb_circuit(
    *,
    image: Image.Image,
    scenario: BulbScenario,
    font_family: str,
    style: Any,
    render_defaults: Mapping[str, Any],
) -> tuple[Image.Image, Dict[str, List[float]], Dict[str, Any], List[Dict[str, Any]]]:
    """Draw the bulb-circuit diagram and return projected bulb boxes."""

    draw = ImageDraw.Draw(image)
    canvas_width, canvas_height = image.size
    title_font = load_font(
        int(render_defaults.get("title_font_size_px", DEFAULT_RENDERING.title_font_size_px)),
        bold=True,
        font_family=font_family,
    )
    label_font = load_font(
        int(render_defaults.get("bulb_label_font_size_px", DEFAULT_RENDERING.bulb_label_font_size_px)),
        bold=True,
        font_family=font_family,
    )
    battery_font = load_font(24, bold=True, font_family=font_family)
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
        text="Ideal battery with labeled bulbs",
        center=((panel[0] + panel[2]) * 0.5, panel[1] + 34.0),
        font=title_font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=1,
    )
    accent_rgb = named_color(str(scenario.accent_color_name))
    if scenario.scene_variant == "series_unequal":
        raw_annotation = draw_series_circuit(
            draw,
            scenario=scenario,
            style=style,
            accent_rgb=accent_rgb,
            label_font=label_font,
            battery_font=battery_font,
            render_defaults=render_defaults,
        )
    elif scenario.scene_variant == "parallel_unequal":
        raw_annotation = draw_parallel_circuit(
            draw,
            scenario=scenario,
            style=style,
            accent_rgb=accent_rgb,
            label_font=label_font,
            battery_font=battery_font,
            render_defaults=render_defaults,
        )
    else:
        raw_annotation = draw_mixed_circuit(
            draw,
            scenario=scenario,
            style=style,
            accent_rgb=accent_rgb,
            label_font=label_font,
            battery_font=battery_font,
            render_defaults=render_defaults,
        )
    annotation = normalize_annotation_bbox_map(
        raw_annotation,
        width=int(canvas_width),
        height=int(canvas_height),
    )
    scene_entities = [
        {
            "entity_id": str(spec.label),
            "entity_type": "bulb",
            "slot_id": str(spec.slot_id),
            "label": str(spec.label),
            "resistance_ohm": int(spec.resistance_ohm),
            "relative_power": float(spec.relative_power),
            "bbox_px": list(annotation[str(spec.label)]),
        }
        for spec in scenario.bulbs
    ]
    render_map = {
        "panel_bbox": bbox(panel),
        "scene_variant": str(scenario.scene_variant),
        "target_direction": str(scenario.target_direction),
        "branch_single_position": str(scenario.branch_single_position),
        "bulb_bboxes": dict(annotation),
        "bulb_specs": [
            {
                "slot_id": str(spec.slot_id),
                "label": str(spec.label),
                "resistance_ohm": int(spec.resistance_ohm),
                "relative_power": round(float(spec.relative_power), 8),
            }
            for spec in scenario.bulbs
        ],
        "correct_label": str(scenario.correct_label),
        "brightest_label": str(scenario.brightest_label),
        "dimmest_label": str(scenario.dimmest_label),
        "annotation_bbox_map": dict(annotation),
    }
    return image, annotation, render_map, scene_entities


def render_bulb_circuit(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: BulbScenario,
    render_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> RenderedBulbScene:
    """Render the full bulb-circuit scene and attach style/noise metadata."""

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
    rendered, annotation_map, render_map, scene_entities = draw_bulb_circuit(
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
    return RenderedBulbScene(
        image=image,
        annotation_bbox_map={
            str(key): clip_bbox(value, width=int(canvas_width), height=int(canvas_height))
            for key, value in annotation_map.items()
        },
        scene_entities=scene_entities,
        render_map=render_map,
        font_family=str(font_family),
    )


__all__ = [
    "draw_battery",
    "draw_bulb",
    "draw_bulb_circuit",
    "draw_label_tag",
    "draw_mixed_circuit",
    "draw_parallel_circuit",
    "draw_series_circuit",
    "render_bulb_circuit",
]
