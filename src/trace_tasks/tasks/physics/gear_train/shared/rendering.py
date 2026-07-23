"""Rendering helpers for physics gear-train scenes."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.bbox_projection import bbox_union_many
from trace_tasks.tasks.shared.drawing import draw_arrow, draw_centered_text
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .annotations import bbox, clamp_bbox
from .state import (
    DIRECTION_OPTION_LETTERS,
    GearDirectionChoiceScenario,
    GearDirectionScenario,
    GearSpeedScenario,
    GearTrainDefaults,
    RenderedGearTrainScene,
    SCENE_ID,
)


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)
GEAR_PALETTE: Tuple[Tuple[int, int, int], ...] = (
    (91, 145, 217),
    (221, 139, 69),
    (94, 172, 128),
    (176, 112, 204),
    (209, 91, 110),
    (78, 164, 180),
)


def layout_centers(
    *,
    scenario: GearDirectionScenario | GearSpeedScenario,
    panel: Sequence[float],
    instance_seed: int,
    namespace: str,
) -> Tuple[Tuple[Tuple[float, float], ...], Tuple[float, ...]]:
    """Return scaled gear centers and radii that fit inside the panel."""

    radii = [float(value) for value in scenario.radii_px]
    coords: List[Tuple[float, float]] = [(0.0, 0.0)]
    rng = spawn_rng(int(instance_seed), f"{namespace}.layout")
    if str(scenario.scene_variant) == "straight_chain":
        segment_angles = [0.0 for _ in range(max(0, int(scenario.gear_count) - 1))]
    elif str(scenario.scene_variant) == "staggered_chain":
        sign = float(rng.choice((-1.0, 1.0)))
        segment_angles = [math.radians(sign * (24.0 if index % 2 == 0 else -24.0)) for index in range(max(0, int(scenario.gear_count) - 1))]
    else:
        reverse = float(rng.choice((-1.0, 1.0)))
        count = max(1, int(scenario.gear_count) - 1)
        if count == 1:
            segment_angles = [math.radians(reverse * rng.choice([-18.0, 18.0]))]
        else:
            segment_angles = [
                math.radians(reverse * (-28.0 + 56.0 * float(index) / float(count - 1)))
                for index in range(count)
            ]
    for index, angle in enumerate(segment_angles):
        distance = float(radii[index] + radii[index + 1] + 1.5)
        last_x, last_y = coords[-1]
        coords.append((float(last_x + distance * math.cos(angle)), float(last_y + distance * math.sin(angle))))

    min_x = min(x - radius for (x, _), radius in zip(coords, radii))
    max_x = max(x + radius for (x, _), radius in zip(coords, radii))
    min_y = min(y - radius for (_, y), radius in zip(coords, radii))
    max_y = max(y + radius for (_, y), radius in zip(coords, radii))
    available_w = float(panel[2] - panel[0] - 126.0)
    available_h = float(panel[3] - panel[1] - 142.0)
    raw_w = max(1.0, float(max_x - min_x))
    raw_h = max(1.0, float(max_y - min_y))
    scale = min(1.0, float(available_w / raw_w), float(available_h / raw_h))
    scaled_radii = tuple(float(radius * scale) for radius in radii)
    scaled = [(float(x * scale), float(y * scale)) for x, y in coords]
    min_x = min(x - radius for (x, _), radius in zip(scaled, scaled_radii))
    max_x = max(x + radius for (x, _), radius in zip(scaled, scaled_radii))
    min_y = min(y - radius for (_, y), radius in zip(scaled, scaled_radii))
    max_y = max(y + radius for (_, y), radius in zip(scaled, scaled_radii))
    target_cx = float((panel[0] + panel[2]) * 0.5)
    target_cy = float((panel[1] + panel[3]) * 0.54)
    shift_x = float(target_cx - 0.5 * (min_x + max_x))
    shift_y = float(target_cy - 0.5 * (min_y + max_y))
    centers = tuple((float(x + shift_x), float(y + shift_y)) for x, y in scaled)
    return centers, scaled_radii


def gear_polygon(center: Tuple[float, float], radius: float, teeth: int) -> List[Tuple[float, float]]:
    """Return a simple toothed gear polygon."""

    points: List[Tuple[float, float]] = []
    cx, cy = float(center[0]), float(center[1])
    for index in range(int(teeth) * 2):
        angle = -math.pi / 2.0 + float(index) * math.pi / float(teeth)
        local_radius = float(radius * (1.12 if index % 2 == 0 else 0.94))
        points.append((float(cx + local_radius * math.cos(angle)), float(cy + local_radius * math.sin(angle))))
    return points


def draw_gear(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    radius: float,
    fill_rgb: Tuple[int, int, int],
    style: Any,
) -> List[float]:
    """Draw one gear and return its bbox."""

    cx, cy = float(center[0]), float(center[1])
    teeth = max(12, min(24, int(round(float(radius) / 3.0))))
    stroke = tuple(int(v) for v in style.stroke_rgb)
    polygon = gear_polygon(center, float(radius), int(teeth))
    draw.polygon(polygon, fill=tuple(int(v) for v in fill_rgb), outline=stroke)
    draw.line(polygon + [polygon[0]], fill=stroke, width=3)
    inner_radius = float(radius * 0.54)
    hub_radius = float(radius * 0.20)
    for angle_index in range(6):
        angle = float(angle_index) * math.tau / 6.0
        start = (float(cx + hub_radius * math.cos(angle)), float(cy + hub_radius * math.sin(angle)))
        end = (float(cx + inner_radius * math.cos(angle)), float(cy + inner_radius * math.sin(angle)))
        draw.line([start, end], fill=stroke, width=max(2, int(radius * 0.055)))
    draw.ellipse((cx - inner_radius, cy - inner_radius, cx + inner_radius, cy + inner_radius), outline=stroke, width=3)
    draw.ellipse(
        (cx - hub_radius, cy - hub_radius, cx + hub_radius, cy + hub_radius),
        fill=tuple(int(v) for v in style.panel_alt_fill_rgb),
        outline=stroke,
        width=3,
    )
    return bbox((cx - radius * 1.16, cy - radius * 1.16, cx + radius * 1.16, cy + radius * 1.16))


def draw_label_box(
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
    """Draw one rounded text label and return its bbox."""

    text_bbox = draw.textbbox((0, 0), str(text), font=font, stroke_width=1)
    text_w = float(text_bbox[2] - text_bbox[0])
    text_h = float(text_bbox[3] - text_bbox[1])
    cx, cy = float(center[0]), float(center[1])
    box = bbox((cx - text_w / 2.0 - 12.0, cy - text_h / 2.0 - 7.0, cx + text_w / 2.0 + 12.0, cy + text_h / 2.0 + 7.0))
    fill = fill_rgb if fill_rgb is not None else tuple(int(v) for v in style.label_fill_rgb)
    outline = outline_rgb if outline_rgb is not None else tuple(int(v) for v in style.label_border_rgb)
    text_fill = text_rgb if text_rgb is not None else tuple(int(v) for v in style.label_rgb)
    draw.rounded_rectangle(tuple(box), radius=9, fill=fill, outline=outline, width=2)
    text_draw_bbox = draw_centered_text(
        draw,
        text=str(text),
        center=(cx, cy),
        font=font,
        fill=text_fill,
        stroke_fill=resolve_text_stroke_fill(text_fill),
        stroke_width=1,
    )
    return bbox(bbox_union_many(box, text_draw_bbox, padding=1.0))


def draw_rotation_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    center: Tuple[float, float],
    radius: float,
    direction: str,
    fill: Tuple[int, int, int],
) -> List[float]:
    """Draw the visible input rotation arrow and return its bbox."""

    cx, cy = float(center[0]), float(center[1])
    arc_radius = float(radius + 26.0)
    if str(direction) == "clockwise":
        start_angle = math.radians(-138.0)
        end_angle = math.radians(118.0)
    else:
        start_angle = math.radians(138.0)
        end_angle = math.radians(-118.0)
    steps = 28
    angles = [start_angle + (end_angle - start_angle) * float(index) / float(steps - 1) for index in range(steps)]
    points = [(float(cx + arc_radius * math.cos(angle)), float(cy + arc_radius * math.sin(angle))) for angle in angles]
    draw.line(points[:-1], fill=fill, width=7, joint="curve")
    draw_arrow(draw, start=points[-4], end=points[-1], fill=fill, width=7, head_length_px=20, head_width_px=19)
    return bbox((cx - arc_radius - 13.0, cy - arc_radius - 13.0, cx + arc_radius + 13.0, cy + arc_radius + 13.0))


def panel_bounds(render_defaults: Mapping[str, Any], fallback: GearTrainDefaults) -> tuple[int, int, tuple[float, float, float, float]]:
    """Resolve canvas dimensions and panel bounds."""

    canvas_width = int(render_defaults.get("canvas_width", fallback.canvas_width))
    canvas_height = int(render_defaults.get("canvas_height", fallback.canvas_height))
    panel = (
        float(render_defaults.get("panel_margin_x_px", fallback.panel_margin_x_px)),
        float(render_defaults.get("panel_margin_top_px", fallback.panel_margin_top_px)),
        float(canvas_width - render_defaults.get("panel_margin_x_px", fallback.panel_margin_x_px)),
        float(canvas_height - render_defaults.get("panel_margin_bottom_px", fallback.panel_margin_bottom_px)),
    )
    return int(canvas_width), int(canvas_height), panel


def draw_direction_train_panel(
    *,
    draw: ImageDraw.ImageDraw,
    scenario: GearDirectionScenario,
    panel: Sequence[float],
    instance_seed: int,
    namespace: str,
    label_font: Any,
    title_font: Any,
    diagram_style: Any,
    title_text: str,
    compact: bool = False,
    option_letter: str | None = None,
    entity_prefix: str = "",
) -> tuple[Dict[str, List[float]], List[Dict[str, Any]], Dict[str, Any]]:
    """Draw one gear-direction panel and return role boxes plus diagnostics."""

    draw.rounded_rectangle(
        tuple(panel),
        radius=14 if bool(compact) else 18,
        fill=tuple(int(v) for v in diagram_style.panel_fill_rgb),
        outline=tuple(int(v) for v in diagram_style.panel_border_rgb),
        width=3,
    )
    if option_letter is not None:
        draw_label_box(
            draw,
            text=str(option_letter),
            center=(float(panel[0] + 30.0), float(panel[1] + 30.0)),
            font=title_font,
            style=diagram_style,
            fill_rgb=tuple(int(v) for v in diagram_style.panel_alt_fill_rgb),
        )
    draw_label_box(
        draw,
        text=str(title_text),
        center=(float((panel[0] + panel[2]) * 0.5), float(panel[1] + (28.0 if bool(compact) else 40.0))),
        font=title_font,
        style=diagram_style,
    )
    centers, radii = layout_centers(
        scenario=scenario,
        panel=panel,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    color_offset = int(spawn_rng(int(instance_seed), f"{namespace}.color_offset").randrange(len(GEAR_PALETTE)))
    gear_bboxes: Dict[str, List[float]] = {}
    scene_entities: List[Dict[str, Any]] = []
    for index, (center, radius) in enumerate(zip(centers, radii)):
        gear_id = f"gear_{index + 1}"
        entity_id = f"{entity_prefix}{gear_id}" if entity_prefix else gear_id
        fill = GEAR_PALETTE[(index + color_offset) % len(GEAR_PALETTE)]
        gear_bbox = draw_gear(draw, center=center, radius=float(radius), fill_rgb=fill, style=diagram_style)
        gear_bboxes[gear_id] = gear_bbox
        scene_entities.append(
            {
                "id": str(entity_id),
                "role": "input" if index == 0 else ("output" if index == int(scenario.gear_count) - 1 else "idler"),
                "option_letter": str(option_letter) if option_letter is not None else "",
                "center_px": [round(float(center[0]), 3), round(float(center[1]), 3)],
                "radius_px": round(float(radius), 3),
                "bbox_px": list(gear_bbox),
            }
        )

    input_center = centers[0]
    output_center = centers[-1]
    input_radius = float(radii[0])
    output_radius = float(radii[-1])
    accent = tuple(int(v) for v in diagram_style.accent_rgb)
    arrow_bbox = draw_rotation_arrow(
        draw,
        center=input_center,
        radius=input_radius,
        direction=str(scenario.input_direction),
        fill=accent,
    )
    input_label = draw_label_box(
        draw,
        text="IN" if bool(compact) else "INPUT",
        center=(float(input_center[0]), float(input_center[1] - input_radius - (36.0 if bool(compact) else 58.0))),
        font=label_font,
        style=diagram_style,
    )
    target_fill = (255, 232, 232)
    target_outline = (172, 42, 42)
    target_text = (178, 38, 38)
    output_label = draw_label_box(
        draw,
        text="OUT ?" if bool(compact) else "OUTPUT ?",
        center=(float(output_center[0]), float(output_center[1] + output_radius + (36.0 if bool(compact) else 58.0))),
        font=label_font,
        style=diagram_style,
        fill_rgb=target_fill,
        outline_rgb=target_outline,
        text_rgb=target_text,
    )
    draw.line(
        [
            (float(output_center[0]), float(output_center[1] + output_radius * 0.80)),
            (float(output_center[0]), float(output_center[1] + output_radius + (22.0 if bool(compact) else 34.0))),
        ],
        fill=target_outline,
        width=3,
    )
    train_bbox = bbox(bbox_union_many(*gear_bboxes.values(), padding=8.0))
    annotation_map = {
        "input_gear": bbox(bbox_union_many(gear_bboxes["gear_1"], input_label, padding=5.0)),
        "input_rotation_arrow": bbox(arrow_bbox),
        "output_gear": bbox(bbox_union_many(gear_bboxes[f"gear_{int(scenario.gear_count)}"], output_label, padding=5.0)),
        "gear_train": bbox(train_bbox),
    }
    render_map = {
        "scene_variant": str(scenario.scene_variant),
        "gear_count": int(scenario.gear_count),
        "input_direction": str(scenario.input_direction),
        "output_direction": str(scenario.output_direction),
        "gear_centers_px": [[round(float(x), 3), round(float(y), 3)] for x, y in centers],
        "gear_radii_px": [round(float(value), 3) for value in radii],
        "gear_bboxes_px": {str(key): list(value) for key, value in gear_bboxes.items()},
        "input_rotation_arrow_bbox_px": list(arrow_bbox),
        "panel_bbox_px": bbox(panel),
    }
    return annotation_map, scene_entities, render_map


def render_direction_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: GearDirectionScenario,
    render_defaults: Mapping[str, Any],
    fallback: GearTrainDefaults,
    namespace: str,
) -> RenderedGearTrainScene:
    """Render the direction diagram and preserve gear witness boxes."""

    canvas_width, canvas_height, panel = panel_bounds(render_defaults, fallback)
    background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        require_grid=True,
    )
    draw = ImageDraw.Draw(background)
    font_family = sample_font_family(role="readout", instance_seed=int(instance_seed), namespace=f"{namespace}.font", params=params)
    label_font = load_font(int(render_defaults.get("label_font_size_px", fallback.label_font_size_px)), bold=True, font_family=str(font_family))
    title_font = load_font(int(render_defaults.get("title_font_size_px", fallback.title_font_size_px)), bold=True, font_family=str(font_family))
    annotation_map, scene_entities, render_map = draw_direction_train_panel(
        draw=draw,
        scenario=scenario,
        panel=panel,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
        label_font=label_font,
        title_font=title_font,
        diagram_style=diagram_style,
        title_text="Meshed gear train",
    )
    image, post_noise_meta = apply_post_image_noise(background, instance_seed=int(instance_seed), params=params, default_config=POST_IMAGE_NOISE_DEFAULTS)
    annotation_map = {str(key): clamp_bbox(value, width=int(image.size[0]), height=int(image.size[1])) for key, value in annotation_map.items()}
    render_map = dict(render_map)
    render_map.update(
        {
            "technical_diagram_style": dict(diagram_style_meta),
            "background_style": background_meta,
            "post_image_noise": post_noise_meta,
        }
    )
    return RenderedGearTrainScene(
        image=image,
        annotation_bbox_map=annotation_map,
        scene_entities=scene_entities,
        render_map=render_map,
        font_family=str(font_family),
        diagram_style_meta=dict(diagram_style_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


def render_direction_choice_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: GearDirectionChoiceScenario,
    render_defaults: Mapping[str, Any],
    fallback: GearTrainDefaults,
    namespace: str,
) -> RenderedGearTrainScene:
    """Render a four-panel gear-direction MCQ scene."""

    canvas_width = int(render_defaults.get("direction_choice_canvas_width", render_defaults.get("canvas_width", fallback.canvas_width)))
    canvas_height = int(render_defaults.get("direction_choice_canvas_height", render_defaults.get("canvas_height", fallback.canvas_height)))
    background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        require_grid=True,
    )
    draw = ImageDraw.Draw(background)
    font_family = sample_font_family(role="readout", instance_seed=int(instance_seed), namespace=f"{namespace}.font", params=params)
    label_font = load_font(int(render_defaults.get("direction_choice_label_font_size_px", 17)), bold=True, font_family=str(font_family))
    title_font = load_font(int(render_defaults.get("direction_choice_title_font_size_px", 20)), bold=True, font_family=str(font_family))
    margin_x = float(render_defaults.get("direction_choice_margin_x_px", 42))
    margin_top = float(render_defaults.get("direction_choice_margin_top_px", 36))
    margin_bottom = float(render_defaults.get("direction_choice_margin_bottom_px", 42))
    gap_x = float(render_defaults.get("direction_choice_gap_x_px", 30))
    gap_y = float(render_defaults.get("direction_choice_gap_y_px", 26))
    panel_width = (float(canvas_width) - 2.0 * margin_x - gap_x) / 2.0
    panel_height = (float(canvas_height) - margin_top - margin_bottom - gap_y) / 2.0
    panel_bboxes: Dict[str, List[float]] = {}
    panel_render_maps: Dict[str, Dict[str, Any]] = {}
    scene_entities: List[Dict[str, Any]] = []

    for index, letter in enumerate(DIRECTION_OPTION_LETTERS):
        row = int(index // 2)
        col = int(index % 2)
        left = float(margin_x + float(col) * (panel_width + gap_x))
        top = float(margin_top + float(row) * (panel_height + gap_y))
        panel = bbox((left, top, left + panel_width, top + panel_height))
        panel_bboxes[str(letter)] = list(panel)
        panel_annotation, panel_entities, panel_map = draw_direction_train_panel(
            draw=draw,
            scenario=scenario.panel_scenarios[str(letter)],
            panel=panel,
            instance_seed=int(instance_seed) + 104729 * (index + 1),
            namespace=f"{namespace}.option_{letter}",
            label_font=label_font,
            title_font=title_font,
            diagram_style=diagram_style,
            title_text=f"Panel {letter}",
            compact=True,
            option_letter=str(letter),
            entity_prefix=f"panel_{letter}__",
        )
        scene_entities.extend(panel_entities)
        panel_render_maps[str(letter)] = {
            **dict(panel_map),
            "annotation_bbox_map_px": {str(key): list(value) for key, value in panel_annotation.items()},
        }

    selected_panel = clamp_bbox(
        panel_bboxes[str(scenario.correct_option_letter)],
        width=int(canvas_width),
        height=int(canvas_height),
    )
    annotation_map = {"selected_panel": list(selected_panel)}
    image, post_noise_meta = apply_post_image_noise(
        background,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_map = {
        str(key): clamp_bbox(value, width=int(image.size[0]), height=int(image.size[1]))
        for key, value in annotation_map.items()
    }
    render_map = {
        "target_direction": str(scenario.target_direction),
        "correct_option_letter": str(scenario.correct_option_letter),
        "answer_support": list(DIRECTION_OPTION_LETTERS),
        "panel_bboxes_px": {str(key): list(value) for key, value in panel_bboxes.items()},
        "selected_panel_bbox_px": list(annotation_map["selected_panel"]),
        "panel_output_directions": {
            str(letter): str(panel_scenario.output_direction)
            for letter, panel_scenario in scenario.panel_scenarios.items()
        },
        "panel_input_directions": {
            str(letter): str(panel_scenario.input_direction)
            for letter, panel_scenario in scenario.panel_scenarios.items()
        },
        "panel_gear_counts": {
            str(letter): int(panel_scenario.gear_count)
            for letter, panel_scenario in scenario.panel_scenarios.items()
        },
        "panels": {str(key): dict(value) for key, value in panel_render_maps.items()},
        "technical_diagram_style": dict(diagram_style_meta),
        "background_style": background_meta,
        "post_image_noise": post_noise_meta,
    }
    return RenderedGearTrainScene(
        image=image,
        annotation_bbox_map=annotation_map,
        scene_entities=scene_entities,
        render_map=render_map,
        font_family=str(font_family),
        diagram_style_meta=dict(diagram_style_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


def render_speed_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: GearSpeedScenario,
    render_defaults: Mapping[str, Any],
    fallback: GearTrainDefaults,
    namespace: str,
) -> RenderedGearTrainScene:
    """Render the speed-ratio diagram and preserve readable label witnesses."""

    canvas_width, canvas_height, panel = panel_bounds(render_defaults, fallback)
    background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        require_grid=True,
    )
    draw = ImageDraw.Draw(background)
    font_family = sample_font_family(role="readout", instance_seed=int(instance_seed), namespace=f"{namespace}.font", params=params)
    label_font = load_font(int(render_defaults.get("label_font_size_px", fallback.label_font_size_px)), bold=True, font_family=str(font_family))
    small_font = load_font(int(render_defaults.get("tooth_label_font_size_px", 20)), bold=True, font_family=str(font_family))
    title_font = load_font(int(render_defaults.get("title_font_size_px", fallback.title_font_size_px)), bold=True, font_family=str(font_family))
    draw.rounded_rectangle(panel, radius=18, fill=tuple(int(v) for v in diagram_style.panel_fill_rgb), outline=tuple(int(v) for v in diagram_style.panel_border_rgb), width=3)
    draw_label_box(draw, text="Gear ratio train", center=(float(canvas_width * 0.5), float(panel[1] + 40.0)), font=title_font, style=diagram_style)
    centers, radii = layout_centers(scenario=scenario, panel=panel, instance_seed=int(instance_seed), namespace=str(namespace))
    color_offset = int(spawn_rng(int(instance_seed), f"{namespace}.color_offset").randrange(len(GEAR_PALETTE)))
    tooth_counts = (int(scenario.input_teeth),) + tuple(int(value) for value in scenario.idler_teeth) + (int(scenario.output_teeth),)
    gear_bboxes: Dict[str, List[float]] = {}
    tooth_label_bboxes: Dict[str, List[float]] = {}
    scene_entities: List[Dict[str, Any]] = []
    for index, (center, radius, tooth_count) in enumerate(zip(centers, radii, tooth_counts)):
        gear_id = f"gear_{index + 1}"
        fill = GEAR_PALETTE[(index + color_offset) % len(GEAR_PALETTE)]
        gear_bbox = draw_gear(draw, center=center, radius=float(radius), fill_rgb=fill, style=diagram_style)
        tooth_bbox = draw_label_box(draw, text=f"T={int(tooth_count)}", center=(float(center[0]), float(center[1])), font=small_font, style=diagram_style)
        gear_bboxes[gear_id] = bbox(bbox_union_many(gear_bbox, tooth_bbox, padding=2.0))
        tooth_label_bboxes[gear_id] = tooth_bbox
        scene_entities.append(
            {
                "id": gear_id,
                "role": "input" if index == 0 else ("output" if index == int(scenario.gear_count) - 1 else "idler"),
                "center_px": [round(float(center[0]), 3), round(float(center[1]), 3)],
                "radius_px": round(float(radius), 3),
                "tooth_count": int(tooth_count),
                "bbox_px": list(gear_bboxes[gear_id]),
            }
        )

    input_center = centers[0]
    output_center = centers[-1]
    input_radius = float(radii[0])
    output_radius = float(radii[-1])
    input_label = draw_label_box(
        draw,
        text=f"input = {int(scenario.input_rpm)} rpm",
        center=(float(input_center[0]), float(input_center[1] - input_radius - 48.0)),
        font=label_font,
        style=diagram_style,
    )
    target_fill = (255, 232, 232)
    target_outline = (172, 42, 42)
    target_text = (178, 38, 38)
    output_label = draw_label_box(
        draw,
        text="output = ? rpm",
        center=(float(output_center[0]), float(output_center[1] + output_radius + 48.0)),
        font=label_font,
        style=diagram_style,
        fill_rgb=target_fill,
        outline_rgb=target_outline,
        text_rgb=target_text,
    )
    draw.line(
        [
            (float(output_center[0]), float(output_center[1] + output_radius * 0.80)),
            (float(output_center[0]), float(output_center[1] + output_radius + 28.0)),
        ],
        fill=target_outline,
        width=3,
    )
    train_bbox = bbox(bbox_union_many(*gear_bboxes.values(), padding=8.0))
    annotation_map = {
        "input_gear": bbox(bbox_union_many(gear_bboxes["gear_1"], input_label, padding=5.0)),
        "output_gear": bbox(bbox_union_many(gear_bboxes[f"gear_{int(scenario.gear_count)}"], output_label, padding=5.0)),
    }
    image, post_noise_meta = apply_post_image_noise(background, instance_seed=int(instance_seed), params=params, default_config=POST_IMAGE_NOISE_DEFAULTS)
    annotation_map = {str(key): clamp_bbox(value, width=int(image.size[0]), height=int(image.size[1])) for key, value in annotation_map.items()}
    render_map = {
        "scene_variant": str(scenario.scene_variant),
        "gear_count": int(scenario.gear_count),
        "input_teeth": int(scenario.input_teeth),
        "output_teeth": int(scenario.output_teeth),
        "idler_teeth": [int(value) for value in scenario.idler_teeth],
        "input_rpm": int(scenario.input_rpm),
        "output_rpm": int(scenario.output_rpm),
        "speed_relation": str(scenario.speed_relation),
        "ratio_numerator": int(scenario.input_rpm * scenario.input_teeth),
        "ratio_denominator": int(scenario.output_teeth),
        "gear_centers_px": [[round(float(x), 3), round(float(y), 3)] for x, y in centers],
        "gear_radii_px": [round(float(value), 3) for value in radii],
        "gear_bboxes_px": {str(key): list(value) for key, value in gear_bboxes.items()},
        "gear_train_bbox_px": list(train_bbox),
        "tooth_label_bboxes_px": {str(key): list(value) for key, value in tooth_label_bboxes.items()},
        "technical_diagram_style": dict(diagram_style_meta),
        "background_style": background_meta,
        "post_image_noise": post_noise_meta,
    }
    return RenderedGearTrainScene(
        image=image,
        annotation_bbox_map=annotation_map,
        scene_entities=scene_entities,
        render_map=render_map,
        font_family=str(font_family),
        diagram_style_meta=dict(diagram_style_meta),
        background_meta=dict(background_meta),
        post_noise_meta=dict(post_noise_meta),
    )


__all__ = [
    "render_direction_choice_scene",
    "render_direction_scene",
    "render_speed_scene",
]
