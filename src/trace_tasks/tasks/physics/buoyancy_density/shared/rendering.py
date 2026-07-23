"""Rendering primitives for buoyancy-density diagrams."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.drawing import draw_centered_text
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.render_variation import resolve_layout_jitter, resolve_render_int
from trace_tasks.tasks.shared.text_legibility import draw_traced_text
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .annotations import bbox, normalize_annotation_bbox_map
from .formulas import format_density
from .state import (
    DEFAULT_RENDERING,
    SCENE_ID,
    SCENE_NAMESPACE,
    BuoyancyRenderDefaults,
    BuoyancyScenario,
    RenderedBuoyancyScene,
)


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def _draw_label(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font: Any,
    fill: Tuple[int, int, int],
) -> List[float]:
    """Draw a required readable label and return its bbox."""

    text_bbox = draw.textbbox((float(xy[0]), float(xy[1])), str(text), font=font, stroke_width=1)
    draw_traced_text(
        draw,
        xy=(float(xy[0]), float(xy[1])),
        text=str(text),
        font=font,
        fill_rgb=fill,
        stroke_width=1,
        stroke_rgb=resolve_text_stroke_fill(fill),
        role="readout",
        required=True,
    )
    return bbox(tuple(float(value) for value in text_bbox))


def _object_geometry(render_defaults: Mapping[str, Any], scenario: BuoyancyScenario) -> Dict[str, float]:
    """Resolve floating-object dimensions from the visible fraction support."""

    den = int(scenario.fraction_den)
    configured_part_height = float(render_defaults["object_part_height_px"])
    max_height = 282.0
    part_height = min(configured_part_height, max_height / float(max(1, den)))
    object_height = float(part_height * den)
    waterline_y = float(render_defaults["waterline_y_px"])
    top = float(waterline_y - ((den - int(scenario.fraction_num)) * part_height))
    bottom = float(top + object_height)
    center_x = float(render_defaults["object_center_x_px"])
    width = float(render_defaults["object_width_px"])
    return {
        "left": float(center_x - width / 2.0),
        "right": float(center_x + width / 2.0),
        "top": float(top),
        "bottom": float(bottom),
        "width": float(width),
        "height": float(object_height),
        "part_height": float(part_height),
        "waterline_y": float(waterline_y),
    }


def _draw_floating_object(
    draw: ImageDraw.ImageDraw,
    *,
    scenario: BuoyancyScenario,
    geometry: Mapping[str, float],
    style: Any,
    object_rgb: Tuple[int, int, int],
) -> Dict[str, Any]:
    """Draw the divided floating object and fraction witness marker."""

    left = float(geometry["left"])
    right = float(geometry["right"])
    top = float(geometry["top"])
    bottom = float(geometry["bottom"])
    waterline_y = float(geometry["waterline_y"])
    part_height = float(geometry["part_height"])
    radius = 8
    if str(scenario.object_shape) == "rounded_block":
        radius = 18

    outline = tuple(int(value) for value in style.stroke_rgb)
    submerged_rgb = tuple(max(0, int(value) - 38) for value in object_rgb)
    draw.rounded_rectangle((left, top, right, bottom), radius=radius, fill=object_rgb, outline=outline, width=4)
    draw.rectangle((left + 3, waterline_y, right - 3, bottom - 3), fill=submerged_rgb)
    draw.rounded_rectangle((left, top, right, bottom), radius=radius, outline=outline, width=4)
    for index in range(1, int(scenario.fraction_den)):
        y = float(top + index * part_height)
        draw.line((left + 5, y, right - 5, y), fill=outline, width=2)

    marker_x = right + 22.0
    marker_rgb = tuple(int(value) for value in style.accent_rgb)
    draw.line((marker_x, top, marker_x, bottom), fill=marker_rgb, width=4)
    for index in range(0, int(scenario.fraction_den) + 1):
        y = float(top + index * part_height)
        draw.line((marker_x - 10, y, marker_x + 10, y), fill=marker_rgb, width=3)
    draw.line((marker_x - 13, waterline_y, marker_x + 13, waterline_y), fill=marker_rgb, width=5)
    return {
        "floating_object_bbox": bbox((left, top, right, bottom)),
        "fraction_marker_bbox": bbox((marker_x - 16, top - 4, marker_x + 16, bottom + 4)),
        "object_geometry": {
            "left": round(left, 3),
            "right": round(right, 3),
            "top": round(top, 3),
            "bottom": round(bottom, 3),
            "waterline_y": round(waterline_y, 3),
            "part_height": round(part_height, 3),
        },
    }


def _draw_scene_body(
    *,
    background: Image.Image,
    scenario: BuoyancyScenario,
    render_defaults: Mapping[str, Any],
    font_family: str,
    style: Any,
) -> RenderedBuoyancyScene:
    """Draw tank, object, labels, and role-keyed visual witnesses."""

    image = background
    draw = ImageDraw.Draw(image)
    canvas_width, canvas_height = image.size
    label_font = load_font(int(render_defaults["label_font_size_px"]), bold=True, font_family=font_family)
    small_font = load_font(int(render_defaults["small_font_size_px"]), bold=True, font_family=font_family)
    title_font = load_font(int(render_defaults["title_font_size_px"]), bold=True, font_family=font_family)
    text_rgb = tuple(int(value) for value in style.label_rgb)
    stroke_rgb = tuple(int(value) for value in style.stroke_rgb)
    guide_rgb = tuple(int(value) for value in style.guide_rgb)

    panel = (
        float(render_defaults["panel_left_px"]),
        float(render_defaults["panel_top_px"]),
        float(canvas_width - int(render_defaults["panel_right_margin_px"])),
        float(canvas_height - int(render_defaults["panel_bottom_margin_px"])),
    )
    draw.rounded_rectangle(
        panel,
        radius=18,
        fill=tuple(style.panel_fill_rgb),
        outline=tuple(style.panel_border_rgb),
        width=3,
    )

    tank_left = float(render_defaults["tank_left_px"])
    tank_top = float(render_defaults["tank_top_px"])
    tank_width = float(render_defaults["tank_width_px"])
    tank_height = float(render_defaults["tank_height_px"])
    if str(scenario.scene_variant) == "wide_tank":
        tank_left -= 40
        tank_width += 86
    elif str(scenario.scene_variant) == "beaker_tank":
        tank_left += 20
        tank_width -= 42
    tank_right = float(tank_left + tank_width)
    tank_bottom = float(tank_top + tank_height)
    waterline_y = float(render_defaults["waterline_y_px"])

    liquid_palette = (
        (92, 169, 219),
        (83, 184, 160),
        (118, 154, 224),
        (105, 181, 203),
    )
    object_palette = (
        (225, 151, 75),
        (211, 118, 93),
        (177, 139, 221),
        (219, 180, 82),
    )
    liquid_rgb = spawn_rng(
        int(scenario.object_density_tenths),
        f"{SCENE_NAMESPACE}.liquid_color.{int(scenario.liquid_density_tenths)}",
    ).choice(liquid_palette)
    object_rgb = spawn_rng(
        int(scenario.liquid_density_tenths),
        f"{SCENE_NAMESPACE}.object_color",
    ).choice(object_palette)

    if str(scenario.scene_variant) == "beaker_tank":
        lip = 34.0
        beaker_points = [
            (tank_left + lip, tank_top),
            (tank_right - lip, tank_top),
            (tank_right - 10, tank_bottom),
            (tank_left + 10, tank_bottom),
        ]
        draw.polygon(beaker_points, fill=(239, 248, 252), outline=stroke_rgb)
        draw.line((tank_left + lip, tank_top, tank_right - lip, tank_top), fill=stroke_rgb, width=5)
        liquid_points = [
            (tank_left + 22, waterline_y),
            (tank_right - 22, waterline_y),
            (tank_right - 14, tank_bottom - 10),
            (tank_left + 14, tank_bottom - 10),
        ]
        draw.polygon(liquid_points, fill=liquid_rgb)
        draw.line((tank_left + 22, waterline_y, tank_right - 22, waterline_y), fill=tuple(max(0, v - 45) for v in liquid_rgb), width=5)
        tank_box = bbox((tank_left + 8, tank_top, tank_right - 8, tank_bottom))
        waterline_box = bbox((tank_left + 22, waterline_y - 7, tank_right - 22, waterline_y + 7))
    else:
        draw.rounded_rectangle(
            (tank_left, tank_top, tank_right, tank_bottom),
            radius=20,
            fill=(239, 248, 252),
            outline=stroke_rgb,
            width=5,
        )
        draw.rectangle((tank_left + 10, waterline_y, tank_right - 10, tank_bottom - 10), fill=liquid_rgb)
        draw.line((tank_left + 10, waterline_y, tank_right - 10, waterline_y), fill=tuple(max(0, v - 45) for v in liquid_rgb), width=5)
        tank_box = bbox((tank_left, tank_top, tank_right, tank_bottom))
        waterline_box = bbox((tank_left + 10, waterline_y - 7, tank_right - 10, waterline_y + 7))

    for offset in (0.25, 0.50, 0.75):
        y = tank_top + tank_height * offset
        draw.line((tank_left + 16, y, tank_right - 16, y), fill=guide_rgb, width=1)

    obj_geometry = _object_geometry(render_defaults, scenario)
    object_render = _draw_floating_object(
        draw,
        scenario=scenario,
        geometry=obj_geometry,
        style=style,
        object_rgb=object_rgb,
    )
    draw.line(
        (
            max(tank_left + 12, float(obj_geometry["left"]) - 38),
            waterline_y,
            min(tank_right - 12, float(obj_geometry["right"]) + 68),
            waterline_y,
        ),
        fill=tuple(max(0, v - 55) for v in liquid_rgb),
        width=3,
    )

    draw_centered_text(
        draw,
        text="floating object",
        center=((float(obj_geometry["left"]) + float(obj_geometry["right"])) / 2.0, float(obj_geometry["top"]) - 28),
        font=small_font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=1,
    )
    title_box = draw_centered_text(
        draw,
        text="Buoyancy density diagram",
        center=(canvas_width * 0.5, panel[1] + 32),
        font=title_font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=1,
    )
    density_text = f"rho_liquid = {format_density(scenario.liquid_density_tenths)} g/cm^3"
    density_box = _draw_label(
        draw,
        (tank_right - 290, tank_top - 56),
        density_text,
        label_font,
        text_rgb,
    )
    surface_rgb = tuple(max(0, v - 45) for v in liquid_rgb)
    draw_centered_text(
        draw,
        text="liquid surface",
        center=(tank_left + 92, waterline_y - 22),
        font=small_font,
        fill=surface_rgb,
        stroke_fill=resolve_text_stroke_fill(surface_rgb),
        stroke_width=1,
    )

    annotation_map = normalize_annotation_bbox_map(
        {
            "floating_object": list(object_render["floating_object_bbox"]),
            "waterline": list(waterline_box),
            "fluid_density_label": list(density_box),
            "submerged_fraction_marker": list(object_render["fraction_marker_bbox"]),
        },
        width=int(canvas_width),
        height=int(canvas_height),
    )
    scene_entities = [
        {"entity_id": "tank", "entity_type": "container", "bbox_px": tank_box},
        {"entity_id": "floating_object", "entity_type": "floating_object", "bbox_px": list(object_render["floating_object_bbox"])},
        {"entity_id": "waterline", "entity_type": "liquid_surface", "bbox_px": list(waterline_box)},
        {"entity_id": "fluid_density_label", "entity_type": "density_label", "bbox_px": list(density_box)},
        {"entity_id": "submerged_fraction_marker", "entity_type": "fraction_marker", "bbox_px": list(object_render["fraction_marker_bbox"])},
    ]
    render_map = {
        "scene_variant": str(scenario.scene_variant),
        "object_shape": str(scenario.object_shape),
        "tank_bbox_px": list(tank_box),
        "title_bbox_px": list(title_box),
        "waterline_bbox_px": list(waterline_box),
        "floating_object_bbox_px": list(object_render["floating_object_bbox"]),
        "fraction_marker_bbox_px": list(object_render["fraction_marker_bbox"]),
        "fluid_density_label_bbox_px": list(density_box),
        "object_geometry": dict(object_render["object_geometry"]),
        "submerged_fraction": {
            "numerator": int(scenario.fraction_num),
            "denominator": int(scenario.fraction_den),
            "value": float(scenario.fraction_num) / float(scenario.fraction_den),
        },
        "liquid_density_g_cm3": float(scenario.liquid_density_tenths) / 10.0,
        "object_density_g_cm3": float(scenario.object_density_tenths) / 10.0,
        "annotation_keyed_bboxes_px": dict(annotation_map),
    }
    return RenderedBuoyancyScene(
        image=image,
        annotation_bbox_map={str(key): list(value) for key, value in annotation_map.items()},
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
        render_spec={},
        font_family=str(font_family),
    )


def _resolve_layout_placement(
    *,
    render_defaults: Mapping[str, Any],
    jitter_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    canvas_width: int,
    canvas_height: int,
    namespace: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply bounded whole-diagram jitter after base geometry is known."""

    content_left = float(render_defaults["panel_left_px"])
    content_top = float(render_defaults["panel_top_px"])
    content_right = float(canvas_width - int(render_defaults["panel_right_margin_px"]))
    content_bottom = float(canvas_height - int(render_defaults["panel_bottom_margin_px"]))
    jitter = resolve_layout_jitter(
        params,
        jitter_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.layout",
    )
    min_margin = int(jitter.get("min_margin_px", 18))
    requested_dx = int(jitter.get("requested_dx_px", 0))
    requested_dy = int(jitter.get("requested_dy_px", 0))
    min_dx = int(math.ceil(float(min_margin) - content_left))
    max_dx = int(math.floor(float(canvas_width) - float(min_margin) - content_right))
    min_dy = int(math.ceil(float(min_margin) - content_top))
    max_dy = int(math.floor(float(canvas_height) - float(min_margin) - content_bottom))
    if min_dx > max_dx:
        min_dx = max_dx = 0
    if min_dy > max_dy:
        min_dy = max_dy = 0
    if not bool(jitter.get("enabled", False)):
        requested_dx = 0
        requested_dy = 0
    dx = max(min_dx, min(max_dx, requested_dx))
    dy = max(min_dy, min(max_dy, requested_dy))
    adjusted = dict(render_defaults)
    for key in ("panel_left_px", "tank_left_px", "object_center_x_px"):
        adjusted[key] = int(adjusted[key]) + int(dx)
    for key in ("panel_top_px", "tank_top_px", "waterline_y_px"):
        adjusted[key] = int(adjusted[key]) + int(dy)
    placement = dict(jitter)
    placement.update(
        {
            "mode": "whole_buoyancy_diagram_offset",
            "content_bbox_px": bbox((content_left, content_top, content_right, content_bottom)),
            "final_content_bbox_px": bbox((content_left + dx, content_top + dy, content_right + dx, content_bottom + dy)),
            "canvas_size_px": [int(canvas_width), int(canvas_height)],
            "available_offset_x_px": [int(min_dx), int(max_dx)],
            "available_offset_y_px": [int(min_dy), int(max_dy)],
            "sampled_offset_px": [int(requested_dx), int(requested_dy)],
            "final_offset_px": [int(dx), int(dy)],
        }
    )
    return adjusted, placement


def _resolve_render_defaults(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> Dict[str, Any]:
    """Resolve scalar rendering defaults with configured random variation."""

    fallback = DEFAULT_RENDERING
    keys = (
        "panel_left_px",
        "panel_top_px",
        "panel_right_margin_px",
        "panel_bottom_margin_px",
        "tank_left_px",
        "tank_top_px",
        "tank_width_px",
        "tank_height_px",
        "waterline_y_px",
        "object_center_x_px",
        "object_width_px",
        "object_part_height_px",
        "label_font_size_px",
        "small_font_size_px",
        "title_font_size_px",
        "label_stroke_width_px",
        "marker_width_px",
    )
    resolved: Dict[str, Any] = {}
    for key in keys:
        resolved[str(key)] = resolve_render_int(
            params,
            render_defaults,
            str(key),
            int(getattr(fallback, str(key))),
            instance_seed=int(instance_seed),
            namespace=str(namespace),
        )
    return resolved


def render_buoyancy_density(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: BuoyancyScenario,
    render_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> RenderedBuoyancyScene:
    """Render one complete buoyancy-density scene and projected annotation."""

    fallback: BuoyancyRenderDefaults = DEFAULT_RENDERING
    canvas_width = int(params.get("canvas_width", render_defaults.get("canvas_width", fallback.canvas_width)))
    canvas_height = int(params.get("canvas_height", render_defaults.get("canvas_height", fallback.canvas_height)))
    background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        require_grid=True,
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font",
        params=params,
    )
    resolved_defaults = _resolve_render_defaults(
        params=params,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    resolved_defaults, layout_meta = _resolve_layout_placement(
        render_defaults=resolved_defaults,
        jitter_defaults=render_defaults,
        params=params,
        instance_seed=int(instance_seed),
        canvas_width=int(canvas_width),
        canvas_height=int(canvas_height),
        namespace=str(namespace),
    )
    rendered = _draw_scene_body(
        background=background,
        scenario=scenario,
        render_defaults=resolved_defaults,
        font_family=str(font_family),
        style=diagram_style,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered.image,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    render_map = dict(rendered.render_map)
    render_map.update(
        {
            "technical_diagram_style": dict(diagram_style_meta),
            "background_style": dict(background_meta),
            "layout_placement": dict(layout_meta),
            "post_image_noise": dict(post_noise_meta),
        }
    )
    return RenderedBuoyancyScene(
        image=image,
        annotation_bbox_map={str(key): list(value) for key, value in rendered.annotation_bbox_map.items()},
        scene_entities=list(rendered.scene_entities),
        render_map=render_map,
        render_spec={
            "canvas_width": int(image.size[0]),
            "canvas_height": int(image.size[1]),
            "technical_diagram_style": dict(diagram_style_meta),
            "background_style": dict(background_meta),
            "layout_placement": dict(layout_meta),
            "post_image_noise": dict(post_noise_meta),
        },
        font_family=str(font_family),
    )


__all__ = ["render_buoyancy_density"]
