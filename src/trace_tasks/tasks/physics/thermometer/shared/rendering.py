"""Rendering primitives for thermometer diagrams."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.drawing import draw_centered_text
from trace_tasks.tasks.shared.font_assets import font_asset_version, get_font_family_record, sample_font_family
from trace_tasks.tasks.shared.text_legibility import draw_text_traced, text_legibility_metadata_for_surfaces
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .annotations import bbox, bbox_union
from .state import (
    LIQUID_COLORS,
    SCENE_ID,
    SCENE_NAMESPACE,
    RenderedThermometerScene,
    ThermometerDefaults,
    ThermometerGeometry,
    ThermometerProfile,
    ThermometerScenario,
)


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: Tuple[float, float],
    text: str,
    font: Any,
    fill: Tuple[int, int, int],
    *,
    anchor: str | None = None,
    surface_rgbs: Sequence[Sequence[int]] | None = None,
) -> List[float]:
    """Draw one required readout label and return its pixel bbox."""

    kwargs: Dict[str, Any] = {}
    if anchor is not None:
        kwargs["anchor"] = str(anchor)
    record = draw_text_traced(
        draw,
        (float(xy[0]), float(xy[1])),
        str(text),
        font=font,
        fill=fill,
        stroke_width=0,
        stroke_fill=resolve_text_stroke_fill(fill),
        role="readout",
        required=True,
        extra_metadata=(
            text_legibility_metadata_for_surfaces(fill_rgb=fill, surface_rgbs=surface_rgbs)
            if surface_rgbs is not None
            else None
        ),
        **kwargs,
    )
    raw_bbox = tuple(float(value) for value in record["bbox_px"])
    return bbox((raw_bbox[0], raw_bbox[1], raw_bbox[2], raw_bbox[3]))


def temperature_to_y(geometry: ThermometerGeometry, profile: ThermometerProfile, temperature: int) -> float:
    """Project a source temperature onto the visible thermometer scale."""

    frac = (float(temperature) - float(profile.scale_min)) / float(profile.scale_max - profile.scale_min)
    frac = max(0.0, min(1.0, frac))
    return float(geometry.scale_bottom - frac * (geometry.scale_bottom - geometry.scale_top))


def draw_thermometer(
    draw: ImageDraw.ImageDraw,
    *,
    geometry: ThermometerGeometry,
    scenario: ThermometerScenario,
    font_family: str,
    style: Any,
    liquid_rgb: Tuple[int, int, int],
    render_defaults: Mapping[str, Any],
    defaults: ThermometerDefaults,
) -> Dict[str, Any]:
    """Draw one vertical thermometer and return projected witness boxes."""

    profile = scenario.profile
    tick_font = load_font(int(render_defaults.get("tick_font_size_px", defaults.tick_font_size_px)), bold=True, font_family=font_family)
    unit_font = load_font(int(render_defaults.get("unit_font_size_px", defaults.unit_font_size_px)), bold=True, font_family=font_family)
    title_font = load_font(int(render_defaults.get("title_font_size_px", defaults.title_font_size_px)), bold=True, font_family=font_family)
    outline = tuple(int(value) for value in style.stroke_rgb)
    guide = tuple(int(value) for value in style.guide_rgb)
    text_rgb = tuple(int(value) for value in style.label_rgb)
    panel_fill_rgb = tuple(int(value) for value in style.panel_fill_rgb)
    glass_fill = (239, 248, 252)
    tube_left = float(geometry.center_x - geometry.tube_width * 0.5)
    tube_right = float(geometry.center_x + geometry.tube_width * 0.5)
    tube_top = float(geometry.scale_top - 8)
    tube_bottom = float(geometry.scale_bottom + 20)
    bulb_cy = float(geometry.scale_bottom + geometry.bulb_radius * 0.66)
    bulb_bbox = (
        float(geometry.center_x - geometry.bulb_radius),
        float(bulb_cy - geometry.bulb_radius),
        float(geometry.center_x + geometry.bulb_radius),
        float(bulb_cy + geometry.bulb_radius),
    )

    draw.rounded_rectangle((tube_left, tube_top, tube_right, tube_bottom), radius=22, fill=glass_fill, outline=outline, width=4)
    draw.ellipse(bulb_bbox, fill=glass_fill, outline=outline, width=4)

    level_y = temperature_to_y(geometry, profile, int(scenario.source_temperature))
    liquid_left = float(tube_left + 9)
    liquid_right = float(tube_right - 9)
    liquid_level_segment = [
        [round(float(liquid_left), 3), round(float(level_y), 3)],
        [round(float(liquid_right), 3), round(float(level_y), 3)],
    ]
    draw.rounded_rectangle(
        (liquid_left, float(level_y), liquid_right, float(tube_bottom + 3)),
        radius=10,
        fill=tuple(int(value) for value in liquid_rgb),
    )
    draw.ellipse(
        (
            float(geometry.center_x - geometry.bulb_radius + 12),
            float(bulb_cy - geometry.bulb_radius + 12),
            float(geometry.center_x + geometry.bulb_radius - 12),
            float(bulb_cy + geometry.bulb_radius - 12),
        ),
        fill=tuple(int(value) for value in liquid_rgb),
        outline=tuple(max(0, int(value) - 45) for value in liquid_rgb),
        width=3,
    )
    draw.line((liquid_left, level_y, liquid_right, level_y), fill=tuple(max(0, int(value) - 55) for value in liquid_rgb), width=4)

    scale_x = float(geometry.center_x - 96 if geometry.scale_left else geometry.center_x + 96)
    tick_dir = 1 if geometry.scale_left else -1
    label_anchor = "rm" if geometry.scale_left else "lm"
    label_x = float(scale_x - 12 if geometry.scale_left else scale_x + 12)
    tick_bboxes: List[List[float]] = []
    label_bboxes: List[List[float]] = []
    for tick_value in range(int(profile.scale_min), int(profile.scale_max) + 1, int(profile.minor_step)):
        y = temperature_to_y(geometry, profile, int(tick_value))
        is_major = (tick_value - int(profile.scale_min)) % int(profile.major_step) == 0
        tick_len = 28 if is_major else 14
        x1 = float(scale_x)
        x2 = float(scale_x + tick_dir * tick_len)
        draw.line((x1, y, x2, y), fill=outline if is_major else guide, width=3 if is_major else 1)
        tick_bboxes.append(bbox((min(x1, x2), y - 2, max(x1, x2), y + 2)))
        if is_major:
            label_bboxes.append(
                draw_label(
                    draw,
                    (label_x, y),
                    str(tick_value),
                    tick_font,
                    text_rgb,
                    anchor=label_anchor,
                    surface_rgbs=(panel_fill_rgb,),
                )
            )

    unit_y = float(geometry.scale_top - 42)
    unit_bbox = draw_label(
        draw,
        (scale_x, unit_y),
        str(profile.source_unit),
        unit_font,
        text_rgb,
        anchor="mm",
        surface_rgbs=(panel_fill_rgb,),
    )
    draw_centered_text(
        draw,
        text="Thermometer",
        center=(float(geometry.center_x), float(geometry.scale_top - 78)),
        font=title_font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=1,
    )
    scale_region = bbox_union([*tick_bboxes, *label_bboxes, unit_bbox])
    liquid_level = bbox((liquid_left - 5, float(level_y - 10), liquid_right + 5, float(level_y + 10)))
    return {
        "liquid_level": liquid_level,
        "liquid_level_segment": liquid_level_segment,
        "scale_region": scale_region,
        "source_unit_label": unit_bbox,
        "thermometer_body": bbox((min(tube_left, bulb_bbox[0]), tube_top, max(tube_right, bulb_bbox[2]), bulb_bbox[3])),
        "level_y_px": round(float(level_y), 3),
    }


def render_thermometer(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: ThermometerScenario,
    render_defaults: Mapping[str, Any],
    defaults: ThermometerDefaults,
) -> RenderedThermometerScene:
    """Render a thermometer scenario and projected annotation witnesses."""

    canvas_width = int(render_defaults.get("canvas_width", defaults.canvas_width))
    canvas_height = int(render_defaults.get("canvas_height", defaults.canvas_height))
    background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        require_grid=True,
    )
    draw = ImageDraw.Draw(background)
    rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.layout")
    panel_left = int(render_defaults.get("panel_left_px", defaults.panel_left_px))
    panel_top = int(render_defaults.get("panel_top_px", defaults.panel_top_px))
    panel_right = int(canvas_width - int(render_defaults.get("panel_right_margin_px", defaults.panel_right_margin_px)))
    panel_bottom = int(canvas_height - int(render_defaults.get("panel_bottom_margin_px", defaults.panel_bottom_margin_px)))
    draw.rounded_rectangle(
        (panel_left, panel_top, panel_right, panel_bottom),
        radius=18,
        fill=tuple(int(value) for value in diagram_style.panel_fill_rgb),
        outline=tuple(int(value) for value in diagram_style.panel_border_rgb),
        width=3,
    )

    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.font",
        params=params,
    )
    font_record = get_font_family_record(str(font_family))
    liquid_rgb = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.liquid").choice(LIQUID_COLORS)
    geometry = ThermometerGeometry(
        center_x=float(int(render_defaults.get("thermometer_center_x_px", defaults.thermometer_center_x_px)) + rng.randint(-16, 16)),
        scale_top=float(int(render_defaults.get("thermometer_scale_top_px", defaults.thermometer_scale_top_px)) + rng.randint(-8, 8)),
        scale_bottom=float(int(render_defaults.get("thermometer_scale_bottom_px", defaults.thermometer_scale_bottom_px)) + rng.randint(-6, 8)),
        tube_width=float(int(render_defaults.get("tube_width_px", defaults.tube_width_px))),
        bulb_radius=float(int(render_defaults.get("bulb_radius_px", defaults.bulb_radius_px))),
        scale_left=bool(rng.randrange(2)),
    )
    rendered = draw_thermometer(
        draw,
        geometry=geometry,
        scenario=scenario,
        font_family=str(font_family),
        style=diagram_style,
        liquid_rgb=tuple(int(value) for value in liquid_rgb),
        render_defaults=render_defaults,
        defaults=defaults,
    )
    image, post_noise_meta = apply_post_image_noise(
        background,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_map = {
        "liquid_level": list(rendered["liquid_level"]),
        "scale_region": list(rendered["scale_region"]),
        "source_unit_label": list(rendered["source_unit_label"]),
    }
    annotation_segment = [list(point) for point in rendered["liquid_level_segment"]]
    scene_entities = [
        {
            "id": "liquid_level",
            "bbox_px": list(annotation_map["liquid_level"]),
            "segment_px": [list(point) for point in annotation_segment],
            "source_temperature": int(scenario.source_temperature),
            "source_unit": str(scenario.profile.source_unit),
        },
        {
            "id": "scale_region",
            "bbox_px": list(annotation_map["scale_region"]),
            "scale_min": int(scenario.profile.scale_min),
            "scale_max": int(scenario.profile.scale_max),
            "source_unit": str(scenario.profile.source_unit),
        },
        {"id": "source_unit_label", "bbox_px": list(annotation_map["source_unit_label"]), "unit": str(scenario.profile.source_unit)},
    ]
    render_map = {
        "thermometer": dict(rendered),
        "source_temperature": int(scenario.source_temperature),
        "source_unit": str(scenario.profile.source_unit),
        "target_temperature": int(scenario.target_temperature),
        "target_unit": str(scenario.profile.target_unit),
        "annotation_source": "liquid_level_segment_px",
        "annotation_segment_px": [list(point) for point in annotation_segment],
        "liquid_level_segment_px": [list(point) for point in annotation_segment],
        "context_bbox_map_px": {str(key): list(value) for key, value in annotation_map.items()},
        "scale_profile": {
            "profile_id": str(scenario.profile.profile_id),
            "scale_min": int(scenario.profile.scale_min),
            "scale_max": int(scenario.profile.scale_max),
            "major_step": int(scenario.profile.major_step),
            "minor_step": int(scenario.profile.minor_step),
        },
        "font": {
            "font_family": str(font_family),
            "font_asset_version": font_asset_version(),
            "font_asset": font_record.to_trace(),
        },
        "technical_diagram_style": dict(diagram_style_meta),
        "background_style": dict(background_meta),
        "post_image_noise": dict(post_noise_meta),
    }
    return RenderedThermometerScene(
        image=image,
        annotation_segment=[list(point) for point in annotation_segment],
        scene_entities=scene_entities,
        render_map=render_map,
        font_family=str(font_family),
    )


__all__ = ["draw_label", "draw_thermometer", "render_thermometer", "temperature_to_y"]
