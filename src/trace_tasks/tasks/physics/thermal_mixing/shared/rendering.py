"""Rendering primitives for thermal-mixing diagrams."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.text_legibility import draw_text_traced
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .annotations import bbox
from .state import (
    LIQUID_COLORS,
    SCENE_ID,
    SCENE_NAMESPACE,
    RenderedThermalMixingScene,
    ThermalMixingDefaults,
    ThermalMixingScenario,
)


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def text_bbox(
    draw: ImageDraw.ImageDraw,
    text: str,
    center: Tuple[float, float],
    font: Any,
    padding: float = 7.0,
) -> List[float]:
    """Return a padded centered text bbox."""

    raw = draw.textbbox((0, 0), str(text), font=font, stroke_width=0)
    width = float(raw[2] - raw[0])
    height = float(raw[3] - raw[1])
    return bbox(
        (
            float(center[0]) - width / 2.0 - float(padding),
            float(center[1]) - height / 2.0 - float(padding),
            float(center[0]) + width / 2.0 + float(padding),
            float(center[1]) + height / 2.0 + float(padding),
        )
    )


def draw_text_centered_traced(
    draw: ImageDraw.ImageDraw,
    *,
    text: str,
    center: Tuple[float, float],
    font: Any,
    fill: Tuple[int, int, int],
    stroke_width: int = 0,
    required: bool = True,
) -> List[float]:
    """Draw centered traced text and return its bbox."""

    output_bbox = text_bbox(
        draw,
        str(text),
        center,
        font,
        padding=max(5.0, float(stroke_width) + 4.0),
    )
    draw_text_traced(
        draw,
        (float(center[0]), float(center[1])),
        str(text),
        font=font,
        fill=fill,
        anchor="mm",
        stroke_width=int(stroke_width),
        stroke_fill=resolve_text_stroke_fill(fill),
        role="readout" if bool(required) else "decorative_label",
        required=bool(required),
    )
    return output_bbox


def draw_cup(
    *,
    draw: ImageDraw.ImageDraw,
    center_x: float,
    top_y: float,
    cup_width: float,
    cup_height: float,
    label: str,
    temperature_c: int,
    liquid_rgb: Tuple[int, int, int],
    style: Any,
    font_family: str,
    render_defaults: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[List[float], Dict[str, Any]]:
    """Draw one initial liquid cup and return its temperature-readout annotation bbox/entity."""

    label_font = load_font(int(render_defaults["label_font_size_px"]), bold=True, font_family=font_family)
    temp_font = load_font(int(render_defaults["temp_font_size_px"]), bold=True, font_family=font_family)
    note_font = load_font(int(render_defaults["note_font_size_px"]), bold=True, font_family=font_family)
    stroke_rgb = tuple(int(value) for value in style.stroke_rgb)
    label_rgb = tuple(int(value) for value in style.label_rgb)
    readout_rgb = (10, 14, 22)

    x0 = float(center_x - cup_width / 2.0)
    x1 = float(center_x + cup_width / 2.0)
    y0 = float(top_y)
    y1 = float(top_y + cup_height)
    rim_y = y0 + 8.0
    cup_poly = [
        (x0 + 10.0, y0),
        (x1 - 10.0, y0),
        (x1 - 24.0, y1),
        (x0 + 24.0, y1),
    ]
    draw.rounded_rectangle(
        [x0 + 6.0, y0 - 14.0, x1 - 6.0, y0 + 12.0],
        radius=12,
        fill=tuple(style.panel_fill_rgb),
        outline=stroke_rgb,
        width=3,
    )
    draw.polygon(cup_poly, fill=tuple(style.panel_alt_fill_rgb), outline=stroke_rgb)
    draw.line(
        [(x0 + 10.0, y0), (x0 + 24.0, y1), (x1 - 24.0, y1), (x1 - 10.0, y0)],
        fill=stroke_rgb,
        width=3,
    )
    liquid_top = y0 + cup_height * 0.46
    liquid_poly = [
        (x0 + 17.0, liquid_top),
        (x1 - 17.0, liquid_top),
        (x1 - 24.0, y1 - 7.0),
        (x0 + 24.0, y1 - 7.0),
    ]
    draw.polygon(
        liquid_poly,
        fill=tuple(int(value) for value in liquid_rgb),
        outline=tuple(int(max(0, value - 45)) for value in liquid_rgb),
    )
    draw.line(
        [(x0 + 18.0, liquid_top), (x1 - 18.0, liquid_top)],
        fill=tuple(min(255, int(value) + 35) for value in liquid_rgb),
        width=3,
    )
    draw.arc([x0 + 7.0, y0 - 15.0, x1 - 7.0, y0 + 13.0], start=0, end=180, fill=stroke_rgb, width=3)
    draw.arc(
        [x0 + 7.0, rim_y - 14.0, x1 - 7.0, rim_y + 14.0],
        start=0,
        end=180,
        fill=tuple(int(value) for value in style.guide_rgb),
        width=2,
    )

    label_bbox = text_bbox(draw, str(label), (center_x, y0 - 36.0), label_font, padding=8.0)
    draw.rounded_rectangle(
        label_bbox,
        radius=8,
        fill=tuple(style.label_fill_rgb),
        outline=tuple(style.label_border_rgb),
        width=2,
    )
    draw_text_centered_traced(
        draw,
        text=str(label),
        center=(center_x, y0 - 36.0),
        font=label_font,
        fill=label_rgb,
        stroke_width=int(render_defaults["label_stroke_width_px"]),
        required=False,
    )
    temp_text = f"{int(temperature_c)} C"
    temp_bbox = text_bbox(draw, temp_text, (center_x, y0 + cup_height * 0.35), temp_font, padding=13.0)
    draw.rounded_rectangle(
        temp_bbox,
        radius=10,
        fill=(250, 252, 255),
        outline=(198, 207, 219),
        width=2,
    )
    draw_text_centered_traced(
        draw,
        text=temp_text,
        center=(center_x, y0 + cup_height * 0.35),
        font=temp_font,
        fill=readout_rgb,
        stroke_width=0,
        required=True,
    )
    draw_text_centered_traced(
        draw,
        text="equal amount",
        center=(center_x, y1 + 24.0),
        font=note_font,
        fill=label_rgb,
        stroke_width=0,
        required=False,
    )
    cup_bbox = bbox((x0 + 5.0, y0 - 14.0, x1 - 5.0, y1 + 3.0))
    annotation_bbox = bbox(temp_bbox)
    entity = {
        "entity_id": f"initial_cup_{label.lower()}",
        "entity_type": "initial_liquid_cup",
        "bbox_px": list(cup_bbox),
        "meta": {
            "label": str(label),
            "temperature_c": int(temperature_c),
            "temperature_label_bbox_px": list(annotation_bbox),
            "amount": "equal",
            "liquid": "same",
        },
    }
    return list(annotation_bbox), entity


def draw_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    start: Tuple[float, float],
    end: Tuple[float, float],
    fill: Tuple[int, int, int],
    width: int = 5,
) -> None:
    """Draw a directed pour arrow."""

    draw.line([start, end], fill=fill, width=int(width))
    dx = float(end[0] - start[0])
    dy = float(end[1] - start[1])
    length = max(1.0, (dx * dx + dy * dy) ** 0.5)
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    head = 18.0
    wing = 10.0
    points = [
        (float(end[0]), float(end[1])),
        (float(end[0]) - ux * head + px * wing, float(end[1]) - uy * head + py * wing),
        (float(end[0]) - ux * head - px * wing, float(end[1]) - uy * head - py * wing),
    ]
    draw.polygon(points, fill=fill)


def render_thermal_mixing_body(
    *,
    image: Image.Image,
    scenario: ThermalMixingScenario,
    render_defaults: Mapping[str, Any],
    font_family: str,
    style: Any,
    instance_seed: int,
) -> RenderedThermalMixingScene:
    """Render thermal-mixing apparatus onto an existing image."""

    draw = ImageDraw.Draw(image)
    width, height = image.size
    panel_left = float(render_defaults["panel_left_px"])
    panel_top = float(render_defaults["panel_top_px"])
    panel_right = float(width - int(render_defaults["panel_right_margin_px"]))
    panel_bottom = float(height - int(render_defaults["panel_bottom_margin_px"]))
    panel_bbox = [panel_left, panel_top, panel_right, panel_bottom]
    draw.rounded_rectangle(
        panel_bbox,
        radius=22,
        fill=tuple(style.panel_fill_rgb),
        outline=tuple(style.panel_border_rgb),
        width=3,
    )

    title_font = load_font(int(render_defaults["title_font_size_px"]), bold=True, font_family=font_family)
    label_font = load_font(int(render_defaults["label_font_size_px"]), bold=True, font_family=font_family)
    label_rgb = tuple(int(value) for value in style.label_rgb)
    draw_text_centered_traced(
        draw,
        text="thermal mixing in an insulated container",
        center=((panel_left + panel_right) / 2.0, panel_top + 32.0),
        font=title_font,
        fill=label_rgb,
        stroke_width=1,
        required=False,
    )

    cup_count = int(scenario.cup_count)
    cup_width = float(render_defaults["cup_width_px"])
    cup_height = float(render_defaults["cup_height_px"])
    cup_gap = float(render_defaults["cup_gap_px"])
    total_width = cup_count * cup_width + max(0, cup_count - 1) * cup_gap
    start_x = float((panel_left + panel_right) / 2.0 - total_width / 2.0 + cup_width / 2.0)
    cup_top = float(render_defaults["cup_top_px"])
    liquid_rgb = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.liquid").choice(LIQUID_COLORS)
    annotation_bboxes: List[List[float]] = []
    entities: List[Dict[str, Any]] = []
    cup_centers: List[Tuple[float, float]] = []
    for index, temperature in enumerate(scenario.initial_temperatures_c):
        center_x = start_x + index * (cup_width + cup_gap)
        annotation_bbox, entity = draw_cup(
            draw=draw,
            center_x=float(center_x),
            top_y=cup_top,
            cup_width=cup_width,
            cup_height=cup_height,
            label=chr(ord("A") + index),
            temperature_c=int(temperature),
            liquid_rgb=tuple(int(value) for value in liquid_rgb),
            style=style,
            font_family=str(font_family),
            render_defaults=render_defaults,
            instance_seed=int(instance_seed),
        )
        annotation_bboxes.append(list(annotation_bbox))
        entities.append(dict(entity))
        cup_centers.append((float(center_x), cup_top + cup_height + 48.0))

    mixer_width = float(render_defaults["mixer_width_px"])
    mixer_height = float(render_defaults["mixer_height_px"])
    mixer_top = float(render_defaults["mixer_top_px"])
    mixer_left = float((panel_left + panel_right) / 2.0 - mixer_width / 2.0)
    mixer_right = float(mixer_left + mixer_width)
    mixer_bottom = float(mixer_top + mixer_height)
    mixer_bbox = [mixer_left, mixer_top, mixer_right, mixer_bottom]
    draw.rounded_rectangle(
        mixer_bbox,
        radius=18,
        fill=tuple(style.panel_alt_fill_rgb),
        outline=tuple(style.stroke_rgb),
        width=4,
    )
    insulation_band = [mixer_left + 16.0, mixer_top + 16.0, mixer_right - 16.0, mixer_bottom - 16.0]
    draw.rounded_rectangle(insulation_band, radius=12, outline=tuple(style.guide_rgb), width=3)
    liquid_top = mixer_top + mixer_height * 0.54
    draw.rounded_rectangle(
        [mixer_left + 26.0, liquid_top, mixer_right - 26.0, mixer_bottom - 22.0],
        radius=14,
        fill=tuple(int(value) for value in liquid_rgb),
        outline=tuple(int(max(0, value - 45)) for value in liquid_rgb),
        width=2,
    )
    draw_text_centered_traced(
        draw,
        text="insulated mixer",
        center=((mixer_left + mixer_right) / 2.0, mixer_top + 34.0),
        font=label_font,
        fill=label_rgb,
        stroke_width=1,
        required=False,
    )
    unknown_bbox = text_bbox(
        draw,
        "? C",
        ((mixer_left + mixer_right) / 2.0, mixer_top + 84.0),
        label_font,
        padding=10.0,
    )
    draw.rounded_rectangle(
        unknown_bbox,
        radius=10,
        fill=tuple(style.label_fill_rgb),
        outline=tuple(style.label_border_rgb),
        width=2,
    )
    draw_text_centered_traced(
        draw,
        text="? C",
        center=((mixer_left + mixer_right) / 2.0, mixer_top + 84.0),
        font=label_font,
        fill=label_rgb,
        stroke_width=1,
        required=False,
    )
    arrow_rgb = tuple(int(value) for value in style.accent_rgb)
    for center in cup_centers:
        draw_arrow(
            draw,
            start=(float(center[0]), float(center[1])),
            end=((mixer_left + mixer_right) / 2.0, mixer_top - 10.0),
            fill=arrow_rgb,
            width=4,
        )

    entities.append(
        {
            "entity_id": "insulated_mixer",
            "entity_type": "final_mixing_container",
            "bbox_px": bbox(mixer_bbox),
            "meta": {
                "final_temperature_c": int(scenario.final_temperature_c),
                "visible_answer": False,
                "system": "closed_insulated",
            },
        }
    )
    render_map = {
        "cup_count": int(scenario.cup_count),
        "initial_temperatures_c": [int(value) for value in scenario.initial_temperatures_c],
        "final_temperature_c": int(scenario.final_temperature_c),
        "annotation_source": "temperature_label_bboxes_px",
        "annotation_bboxes_px": [list(item) for item in annotation_bboxes],
        "temperature_label_bboxes_px": [list(item) for item in annotation_bboxes],
        "mixer_bbox_px": bbox(mixer_bbox),
    }
    return RenderedThermalMixingScene(
        image=image,
        annotation_bboxes=[list(item) for item in annotation_bboxes],
        scene_entities=[dict(entity) for entity in entities],
        render_map=dict(render_map),
        font_family=str(font_family),
    )


def render_thermal_mixing(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: ThermalMixingScenario,
    rendering_defaults: Mapping[str, Any],
    render_defaults: Mapping[str, int],
    defaults: ThermalMixingDefaults,
) -> RenderedThermalMixingScene:
    """Render a complete thermal-mixing scene, including background and noise."""

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
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            require_grid=True,
        )
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.font",
        params=params,
    )
    rendered = render_thermal_mixing_body(
        image=background,
        scenario=scenario,
        render_defaults=render_defaults,
        font_family=str(font_family),
        style=diagram_style,
        instance_seed=int(instance_seed),
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
    return RenderedThermalMixingScene(
        image=image,
        annotation_bboxes=list(rendered.annotation_bboxes),
        scene_entities=list(rendered.scene_entities),
        render_map=render_map,
        font_family=str(font_family),
    )


__all__ = [
    "POST_IMAGE_NOISE_DEFAULTS",
    "draw_arrow",
    "draw_cup",
    "draw_text_centered_traced",
    "render_thermal_mixing",
    "render_thermal_mixing_body",
    "text_bbox",
]
