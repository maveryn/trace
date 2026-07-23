"""Rendering primitives for graduated-cylinder diagrams."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.drawing import draw_centered_text
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.text_legibility import ReadableTextStyle, draw_readable_text, resolve_readable_text_style
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .annotations import bbox
from .sampling import choose_liquid_rgb
from .state import (
    SCENE_ID,
    SCENE_NAMESPACE,
    CylinderGeometry,
    CylinderScale,
    GraduatedCylinderRenderDefaults,
    RenderedCylinderScene,
)


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)
DEFAULTS = GraduatedCylinderRenderDefaults()


def volume_to_y(geometry: CylinderGeometry, scale: CylinderScale, volume_ml: int) -> float:
    """Project one liquid volume onto the drawn cylinder scale."""

    usable_top = float(geometry.top + 28)
    usable_bottom = float(geometry.bottom - 22)
    fraction = max(0.0, min(1.0, float(volume_ml) / float(scale.capacity_ml)))
    return float(usable_bottom - fraction * (usable_bottom - usable_top))


def draw_label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    font: Any,
    fill: tuple[int, int, int],
    label_style: ReadableTextStyle,
) -> list[float]:
    """Draw one required readable label and return its bbox."""

    del fill
    text_bbox = draw.textbbox(
        (float(xy[0]), float(xy[1])),
        str(text),
        font=font,
        stroke_width=0,
    )
    backplate = bbox(
        (
            float(text_bbox[0]) - 10.0,
            float(text_bbox[1]) - 7.0,
            float(text_bbox[2]) + 10.0,
            float(text_bbox[3]) + 7.0,
        )
    )
    draw.rounded_rectangle(
        tuple(backplate),
        radius=4,
        fill=(250, 252, 255),
        outline=(198, 207, 219),
        width=1,
    )
    draw_readable_text(
        draw,
        xy=(float(xy[0]), float(xy[1])),
        text=str(text),
        font=font,
        style=label_style,
        stroke_width=0,
    )
    return bbox(tuple(float(value) for value in text_bbox))


def draw_cylinder(
    draw: ImageDraw.ImageDraw,
    *,
    geometry: CylinderGeometry,
    scale: CylinderScale,
    volume_ml: int,
    title: str,
    font_family: str,
    style: Any,
    liquid_rgb: tuple[int, int, int],
    label_style: ReadableTextStyle,
) -> dict[str, Any]:
    """Draw one graduated cylinder and return projected scale witnesses."""

    small_font = load_font(18, bold=True, font_family=font_family)
    title_font = load_font(24, bold=True, font_family=font_family)
    outline = tuple(int(value) for value in style.stroke_rgb)
    guide = tuple(int(value) for value in style.guide_rgb)
    text_rgb = tuple(int(value) for value in style.label_rgb)
    glass_fill = (238, 247, 252)

    left = float(geometry.left)
    top = float(geometry.top)
    width = float(geometry.width)
    bottom = float(geometry.bottom)
    right = float(left + width)
    body = (left, top + 18.0, right, bottom)
    draw.rounded_rectangle(body, radius=16, fill=glass_fill, outline=outline, width=4)
    draw.ellipse(
        (left, top, right, top + 38.0),
        fill=(248, 252, 254),
        outline=outline,
        width=4,
    )
    draw.arc((left, bottom - 20.0, right, bottom + 20.0), 0, 180, fill=outline, width=3)

    level_y = volume_to_y(geometry, scale, volume_ml)
    fill_box = (left + 8.0, level_y, right - 8.0, bottom - 10.0)
    draw.rectangle(fill_box, fill=tuple(int(value) for value in liquid_rgb))
    dark_liquid = tuple(max(0, int(value) - 40) for value in liquid_rgb)
    darker_liquid = tuple(max(0, int(value) - 45) for value in liquid_rgb)
    draw.arc((left + 8.0, level_y - 10.0, right - 8.0, level_y + 10.0), 0, 180, fill=dark_liquid, width=4)
    draw.line((left + 10.0, level_y, right - 10.0, level_y), fill=darker_liquid, width=2)

    scale_x = left - 20.0 if geometry.scale_left else right + 20.0
    tick_dir = 1 if geometry.scale_left else -1
    label_x = scale_x - 54.0 if geometry.scale_left else scale_x + 10.0
    scale_bboxes: list[list[float]] = []
    for tick_value in range(0, int(scale.capacity_ml) + 1, int(scale.minor_tick_ml)):
        y_pos = volume_to_y(geometry, scale, tick_value)
        is_major = tick_value % int(scale.major_tick_ml) == 0
        tick_len = 22.0 if is_major else 12.0
        draw.line(
            (scale_x, y_pos, scale_x + tick_dir * tick_len, y_pos),
            fill=outline if is_major else guide,
            width=3 if is_major else 2,
        )
        if is_major:
            scale_bboxes.append(
                draw_label(draw, (label_x, y_pos - 11.0), str(tick_value), small_font, text_rgb, label_style)
            )
    scale_bboxes.append(draw_label(draw, (label_x, bottom + 18.0), "mL", small_font, text_rgb, label_style))
    title_bbox = draw_centered_text(
        draw,
        text=str(title),
        center=(left + width * 0.5, top - 24.0),
        font=title_font,
        fill=text_rgb,
        stroke_fill=resolve_text_stroke_fill(text_rgb),
        stroke_width=1,
    )

    scale_region = bbox(
        (
            min([scale_x] + [item[0] for item in scale_bboxes]) - 8.0,
            min(item[1] for item in scale_bboxes) - 8.0,
            max([scale_x] + [item[2] for item in scale_bboxes]) + 8.0,
            max(item[3] for item in scale_bboxes) + 8.0,
        )
    )
    meniscus = bbox((left + 8.0, level_y - 12.0, right - 8.0, level_y + 12.0))
    readout_bbox = bbox(
        (
            min(left - 6.0, scale_region[0]),
            min(top - 4.0, scale_region[1]),
            max(right + 6.0, scale_region[2]),
            max(bottom + 10.0, scale_region[3]),
        )
    )
    return {
        "meniscus": meniscus,
        "scale_region": scale_region,
        "readout_bbox": readout_bbox,
        "title_bbox": title_bbox,
        "level_y": round(float(level_y), 3),
        "volume_ml": int(volume_ml),
    }


def prepare_canvas(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> tuple[Any, ImageDraw.ImageDraw, dict[str, Any], Any, dict[str, Any]]:
    """Prepare one physics-styled canvas and drawing context."""

    canvas_width = int(rendering_defaults.get("canvas_width", DEFAULTS.canvas_width))
    canvas_height = int(rendering_defaults.get("canvas_height", DEFAULTS.canvas_height))
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
    draw = ImageDraw.Draw(background)
    draw.rounded_rectangle(
        (54, 52, canvas_width - 54, canvas_height - 58),
        radius=18,
        fill=tuple(diagram_style.panel_fill_rgb),
        outline=tuple(diagram_style.panel_border_rgb),
        width=3,
    )
    return background, draw, background_meta, diagram_style, diagram_style_meta


def render_single_cylinder_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scale: CylinderScale,
    volume_ml: int,
    rendering_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> RenderedCylinderScene:
    """Render a one-cylinder readout scene and preserve its witness boxes."""

    background, draw, background_meta, diagram_style, diagram_style_meta = prepare_canvas(
        instance_seed=int(instance_seed),
        params=params,
        rendering_defaults=rendering_defaults,
    )
    rng = spawn_rng(int(instance_seed), f"{namespace}.scene")
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font",
        params=params,
    )
    liquid_rgb = choose_liquid_rgb(int(instance_seed), namespace=str(namespace))
    label_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.scale_readout",
        role="readout",
        surface_rgbs=((250, 252, 255), (198, 207, 219)),
        preferred_rgbs=((10, 14, 22),),
    )
    canvas_width, _ = background.size
    geometry = CylinderGeometry(
        left=float(canvas_width * 0.44),
        top=150.0 + rng.randint(-10, 10),
        width=170.0,
        height=420.0,
        bottom=570.0 + rng.randint(-8, 8),
        scale_left=bool(rng.randrange(2)),
    )
    rendered = draw_cylinder(
        draw,
        geometry=geometry,
        scale=scale,
        volume_ml=int(volume_ml),
        title="Volume",
        font_family=str(font_family),
        style=diagram_style,
        liquid_rgb=liquid_rgb,
        label_style=label_style,
    )
    annotation_map = {"readout": list(rendered["readout_bbox"])}
    image, post_noise_meta = apply_post_image_noise(
        background,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedCylinderScene(
        image=image,
        annotation_bbox_map=annotation_map,
        scene_entities=[],
        render_map={"cylinders": {"single": rendered}, "scale": scale.__dict__},
        background_meta=dict(background_meta),
        diagram_style_meta=dict(diagram_style_meta),
        post_noise_meta=dict(post_noise_meta),
        font_family=str(font_family),
    )


def render_before_after_cylinder_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scale: CylinderScale,
    before_ml: int,
    after_ml: int,
    displacement_ml: int,
    rendering_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> RenderedCylinderScene:
    """Render a before/after displacement scene and preserve witness boxes."""

    background, draw, background_meta, diagram_style, diagram_style_meta = prepare_canvas(
        instance_seed=int(instance_seed),
        params=params,
        rendering_defaults=rendering_defaults,
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font",
        params=params,
    )
    liquid_rgb = choose_liquid_rgb(int(instance_seed), namespace=str(namespace))
    label_style = resolve_readable_text_style(
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.scale_readout",
        role="readout",
        surface_rgbs=((250, 252, 255), (198, 207, 219)),
        preferred_rgbs=((10, 14, 22),),
    )
    left_geometry = CylinderGeometry(210.0, 158.0, 155.0, 405.0, 565.0, True)
    right_geometry = CylinderGeometry(650.0, 158.0, 155.0, 405.0, 565.0, False)
    before_rendered = draw_cylinder(
        draw,
        geometry=left_geometry,
        scale=scale,
        volume_ml=int(before_ml),
        title="Before",
        font_family=str(font_family),
        style=diagram_style,
        liquid_rgb=liquid_rgb,
        label_style=label_style,
    )
    after_rendered = draw_cylinder(
        draw,
        geometry=right_geometry,
        scale=scale,
        volume_ml=int(after_ml),
        title="After",
        font_family=str(font_family),
        style=diagram_style,
        liquid_rgb=liquid_rgb,
        label_style=label_style,
    )
    obj_cx = right_geometry.left + right_geometry.width * 0.52
    obj_y = min(right_geometry.bottom - 56.0, float(after_rendered["level_y"]) + 46.0)
    draw.ellipse(
        (obj_cx - 28.0, obj_y - 22.0, obj_cx + 28.0, obj_y + 22.0),
        fill=(154, 114, 88),
        outline=tuple(diagram_style.stroke_rgb),
        width=3,
    )
    annotation_map = {
        "before_cylinder": list(before_rendered["readout_bbox"]),
        "after_cylinder": list(after_rendered["readout_bbox"]),
    }
    image, post_noise_meta = apply_post_image_noise(
        background,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedCylinderScene(
        image=image,
        annotation_bbox_map=annotation_map,
        scene_entities=[],
        render_map={
            "cylinders": {"before": before_rendered, "after": after_rendered},
            "scale": scale.__dict__,
            "before_volume_ml": int(before_ml),
            "after_volume_ml": int(after_ml),
            "displacement_ml": int(displacement_ml),
        },
        background_meta=dict(background_meta),
        diagram_style_meta=dict(diagram_style_meta),
        post_noise_meta=dict(post_noise_meta),
        font_family=str(font_family),
    )
