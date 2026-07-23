"""Rendering primitives for piston-cylinder diagrams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PIL import ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.bbox_projection import bbox_union_many
from trace_tasks.tasks.shared.drawing import draw_arrow, draw_centered_text
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .annotations import bbox
from .state import PistonRenderDefaults, PistonScenario, RenderedPistonScene, SCENE_ID, SCENE_NAMESPACE


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)
DEFAULTS = PistonRenderDefaults()


def union_bbox(*boxes: Sequence[float], padding: float = 0.0) -> list[float]:
    """Return a stable bbox around one or more boxes."""

    return bbox(bbox_union_many(*boxes, padding=float(padding)))


def draw_label_box(
    draw: ImageDraw.ImageDraw,
    *,
    lines: Sequence[str],
    center: tuple[float, float],
    font: Any,
    style: Any,
) -> list[float]:
    """Draw one readable rounded label box and return its bbox."""

    text_bboxes = [draw.textbbox((0, 0), str(line), font=font, stroke_width=1) for line in lines]
    text_width = max(float(box[2] - box[0]) for box in text_bboxes)
    line_height = max(float(box[3] - box[1]) for box in text_bboxes) + 8.0
    box_width = float(text_width + 38.0)
    box_height = float(line_height * len(lines) + 20.0)
    cx, cy = float(center[0]), float(center[1])
    box = bbox((cx - box_width / 2.0, cy - box_height / 2.0, cx + box_width / 2.0, cy + box_height / 2.0))
    draw.rounded_rectangle(
        tuple(box),
        radius=12,
        fill=tuple(int(value) for value in style.label_fill_rgb),
        outline=tuple(int(value) for value in style.label_border_rgb),
        width=3,
    )
    line_boxes: list[list[float]] = []
    first_y = float(cy - (line_height * (len(lines) - 1) / 2.0))
    label_rgb = tuple(int(value) for value in style.label_rgb)
    for index, line in enumerate(lines):
        text_box = draw_centered_text(
            draw,
            text=str(line),
            center=(float(cx), float(first_y + index * line_height)),
            font=font,
            fill=label_rgb,
            stroke_fill=resolve_text_stroke_fill(label_rgb),
            stroke_width=1,
        )
        line_boxes.append(list(text_box))
    return union_bbox(box, *line_boxes)


def draw_vertical_cylinder(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    volume_l: int,
    max_volume_l: int,
    style: Any,
    font: Any,
    label: str,
    gas_rgb: tuple[int, int, int],
    render_defaults: Mapping[str, Any],
) -> tuple[list[float], list[float]]:
    """Draw one vertical state while preserving volume-to-piston visual ordering."""

    cx, bottom = float(center[0]), float(center[1])
    width = float(render_defaults.get("cylinder_width_px", DEFAULTS.cylinder_width_px))
    height = float(render_defaults.get("cylinder_height_px", DEFAULTS.cylinder_height_px))
    left = float(cx - width / 2.0)
    right = float(cx + width / 2.0)
    top = float(bottom - height)
    stroke = tuple(int(value) for value in style.stroke_rgb)
    glass = (238, 248, 251)
    fill_fraction = 0.24 + 0.60 * (float(volume_l) / float(max_volume_l))
    gas_top = float(bottom - 18.0 - fill_fraction * (height - 54.0))
    inner = (left + 18.0, gas_top, right - 18.0, bottom - 18.0)
    draw.rounded_rectangle((left, top, right, bottom), radius=22, fill=glass, outline=stroke, width=5)
    draw.rectangle(inner, fill=gas_rgb)
    draw.arc((inner[0], gas_top - 14.0, inner[2], gas_top + 14.0), 0, 180, fill=tuple(max(0, v - 45) for v in gas_rgb), width=4)
    piston_y = float(gas_top - 18.0)
    piston = (
        left + 8.0,
        piston_y,
        right - 8.0,
        piston_y + float(render_defaults.get("piston_thickness_px", DEFAULTS.piston_thickness_px)),
    )
    draw.rounded_rectangle(piston, radius=9, fill=tuple(int(value) for value in style.muted_fill_rgb), outline=stroke, width=4)
    rod_x = float((piston[0] + piston[2]) / 2.0)
    draw.line((rod_x, top - 44.0, rod_x, piston[1]), fill=stroke, width=8)
    draw.rectangle(
        (rod_x - 32.0, top - 54.0, rod_x + 32.0, top - 38.0),
        fill=tuple(int(value) for value in style.muted_fill_rgb),
        outline=stroke,
        width=3,
    )
    draw.rounded_rectangle((left, top, right, bottom), radius=22, fill=None, outline=stroke, width=5)
    label_box = draw_centered_text(
        draw,
        text=str(label),
        center=(cx, float(top - 82.0)),
        font=font,
        fill=tuple(int(value) for value in style.label_rgb),
        stroke_fill=resolve_text_stroke_fill(tuple(int(value) for value in style.label_rgb)),
        stroke_width=1,
    )
    apparatus_bbox = union_bbox((left, top - 58.0, right, bottom), label_box, padding=6.0)
    return apparatus_bbox, bbox(piston)


def draw_horizontal_cylinder(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    volume_l: int,
    max_volume_l: int,
    style: Any,
    font: Any,
    label: str,
    gas_rgb: tuple[int, int, int],
    render_defaults: Mapping[str, Any],
) -> tuple[list[float], list[float]]:
    """Draw one horizontal state while preserving volume-to-piston visual ordering."""

    cy = float(center[1])
    left = float(center[0])
    width = float(render_defaults.get("horizontal_cylinder_width_px", DEFAULTS.horizontal_cylinder_width_px))
    height = float(render_defaults.get("horizontal_cylinder_height_px", DEFAULTS.horizontal_cylinder_height_px))
    right = float(left + width)
    top = float(cy - height / 2.0)
    bottom = float(cy + height / 2.0)
    stroke = tuple(int(value) for value in style.stroke_rgb)
    glass = (238, 248, 251)
    fill_fraction = 0.24 + 0.60 * (float(volume_l) / float(max_volume_l))
    gas_right = float(left + 18.0 + fill_fraction * (width - 54.0))
    inner = (left + 18.0, top + 18.0, gas_right, bottom - 18.0)
    draw.rounded_rectangle((left, top, right, bottom), radius=22, fill=glass, outline=stroke, width=5)
    draw.rectangle(inner, fill=gas_rgb)
    draw.arc((gas_right - 15.0, inner[1], gas_right + 15.0, inner[3]), 270, 90, fill=tuple(max(0, v - 45) for v in gas_rgb), width=4)
    piston_x = float(gas_right + 14.0)
    thickness = float(render_defaults.get("piston_thickness_px", DEFAULTS.piston_thickness_px))
    piston = (piston_x, top + 8.0, piston_x + thickness, bottom - 8.0)
    draw.rounded_rectangle(piston, radius=9, fill=tuple(int(value) for value in style.muted_fill_rgb), outline=stroke, width=4)
    rod_y = float((piston[1] + piston[3]) / 2.0)
    draw.line((piston[2], rod_y, right + 44.0, rod_y), fill=stroke, width=8)
    draw.rectangle(
        (right + 38.0, rod_y - 32.0, right + 54.0, rod_y + 32.0),
        fill=tuple(int(value) for value in style.muted_fill_rgb),
        outline=stroke,
        width=3,
    )
    draw.rounded_rectangle((left, top, right, bottom), radius=22, fill=None, outline=stroke, width=5)
    label_box = draw_centered_text(
        draw,
        text=str(label),
        center=(float((left + right) / 2.0), float(top - 36.0)),
        font=font,
        fill=tuple(int(value) for value in style.label_rgb),
        stroke_fill=resolve_text_stroke_fill(tuple(int(value) for value in style.label_rgb)),
        stroke_width=1,
    )
    apparatus_bbox = union_bbox((left, top, right + 54.0, bottom), label_box, padding=6.0)
    return apparatus_bbox, bbox(piston)


def render_piston_cylinder_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: PistonScenario,
    render_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> RenderedPistonScene:
    """Render the complete piston-cylinder diagram and projected metadata."""

    canvas_width = int(render_defaults.get("canvas_width", DEFAULTS.canvas_width))
    canvas_height = int(render_defaults.get("canvas_height", DEFAULTS.canvas_height))
    background, background_meta, diagram_style, diagram_style_meta = prepare_physics_diagram_style_and_background(
        instance_seed=int(instance_seed),
        params=params,
        scene_id=SCENE_ID,
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        require_grid=True,
    )
    draw = ImageDraw.Draw(background)
    rng = spawn_rng(int(instance_seed), f"{namespace}.render")
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font",
        params=params,
    )
    label_font = load_font(int(render_defaults.get("label_font_size_px", DEFAULTS.label_font_size_px)), bold=True, font_family=str(font_family))
    state_font = load_font(int(render_defaults.get("state_font_size_px", DEFAULTS.state_font_size_px)), bold=True, font_family=str(font_family))
    title_font = load_font(int(render_defaults.get("title_font_size_px", DEFAULTS.title_font_size_px)), bold=True, font_family=str(font_family))

    panel = (
        float(render_defaults.get("panel_left_px", DEFAULTS.panel_left_px)),
        float(render_defaults.get("panel_top_px", DEFAULTS.panel_top_px)),
        float(canvas_width - render_defaults.get("panel_right_margin_px", DEFAULTS.panel_right_margin_px)),
        float(canvas_height - render_defaults.get("panel_bottom_margin_px", DEFAULTS.panel_bottom_margin_px)),
    )
    draw.rounded_rectangle(
        panel,
        radius=18,
        fill=tuple(int(value) for value in diagram_style.panel_fill_rgb),
        outline=tuple(int(value) for value in diagram_style.panel_border_rgb),
        width=3,
    )
    title_rgb = tuple(int(value) for value in diagram_style.label_rgb)
    pressure_bbox = draw_label_box(
        draw,
        lines=(f"Constant pressure: P = {int(scenario.pressure_mpa)} MPa", "Use 1 MPa x L = 1 kJ"),
        center=(float(canvas_width * 0.5), float(panel[1] + 54.0)),
        font=title_font,
        style=diagram_style,
    )
    stroke_rgb = tuple(int(value) for value in diagram_style.stroke_rgb)
    gas_palette = (
        (92, 169, 219),
        (84, 184, 147),
        (139, 156, 224),
        (217, 153, 82),
        (177, 128, 202),
    )
    gas_rgb = spawn_rng(int(instance_seed), f"{namespace}.gas_color").choice(gas_palette)
    configured_volumes = [int(value) for value in params.get("volume_l_support", (1, 2, 3, 4, 5, 6, 7, 8, 9))]
    max_volume = max(
        configured_volumes
        + [
            int(scenario.initial_volume_l),
            int(scenario.final_volume_l),
        ]
    )
    jitter_x = float(rng.randint(-12, 12))
    jitter_y = float(rng.randint(-8, 8))

    if str(scenario.orientation) == "vertical_pair":
        left_center = (float(canvas_width * 0.32 + jitter_x), float(panel[3] - 104.0 + jitter_y))
        right_center = (float(canvas_width * 0.68 + jitter_x), float(panel[3] - 104.0 + jitter_y))
        initial_bbox, initial_piston_bbox = draw_vertical_cylinder(
            draw,
            center=left_center,
            volume_l=int(scenario.initial_volume_l),
            max_volume_l=int(max_volume),
            style=diagram_style,
            font=label_font,
            label="Initial",
            gas_rgb=gas_rgb,
            render_defaults=render_defaults,
        )
        final_bbox, final_piston_bbox = draw_vertical_cylinder(
            draw,
            center=right_center,
            volume_l=int(scenario.final_volume_l),
            max_volume_l=int(max_volume),
            style=diagram_style,
            font=label_font,
            label="Final",
            gas_rgb=gas_rgb,
            render_defaults=render_defaults,
        )
        arrow_start = (float(left_center[0] + 155.0), float(panel[1] + 210.0))
        arrow_end = (float(right_center[0] - 155.0), float(panel[1] + 210.0))
        initial_label_center = (float(left_center[0]), float(panel[3] - 36.0))
        final_label_center = (float(right_center[0]), float(panel[3] - 36.0))
    else:
        left_origin = (float(panel[0] + 96.0 + jitter_x), float(panel[1] + 330.0 + jitter_y))
        right_origin = (float(canvas_width * 0.57 + jitter_x), float(panel[1] + 330.0 + jitter_y))
        initial_bbox, initial_piston_bbox = draw_horizontal_cylinder(
            draw,
            center=left_origin,
            volume_l=int(scenario.initial_volume_l),
            max_volume_l=int(max_volume),
            style=diagram_style,
            font=label_font,
            label="Initial",
            gas_rgb=gas_rgb,
            render_defaults=render_defaults,
        )
        final_bbox, final_piston_bbox = draw_horizontal_cylinder(
            draw,
            center=right_origin,
            volume_l=int(scenario.final_volume_l),
            max_volume_l=int(max_volume),
            style=diagram_style,
            font=label_font,
            label="Final",
            gas_rgb=gas_rgb,
            render_defaults=render_defaults,
        )
        arrow_start = (float(panel[0] + 500.0), float(panel[1] + 330.0 + jitter_y))
        arrow_end = (float(panel[0] + 590.0), float(panel[1] + 330.0 + jitter_y))
        initial_label_center = (float(panel[0] + 266.0 + jitter_x), float(panel[3] - 76.0))
        final_label_center = (float(canvas_width * 0.72 + jitter_x), float(panel[3] - 76.0))

    draw_arrow(
        draw,
        start=arrow_start,
        end=arrow_end,
        fill=stroke_rgb,
        width=7,
        head_length_px=24,
        head_width_px=22,
    )
    arrow_label_bbox = draw_centered_text(
        draw,
        text="process",
        center=(float((arrow_start[0] + arrow_end[0]) * 0.5), float(arrow_start[1] - 28.0)),
        font=state_font,
        fill=title_rgb,
        stroke_fill=resolve_text_stroke_fill(title_rgb),
        stroke_width=1,
    )
    process_arrow_bbox = union_bbox(
        (
            min(arrow_start[0], arrow_end[0]) - 18.0,
            min(arrow_start[1], arrow_end[1]) - 18.0,
            max(arrow_start[0], arrow_end[0]) + 28.0,
            max(arrow_start[1], arrow_end[1]) + 18.0,
        ),
        arrow_label_bbox,
    )
    initial_state_label = draw_label_box(
        draw,
        lines=(f"P = {int(scenario.pressure_mpa)} MPa", f"Vi = {int(scenario.initial_volume_l)} L"),
        center=initial_label_center,
        font=state_font,
        style=diagram_style,
    )
    final_state_label = draw_label_box(
        draw,
        lines=(f"P = {int(scenario.pressure_mpa)} MPa", f"Vf = {int(scenario.final_volume_l)} L"),
        center=final_label_center,
        font=state_font,
        style=diagram_style,
    )
    piston_cylinder_bbox = union_bbox(initial_bbox, final_bbox, initial_piston_bbox, final_piston_bbox, padding=8.0)

    image, post_noise_meta = apply_post_image_noise(
        background,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    annotation_map = {
        "pressure_readout": pressure_bbox,
        "initial_cylinder": initial_bbox,
        "final_cylinder": final_bbox,
    }
    scene_entities = [
        {"id": "piston_cylinder", "bbox_px": piston_cylinder_bbox},
        {
            "id": "pressure_readout",
            "pressure_mpa": int(scenario.pressure_mpa),
            "bbox_px": pressure_bbox,
        },
        {
            "id": "initial_state_label",
            "pressure_mpa": int(scenario.pressure_mpa),
            "volume_l": int(scenario.initial_volume_l),
            "bbox_px": initial_state_label,
        },
        {
            "id": "final_state_label",
            "pressure_mpa": int(scenario.pressure_mpa),
            "volume_l": int(scenario.final_volume_l),
            "bbox_px": final_state_label,
        },
        {"id": "process_arrow", "bbox_px": process_arrow_bbox},
        {
            "id": "initial_cylinder",
            "volume_l": int(scenario.initial_volume_l),
            "bbox_px": initial_bbox,
        },
        {
            "id": "final_cylinder",
            "volume_l": int(scenario.final_volume_l),
            "bbox_px": final_bbox,
        },
        {
            "id": "initial_piston",
            "volume_l": int(scenario.initial_volume_l),
            "bbox_px": initial_piston_bbox,
        },
        {
            "id": "final_piston",
            "volume_l": int(scenario.final_volume_l),
            "bbox_px": final_piston_bbox,
        },
    ]
    render_map = {
        "orientation": str(scenario.orientation),
        "pressure_mpa": int(scenario.pressure_mpa),
        "initial_volume_l": int(scenario.initial_volume_l),
        "final_volume_l": int(scenario.final_volume_l),
        "delta_volume_l": int(scenario.final_volume_l - scenario.initial_volume_l),
        "boundary_work_kj": int(scenario.boundary_work_kj),
        "initial_cylinder_bbox_px": initial_bbox,
        "final_cylinder_bbox_px": final_bbox,
        "initial_piston_bbox_px": initial_piston_bbox,
        "final_piston_bbox_px": final_piston_bbox,
        "pressure_label_bbox_px": pressure_bbox,
        "piston_cylinder_bbox_px": piston_cylinder_bbox,
        "gas_rgb": list(int(value) for value in gas_rgb),
    }
    return RenderedPistonScene(
        image=image,
        annotation_bbox_map={str(key): list(value) for key, value in annotation_map.items()},
        scene_entities=scene_entities,
        render_map=render_map,
        background_meta=dict(background_meta),
        diagram_style_meta=dict(diagram_style_meta),
        post_noise_meta=dict(post_noise_meta),
        font_family=str(font_family),
    )


__all__ = [
    "draw_horizontal_cylinder",
    "draw_label_box",
    "draw_vertical_cylinder",
    "render_piston_cylinder_scene",
]
