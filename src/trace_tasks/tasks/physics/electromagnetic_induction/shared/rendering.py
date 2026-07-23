"""Rendering primitives for electromagnetic induction diagrams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PIL import Image, ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.drawing import draw_arrow, draw_centered_text
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .state import SCENE_ID, InductionScenario, PanelSpec, RenderedInductionScene


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def draw_field_symbol(
    draw: ImageDraw.ImageDraw,
    *,
    center: tuple[float, float],
    direction: str,
    stroke_rgb: Sequence[int],
    fill_rgb: Sequence[int],
) -> None:
    """Draw a compact into-page or out-of-page magnetic-field marker."""

    x, y = float(center[0]), float(center[1])
    stroke = tuple(int(value) for value in stroke_rgb)
    fill = tuple(int(value) for value in fill_rgb)
    draw.ellipse((x - 11, y - 11, x + 11, y + 11), fill=fill, outline=stroke, width=2)
    if str(direction) == "out_of_page":
        draw.ellipse((x - 3.5, y - 3.5, x + 3.5, y + 3.5), fill=stroke)
    else:
        draw.line((x - 6, y - 6, x + 6, y + 6), fill=stroke, width=3)
        draw.line((x - 6, y + 6, x + 6, y - 6), fill=stroke, width=3)


def draw_field_region(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: tuple[float, float, float, float],
    direction: str,
    style: Any,
    shaded: bool,
) -> None:
    """Fill a panel region with repeated magnetic-field symbols."""

    left, top, right, bottom = [float(value) for value in bbox]
    if shaded:
        draw.rounded_rectangle(
            (left, top, right, bottom),
            radius=12,
            fill=tuple(int(value) for value in style.panel_alt_fill_rgb),
            outline=tuple(int(value) for value in style.panel_border_rgb),
            width=2,
        )
    spacing_x = max(54.0, (right - left) / 4.0)
    spacing_y = max(45.0, (bottom - top) / 4.0)
    x = left + spacing_x * 0.5
    while x <= right - 18:
        y = top + spacing_y * 0.5
        while y <= bottom - 18:
            draw_field_symbol(
                draw,
                center=(x, y),
                direction=str(direction),
                stroke_rgb=style.stroke_rgb,
                fill_rgb=style.panel_fill_rgb,
            )
            y += spacing_y
        x += spacing_x


def draw_loop(
    draw: ImageDraw.ImageDraw,
    *,
    bbox: tuple[float, float, float, float],
    stroke_rgb: Sequence[int],
) -> None:
    """Draw a conducting loop with a copper-colored inner stroke."""

    stroke = tuple(int(value) for value in stroke_rgb)
    copper = (178, 105, 42)
    draw.rounded_rectangle(bbox, radius=10, outline=stroke, width=8)
    inset = (bbox[0] + 4, bbox[1] + 4, bbox[2] - 4, bbox[3] - 4)
    draw.rounded_rectangle(inset, radius=8, outline=copper, width=4)


def draw_area_change_arrows(
    draw: ImageDraw.ImageDraw,
    *,
    loop_bbox: tuple[float, float, float, float],
    expanding: bool,
    fill_rgb: Sequence[int],
) -> None:
    """Draw outward or inward arrows for loop area changes."""

    left, top, right, bottom = [float(value) for value in loop_bbox]
    cx = 0.5 * (left + right)
    cy = 0.5 * (top + bottom)
    if bool(expanding):
        pairs = [
            ((left + 8, cy), (left - 30, cy)),
            ((right - 8, cy), (right + 30, cy)),
            ((cx, top + 8), (cx, top - 28)),
            ((cx, bottom - 8), (cx, bottom + 28)),
        ]
    else:
        pairs = [
            ((left - 30, cy), (left + 8, cy)),
            ((right + 30, cy), (right - 8, cy)),
            ((cx, top - 28), (cx, top + 8)),
            ((cx, bottom + 28), (cx, bottom - 8)),
        ]
    for start, end in pairs:
        draw_arrow(
            draw,
            start=start,
            end=end,
            fill=tuple(int(value) for value in fill_rgb),
            width=4,
            head_length_px=12,
            head_width_px=10,
        )


def panel_cue_text(mechanism: str) -> str:
    """Return the short visible cue label for a flux-change mechanism."""

    return {
        "loop_enters_field": "loop enters field",
        "loop_leaves_field": "loop leaves field",
        "field_strength_increases": "B stronger",
        "field_strength_decreases": "B weaker",
        "loop_area_expands": "area grows",
        "loop_area_contracts": "area shrinks",
        "loop_slides_inside_uniform_field": "slides in uniform B",
        "stationary_constant_field": "constant B",
    }[str(mechanism)]


def draw_panel(
    draw: ImageDraw.ImageDraw,
    *,
    panel: PanelSpec,
    font_family: str,
    style: Any,
) -> None:
    """Draw one induction panel from its symbolic field and flux-change state."""

    left, top, right, bottom = [float(value) for value in panel.bbox_px]
    panel_fill = tuple(int(value) for value in style.panel_fill_rgb)
    panel_border = tuple(int(value) for value in style.panel_border_rgb)
    label_rgb = tuple(int(value) for value in style.label_rgb)
    accent = tuple(int(value) for value in style.accent_rgb)
    draw.rounded_rectangle((left, top, right, bottom), radius=16, fill=panel_fill, outline=panel_border, width=3)

    cue_font = load_font(18, bold=True, font_family=font_family)
    cue_y = bottom - 26.0
    inner = (left + 18.0, top + 18.0, right - 18.0, bottom - 56.0)
    mechanism = str(panel.mechanism)
    full_field = mechanism not in {"loop_enters_field", "loop_leaves_field"}
    if full_field:
        draw_field_region(
            draw,
            bbox=inner,
            direction=str(panel.field_orientation),
            style=style,
            shaded=False,
        )
        loop_w, loop_h = 104.0, 78.0
        cx = 0.5 * (inner[0] + inner[2])
        cy = 0.5 * (inner[1] + inner[3])
        loop_bbox = (cx - loop_w / 2.0, cy - loop_h / 2.0, cx + loop_w / 2.0, cy + loop_h / 2.0)
        draw_loop(draw, bbox=loop_bbox, stroke_rgb=style.stroke_rgb)
        if mechanism == "loop_area_expands":
            draw_area_change_arrows(draw, loop_bbox=loop_bbox, expanding=True, fill_rgb=accent)
        elif mechanism == "loop_area_contracts":
            draw_area_change_arrows(draw, loop_bbox=loop_bbox, expanding=False, fill_rgb=accent)
        elif mechanism == "loop_slides_inside_uniform_field":
            draw_arrow(draw, start=(loop_bbox[2] + 16.0, cy), end=(loop_bbox[2] + 68.0, cy), fill=accent, width=5, head_length_px=15, head_width_px=13)
        elif mechanism == "field_strength_increases":
            draw_arrow(draw, start=(right - 68.0, top + 72.0), end=(right - 68.0, top + 28.0), fill=accent, width=5, head_length_px=14, head_width_px=12)
        elif mechanism == "field_strength_decreases":
            draw_arrow(draw, start=(right - 68.0, top + 28.0), end=(right - 68.0, top + 72.0), fill=accent, width=5, head_length_px=14, head_width_px=12)
    else:
        boundary_x = 0.5 * (inner[0] + inner[2])
        if panel.region_side == "right":
            field_bbox = (boundary_x, inner[1], inner[2], inner[3])
            enter_arrow = ((boundary_x - 92.0, 0.5 * (inner[1] + inner[3])), (boundary_x - 24.0, 0.5 * (inner[1] + inner[3])))
            leave_arrow = ((boundary_x + 80.0, 0.5 * (inner[1] + inner[3])), (boundary_x + 12.0, 0.5 * (inner[1] + inner[3])))
            loop_cx = boundary_x - 18.0 if mechanism == "loop_enters_field" else boundary_x + 50.0
        else:
            field_bbox = (inner[0], inner[1], boundary_x, inner[3])
            enter_arrow = ((boundary_x + 92.0, 0.5 * (inner[1] + inner[3])), (boundary_x + 24.0, 0.5 * (inner[1] + inner[3])))
            leave_arrow = ((boundary_x - 80.0, 0.5 * (inner[1] + inner[3])), (boundary_x - 12.0, 0.5 * (inner[1] + inner[3])))
            loop_cx = boundary_x + 18.0 if mechanism == "loop_enters_field" else boundary_x - 50.0
        draw_field_region(draw, bbox=field_bbox, direction=str(panel.field_orientation), style=style, shaded=True)
        loop_h = 76.0
        loop_w = 98.0
        cy = 0.5 * (inner[1] + inner[3])
        loop_bbox = (loop_cx - loop_w / 2.0, cy - loop_h / 2.0, loop_cx + loop_w / 2.0, cy + loop_h / 2.0)
        draw_loop(draw, bbox=loop_bbox, stroke_rgb=style.stroke_rgb)
        arrow_start, arrow_end = enter_arrow if mechanism == "loop_enters_field" else leave_arrow
        draw_arrow(draw, start=arrow_start, end=arrow_end, fill=accent, width=5, head_length_px=15, head_width_px=13)

    draw_centered_text(
        draw,
        text=panel_cue_text(mechanism),
        center=(0.5 * (left + right), cue_y),
        font=cue_font,
        fill=label_rgb,
        stroke_fill=resolve_text_stroke_fill(label_rgb),
        stroke_width=1,
    )


def draw_induction_panels(
    *,
    image: Image.Image,
    scenario: InductionScenario,
    font_family: str,
    style: Any,
) -> tuple[Image.Image, list[list[float]], list[dict[str, Any]], dict[str, Any]]:
    """Render every panel and return matching full-panel annotation boxes."""

    draw = ImageDraw.Draw(image)
    matching_bboxes: list[list[float]] = []
    entities: list[dict[str, Any]] = []
    panel_bboxes: dict[str, list[float]] = {}
    for panel in scenario.panels:
        draw_panel(draw, panel=panel, font_family=font_family, style=style)
        panel_bboxes[str(panel.panel_id)] = list(panel.bbox_px)
        is_match = str(panel.current_class) == str(scenario.target_current_class)
        if is_match:
            matching_bboxes.append(list(panel.bbox_px))
        entities.append(
            {
                "entity_id": str(panel.panel_id),
                "entity_type": "induction_panel",
                "bbox_px": list(panel.bbox_px),
                "meta": {
                    "field_orientation": str(panel.field_orientation),
                    "flux_change": str(panel.flux_change),
                    "mechanism": str(panel.mechanism),
                    "region_side": str(panel.region_side),
                    "induced_current_class": str(panel.current_class),
                    "matches_query": bool(is_match),
                },
            }
        )
    render_map = {
        "panel_bboxes": dict(panel_bboxes),
        "matching_panel_ids": [str(panel.panel_id) for panel in scenario.panels if str(panel.current_class) == str(scenario.target_current_class)],
        "panel_specs": [
            {
                "panel_id": str(panel.panel_id),
                "field_orientation": str(panel.field_orientation),
                "flux_change": str(panel.flux_change),
                "mechanism": str(panel.mechanism),
                "region_side": str(panel.region_side),
                "induced_current_class": str(panel.current_class),
            }
            for panel in scenario.panels
        ],
    }
    return image, [list(item) for item in matching_bboxes], [dict(entity) for entity in entities], dict(render_map)


def render_induction_scene(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: InductionScenario,
    rendering_defaults: Mapping[str, Any],
    namespace: str,
) -> RenderedInductionScene:
    """Prepare the physics background and render the six-panel induction grid."""

    canvas_width = int(params.get("canvas_width", group_default(rendering_defaults, "canvas_width", 1180)))
    canvas_height = int(params.get("canvas_height", group_default(rendering_defaults, "canvas_height", 820)))
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
    rendered, annotation_bboxes, scene_entities, render_map = draw_induction_panels(
        image=background,
        scenario=scenario,
        font_family=str(font_family),
        style=diagram_style,
    )
    image, post_noise_meta = apply_post_image_noise(
        rendered,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedInductionScene(
        image=image,
        annotation_bboxes=[list(bbox) for bbox in annotation_bboxes],
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
        background_meta=dict(background_meta),
        diagram_style_meta=dict(diagram_style_meta),
        post_noise_meta=dict(post_noise_meta),
        font_family=str(font_family),
    )
