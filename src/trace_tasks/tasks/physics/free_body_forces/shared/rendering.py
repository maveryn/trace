"""Rendering primitives for free-body-force diagrams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from PIL import ImageDraw

from trace_tasks.core.visual.noise import apply_post_image_noise
from trace_tasks.tasks.physics.shared.diagram_style import prepare_physics_diagram_style_and_background
from trace_tasks.tasks.physics.shared.label_tags import draw_text_tag
from trace_tasks.tasks.physics.shared.visual_defaults import load_physics_noise_defaults
from trace_tasks.tasks.shared.bbox_projection import bbox_union_many
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.drawing import draw_arrow, draw_centered_text
from trace_tasks.tasks.shared.font_assets import sample_font_family
from trace_tasks.tasks.shared.named_colors import named_color
from trace_tasks.tasks.shared.text_rendering import load_font, resolve_text_stroke_fill

from .annotations import bbox, clip_bbox
from .formulas import unit_from_direction
from .state import (
    CARDINAL_VECTORS,
    OPTION_LETTERS,
    SCENE_ID,
    SCENE_NAMESPACE,
    ForceScenario,
    ForceSpec,
    RenderedForceScene,
    SamplingAxes,
)


POST_IMAGE_NOISE_DEFAULTS = load_physics_noise_defaults(scene_id=SCENE_ID, apply_prob=0.5)


def draw_grid(
    draw: ImageDraw.ImageDraw,
    *,
    bounds: Sequence[float],
    spacing_px: int,
    fill_rgb: tuple[int, int, int],
) -> None:
    """Draw a simple grid inside the force panel."""

    left, top, right, bottom = [float(value) for value in bounds]
    spacing = max(24.0, float(spacing_px))
    x = left + spacing
    while x < right:
        draw.line((x, top, x, bottom), fill=fill_rgb, width=1)
        x += spacing
    y = top + spacing
    while y < bottom:
        draw.line((left, y, right, y), fill=fill_rgb, width=1)
        y += spacing


def draw_force_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    force: ForceSpec,
    object_center: tuple[float, float],
    object_radius: float,
    arrow_length_px: float,
    font: Any,
    style: Any,
    force_rgb: tuple[int, int, int],
    lateral_offset_px: float,
    stroke_width_px: int,
    head_length_px: float,
    head_width_px: float,
) -> tuple[list[float], list[float], dict[str, Any]]:
    """Draw one applied-force arrow plus magnitude label."""

    unit_x, unit_y = unit_from_direction(force.direction)
    perp_x, perp_y = -unit_y, unit_x
    start = (
        float(object_center[0] + unit_x * (float(object_radius) + 9.0) + perp_x * float(lateral_offset_px)),
        float(object_center[1] + unit_y * (float(object_radius) + 9.0) + perp_y * float(lateral_offset_px)),
    )
    end = (
        float(start[0] + unit_x * float(arrow_length_px)),
        float(start[1] + unit_y * float(arrow_length_px)),
    )
    draw_arrow(
        draw,
        start=start,
        end=end,
        fill=force_rgb,
        width=max(1, int(stroke_width_px)),
        head_length_px=float(head_length_px),
        head_width_px=float(head_width_px),
    )
    arrow_bbox = bbox(
        (
            min(start[0], end[0]) - 18.0,
            min(start[1], end[1]) - 18.0,
            max(start[0], end[0]) + 18.0,
            max(start[1], end[1]) + 18.0,
        )
    )
    label_center = (
        float(end[0] + unit_x * 28.0 + perp_x * float(lateral_offset_px) * 3.0),
        float(end[1] + unit_y * 28.0 + perp_y * float(lateral_offset_px) * 3.0),
    )
    label_bbox = draw_text_tag(
        draw,
        text=f"{force.force_id}={force.magnitude_n} N",
        center=label_center,
        font=font,
        fill_rgb=tuple(int(value) for value in style.label_fill_rgb),
        outline_rgb=tuple(int(value) for value in style.label_border_rgb),
        text_rgb=tuple(int(value) for value in style.stroke_rgb),
        stroke_width_px=1,
    )
    annotation_bbox = bbox_union_many(arrow_bbox, label_bbox)
    return (
        annotation_bbox,
        label_bbox,
        {
            "start": [round(start[0], 3), round(start[1], 3)],
            "end": [round(end[0], 3), round(end[1], 3)],
        },
    )


def render_scene_contents(
    *,
    draw: ImageDraw.ImageDraw,
    scenario: ForceScenario,
    axes: SamplingAxes,
    font_family: str,
    style: Any,
    render_defaults: Mapping[str, Any],
    image_size: tuple[int, int],
) -> tuple[dict[str, list[float]], list[dict[str, Any]], dict[str, Any]]:
    """Draw force scene contents and preserve applied-force bbox witnesses."""

    width, height = int(image_size[0]), int(image_size[1])
    title_font = load_font(
        int(render_defaults.get("title_font_size_px", 27)),
        bold=True,
        font_family=font_family,
    )
    label_font = load_font(
        int(render_defaults.get("label_font_size_px", 22)),
        bold=True,
        font_family=font_family,
    )
    option_font = load_font(
        int(render_defaults.get("option_font_size_px", 22)),
        bold=True,
        font_family=font_family,
    )
    stroke_rgb = tuple(int(value) for value in style.stroke_rgb)
    guide_rgb = tuple(int(value) for value in style.guide_rgb)
    accent_rgb = named_color(str(axes.accent_color_name))
    force_rgb = tuple(
        max(0, min(255, int(0.72 * accent_rgb[index] + 0.28 * stroke_rgb[index])))
        for index in range(3)
    )
    panel = (
        float(render_defaults.get("panel_left_px", 52)),
        float(render_defaults.get("panel_top_px", 52)),
        float(render_defaults.get("panel_right_px", width - 52)),
        float(render_defaults.get("panel_bottom_px", height - 52)),
    )
    draw.rounded_rectangle(
        panel,
        radius=int(render_defaults.get("panel_corner_radius_px", 22)),
        fill=tuple(int(value) for value in style.panel_fill_rgb),
        outline=tuple(int(value) for value in style.panel_border_rgb),
        width=3,
    )
    if str(scenario.scene_variant) in {"gridded_table", "lab_card"}:
        draw_grid(
            draw,
            bounds=panel,
            spacing_px=int(render_defaults.get("grid_spacing_px", 44)),
            fill_rgb=tuple(int(value) for value in style.guide_rgb),
        )

    title_text = "Applied forces on one object"
    title_center = ((panel[0] + panel[2]) * 0.5, panel[1] + 34.0)
    title_bbox = draw.textbbox((0, 0), title_text, font=title_font, stroke_width=1)
    title_width = float(title_bbox[2] - title_bbox[0])
    title_height = float(title_bbox[3] - title_bbox[1])
    title_backdrop = (
        title_center[0] - title_width * 0.5 - 14.0,
        title_center[1] - title_height * 0.5 - 8.0,
        title_center[0] + title_width * 0.5 + 14.0,
        title_center[1] + title_height * 0.5 + 8.0,
    )
    draw.rounded_rectangle(
        title_backdrop,
        radius=9,
        fill=tuple(int(value) for value in style.panel_fill_rgb),
    )
    draw_centered_text(
        draw,
        text=title_text,
        center=title_center,
        font=title_font,
        fill=stroke_rgb,
        stroke_fill=resolve_text_stroke_fill(stroke_rgb),
        stroke_width=1,
    )

    object_center = (
        float(render_defaults.get("object_center_x_px", 590)),
        float(render_defaults.get("object_center_y_px", 340)),
    )
    object_width = float(render_defaults.get("object_width_px", 148))
    object_height = float(render_defaults.get("object_height_px", 104))
    object_bbox = bbox(
        (
            object_center[0] - object_width * 0.5,
            object_center[1] - object_height * 0.5,
            object_center[0] + object_width * 0.5,
            object_center[1] + object_height * 0.5,
        )
    )
    draw.rounded_rectangle(
        tuple(object_bbox),
        radius=18,
        fill=tuple(int(value) for value in style.panel_alt_fill_rgb),
        outline=stroke_rgb,
        width=4,
    )
    draw_centered_text(
        draw,
        text="object",
        center=object_center,
        font=label_font,
        fill=stroke_rgb,
        stroke_fill=resolve_text_stroke_fill(stroke_rgb),
        stroke_width=1,
    )

    annotation_bboxes: list[list[float]] = []
    force_entities: list[dict[str, Any]] = []
    force_arrow_map: dict[str, Any] = {}
    direction_counts = {
        direction: sum(1 for spec in scenario.force_specs if spec.direction == direction)
        for direction in CARDINAL_VECTORS
    }
    direction_seen = {direction: 0 for direction in CARDINAL_VECTORS}
    for force in scenario.force_specs:
        count_for_direction = max(1, int(direction_counts.get(str(force.direction), 1)))
        seen_for_direction = int(direction_seen.get(str(force.direction), 0))
        direction_seen[str(force.direction)] = seen_for_direction + 1
        lateral_offset = (
            float(seen_for_direction) - (float(count_for_direction) - 1.0) * 0.5
        ) * 28.0
        annotation_bbox, label_bbox, arrow_points = draw_force_arrow(
            draw,
            force=force,
            object_center=object_center,
            object_radius=max(object_width, object_height) * 0.5,
            arrow_length_px=float(render_defaults.get("force_arrow_length_px", 112)),
            font=label_font,
            style=style,
            force_rgb=force_rgb,
            lateral_offset_px=float(lateral_offset),
            stroke_width_px=int(render_defaults.get("force_arrow_width_px", 8)),
            head_length_px=float(render_defaults.get("force_arrow_head_length_px", 24)),
            head_width_px=float(render_defaults.get("force_arrow_head_width_px", 20)),
        )
        clipped_annotation = clip_bbox(annotation_bbox, width=width, height=height)
        clipped_label = clip_bbox(label_bbox, width=width, height=height)
        annotation_bboxes.append(clipped_annotation)
        force_arrow_map[str(force.force_id)] = {
            "direction": str(force.direction),
            "magnitude_n": int(force.magnitude_n),
            "vector": [int(force.vector[0]), int(force.vector[1])],
            "bbox_px": list(clipped_annotation),
            "label_bbox_px": list(clipped_label),
            **arrow_points,
        }
        force_entities.append(
            {
                "entity_id": str(force.force_id),
                "entity_type": "applied_force",
                "bbox_px": list(clipped_annotation),
                "meta": {
                    "direction": str(force.direction),
                    "magnitude_n": int(force.magnitude_n),
                    "vector": [int(force.vector[0]), int(force.vector[1])],
                },
            }
        )

    option_panel_top = float(render_defaults.get("option_panel_top_px", 594))
    option_cell_left = float(render_defaults.get("option_cell_left_px", 58))
    option_cell_width = float(render_defaults.get("option_cell_width_px", 132))
    option_cell_height = float(render_defaults.get("option_cell_height_px", 92))
    option_arrow_length = float(render_defaults.get("option_arrow_length_px", 54))
    option_arrow_rgb = tuple(int(value) for value in style.secondary_accent_rgb)
    option_bboxes: dict[str, list[float]] = {}
    for index, letter in enumerate(OPTION_LETTERS):
        left = option_cell_left + index * option_cell_width
        cell = bbox(
            (
                left,
                option_panel_top,
                left + option_cell_width - 12.0,
                option_panel_top + option_cell_height,
            )
        )
        draw.rounded_rectangle(
            tuple(cell),
            radius=12,
            fill=tuple(int(value) for value in style.panel_alt_fill_rgb),
            outline=guide_rgb,
            width=2,
        )
        draw_centered_text(
            draw,
            text=str(letter),
            center=(cell[0] + 22.0, cell[1] + 22.0),
            font=option_font,
            fill=stroke_rgb,
            stroke_fill=resolve_text_stroke_fill(stroke_rgb),
            stroke_width=1,
        )
        direction = str(scenario.option_directions[str(letter)])
        unit_x, unit_y = unit_from_direction(direction)
        center = ((cell[0] + cell[2]) * 0.5 + 10.0, (cell[1] + cell[3]) * 0.5 + 10.0)
        start = (
            center[0] - unit_x * option_arrow_length * 0.5,
            center[1] - unit_y * option_arrow_length * 0.5,
        )
        end = (
            center[0] + unit_x * option_arrow_length * 0.5,
            center[1] + unit_y * option_arrow_length * 0.5,
        )
        draw_arrow(
            draw,
            start=start,
            end=end,
            fill=option_arrow_rgb,
            width=int(render_defaults.get("option_arrow_width_px", 6)),
            head_length_px=float(render_defaults.get("option_arrow_head_length_px", 18)),
            head_width_px=float(render_defaults.get("option_arrow_head_width_px", 16)),
        )
        option_bboxes[str(letter)] = list(cell)

    force_diagram_bbox = clip_bbox(
        bbox_union_many(clip_bbox(object_bbox, width=width, height=height), *annotation_bboxes),
        width=width,
        height=height,
    )
    annotation_bbox_map = {
        "force_diagram": list(force_diagram_bbox),
        "selected_candidate": list(option_bboxes[str(scenario.correct_option_letter)]),
    }
    scene_entities = [
        {
            "entity_id": "object",
            "entity_type": "body",
            "bbox_px": clip_bbox(object_bbox, width=width, height=height),
            "meta": {"role": "object_with_applied_forces"},
        },
        {
            "entity_id": "force_diagram",
            "entity_type": "force_diagram_region",
            "bbox_px": list(force_diagram_bbox),
            "meta": {"role": "object_and_applied_forces"},
        },
        {
            "entity_id": "selected_candidate",
            "entity_type": "net_force_candidate_option",
            "bbox_px": list(option_bboxes[str(scenario.correct_option_letter)]),
            "meta": {
                "role": "selected_candidate_arrow",
                "option_letter": str(scenario.correct_option_letter),
                "direction": str(scenario.net_force_direction),
            },
        },
        *force_entities,
    ]
    render_map = {
        "panel_bbox_px": bbox(panel),
        "object_bbox_px": clip_bbox(object_bbox, width=width, height=height),
        "force_arrows": dict(force_arrow_map),
        "applied_force_bbox_set_px": [list(item) for item in annotation_bboxes],
        "force_diagram_bbox_px": list(force_diagram_bbox),
        "annotation_bbox_map_px": {str(key): list(value) for key, value in annotation_bbox_map.items()},
        "option_bboxes_px": {
            str(letter): list(item) for letter, item in sorted(option_bboxes.items())
        },
        "option_directions": dict(scenario.option_directions),
        "correct_option_letter": str(scenario.correct_option_letter),
        "net_force_direction": str(scenario.net_force_direction),
        "resultant_vector": [
            int(scenario.resultant_vector[0]),
            int(scenario.resultant_vector[1]),
        ],
    }
    return (
        {str(key): list(value) for key, value in annotation_bbox_map.items()},
        [dict(entity) for entity in scene_entities],
        dict(render_map),
    )


def render_free_body_forces(
    *,
    instance_seed: int,
    params: Mapping[str, Any],
    scenario: ForceScenario,
    axes: SamplingAxes,
    rendering_defaults: Mapping[str, Any],
    namespace: str = SCENE_NAMESPACE,
) -> RenderedForceScene:
    """Render one free-body-force diagram with option arrows."""

    canvas_width = int(params.get("canvas_width", group_default(rendering_defaults, "canvas_width", 1180)))
    canvas_height = int(params.get("canvas_height", group_default(rendering_defaults, "canvas_height", 760)))
    background, background_meta, diagram_style, diagram_style_meta = (
        prepare_physics_diagram_style_and_background(
            scene_id=SCENE_ID,
            canvas_width=int(canvas_width),
            canvas_height=int(canvas_height),
            instance_seed=int(instance_seed),
            params=params,
        )
    )
    font_family = sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.font",
        params=params,
    )
    draw = ImageDraw.Draw(background)
    annotation_bbox_map, scene_entities, render_map = render_scene_contents(
        draw=draw,
        scenario=scenario,
        axes=axes,
        font_family=str(font_family),
        style=diagram_style,
        render_defaults=rendering_defaults,
        image_size=background.size,
    )
    image, post_noise_meta = apply_post_image_noise(
        background,
        instance_seed=int(instance_seed),
        params=params,
        default_config=POST_IMAGE_NOISE_DEFAULTS,
    )
    return RenderedForceScene(
        image=image,
        annotation_bbox_map={str(key): list(value) for key, value in annotation_bbox_map.items()},
        scene_entities=[dict(entity) for entity in scene_entities],
        render_map=dict(render_map),
        background_meta=dict(background_meta),
        diagram_style_meta=dict(diagram_style_meta),
        post_noise_meta=dict(post_noise_meta),
        font_family=str(font_family),
    )
