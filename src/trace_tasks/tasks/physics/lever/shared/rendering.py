"""Rendering helpers for lever-balance diagrams."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Sequence, Tuple

from PIL import Image, ImageDraw

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.drawing import draw_centered_text_with_auto_stroke as _draw_centered_text
from trace_tasks.tasks.shared.drawing import draw_rounded_rect
from trace_tasks.tasks.shared.text_rendering import load_font
from trace_tasks.tasks.physics.shared.style import build_physics_lever_theme

from .layout import beam_center_x
from .state import LeverWeightPlacement, LeverWeightSlot, RenderedLeverScene


def _draw_beam_texture(
    draw: ImageDraw.ImageDraw,
    *,
    beam_bbox_px: Sequence[float],
    line_width_px: int,
    spacing_px: int,
    ink_rgb: Tuple[int, int, int],
) -> None:
    """Draw one subtle wood-like texture inside the beam."""

    left, top, right, bottom = [float(value) for value in beam_bbox_px]
    y_value = float(top + (0.5 * spacing_px))
    while float(y_value) < float(bottom):
        draw.line(
            [(float(left + 8.0), float(y_value)), (float(right - 8.0), float(y_value))],
            fill=ink_rgb,
            width=max(1, int(line_width_px)),
        )
        y_value += float(spacing_px)


def render_lever_scene(
    *,
    scene_variant: str,
    accent_color_name: str,
    placements: Sequence[LeverWeightSlot],
    render_defaults: Mapping[str, Any],
    background: Image.Image,
    diagram_style: Any | None = None,
    font_family: str | None = None,
) -> RenderedLeverScene:
    """Render final lever geometry while preserving projected weight boxes."""

    canvas = background.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    canvas_width = int(render_defaults["canvas_width"])
    lever_theme = build_physics_lever_theme(str(accent_color_name), diagram_style=diagram_style)
    scene_rng = spawn_rng(
        int(render_defaults.get("instance_seed", 0)),
        f"physics_lever.scene_layout.{str(scene_variant)}",
    )
    center_x = beam_center_x(
        scene_rng,
        scene_variant=str(scene_variant),
        canvas_width=int(canvas_width),
        fulcrum_offset_px=int(render_defaults["fulcrum_offset_px"]),
    ) + float(render_defaults.get("layout_offset_x_px", 0))
    beam_center_y = float(render_defaults["beam_center_y_px"]) + float(render_defaults.get("layout_offset_y_px", 0))
    beam_width = float(render_defaults["beam_width_px"])
    beam_height = float(render_defaults["beam_height_px"])
    beam_bbox_px = [
        round(float(center_x - (0.5 * beam_width)), 3),
        round(float(beam_center_y - (0.5 * beam_height)), 3),
        round(float(center_x + (0.5 * beam_width)), 3),
        round(float(beam_center_y + (0.5 * beam_height)), 3),
    ]
    fulcrum_width = float(render_defaults["fulcrum_width_px"])
    fulcrum_height = float(render_defaults["fulcrum_height_px"])
    fulcrum_bbox_px = [
        round(float(center_x - (0.5 * fulcrum_width)), 3),
        round(float(beam_bbox_px[3]), 3),
        round(float(center_x + (0.5 * fulcrum_width)), 3),
        round(float(beam_bbox_px[3] + fulcrum_height), 3),
    ]

    draw_rounded_rect(
        draw,
        tuple(float(value) for value in beam_bbox_px),
        radius=int(render_defaults["beam_corner_radius_px"]),
        fill=tuple(int(channel) for channel in lever_theme.beam_fill_rgb),
        outline=tuple(int(channel) for channel in lever_theme.beam_outline_rgb),
        width=3,
    )
    if str(scene_variant) == "textured_beam":
        _draw_beam_texture(
            draw,
            beam_bbox_px=beam_bbox_px,
            line_width_px=int(render_defaults["texture_line_width_px"]),
            spacing_px=int(render_defaults["texture_spacing_px"]),
            ink_rgb=tuple(int(channel) for channel in lever_theme.texture_rgb),
        )

    draw.polygon(
        [
            (float(center_x), float(fulcrum_bbox_px[1] + 6.0)),
            (float(fulcrum_bbox_px[0]), float(fulcrum_bbox_px[3])),
            (float(fulcrum_bbox_px[2]), float(fulcrum_bbox_px[3])),
        ],
        fill=tuple(int(channel) for channel in lever_theme.fulcrum_fill_rgb),
        outline=tuple(int(channel) for channel in lever_theme.fulcrum_outline_rgb),
    )

    distance_values = tuple(int(value) for value in render_defaults["distance_support"])
    slot_spacing = float(render_defaults["slot_spacing_px"])
    resolved_font_family = None if font_family is None else str(font_family)
    distance_font = load_font(int(render_defaults["distance_font_size_px"]), bold=True, font_family=resolved_font_family)
    weight_font = load_font(int(render_defaults["weight_font_size_px"]), bold=True, font_family=resolved_font_family)
    for distance_units in distance_values:
        for _side, sign in (("left", -1.0), ("right", 1.0)):
            tick_x = float(center_x + (sign * float(distance_units) * slot_spacing))
            draw.line(
                [(float(tick_x), float(beam_bbox_px[1] - 8.0)), (float(tick_x), float(beam_bbox_px[3] + 8.0))],
                fill=tuple(int(channel) for channel in lever_theme.beam_tick_rgb),
                width=2,
            )
            _draw_centered_text(
                draw,
                text=str(int(distance_units)),
                center_xy=(float(tick_x), float(beam_bbox_px[3] + 22.0)),
                font=distance_font,
                fill=tuple(int(channel) for channel in lever_theme.distance_text_rgb),
                stroke_width_px=max(0, int(render_defaults["label_stroke_width_px"]) - 1),
            )

    weight_specs: List[LeverWeightPlacement] = []
    relevant_weight_bboxes: List[List[float]] = []
    relevant_weight_ids: List[str] = []
    known_weight_bboxes: List[List[float]] = []
    target_weight_bboxes: List[List[float]] = []
    scene_entities: List[Dict[str, Any]] = [
        {
            "entity_id": "lever_beam",
            "entity_type": "physics_lever_beam",
            "bbox_px": list(beam_bbox_px),
            "meta": {"scene_variant": str(scene_variant)},
        },
        {
            "entity_id": "lever_fulcrum",
            "entity_type": "physics_fulcrum",
            "bbox_px": list(fulcrum_bbox_px),
        },
    ]
    placeholder_bbox_px: List[float] | None = None
    max_distance_units = 0
    for index, slot in enumerate(placements, start=1):
        sign = -1.0 if str(slot.side) == "left" else 1.0
        weight_center_x = float(center_x + (sign * float(slot.distance_units) * slot_spacing))
        box_width = float(render_defaults["weight_box_width_px"])
        box_height = float(render_defaults["weight_box_height_px"])
        box_left = float(weight_center_x - (0.5 * box_width))
        box_top = float(beam_bbox_px[1] - float(render_defaults["weight_box_gap_px"]) - box_height)
        box_bbox = [
            round(float(box_left), 3),
            round(float(box_top), 3),
            round(float(box_left + box_width), 3),
            round(float(box_top + box_height), 3),
        ]
        fill_rgb = (
            tuple(int(channel) for channel in lever_theme.weight_fill_rgb)
            if not slot.missing
            else (255, 238, 240)
        )
        outline_rgb = (
            tuple(int(channel) for channel in lever_theme.weight_outline_rgb)
            if not slot.missing
            else (192, 62, 84)
        )
        draw_rounded_rect(
            draw,
            tuple(float(value_px) for value_px in box_bbox),
            radius=10,
            fill=fill_rgb,
            outline=outline_rgb,
            width=3,
        )
        label_bbox = _draw_centered_text(
            draw,
            text="?" if slot.missing else str(int(slot.value or 0)),
            center_xy=(float(weight_center_x), float(box_top + (0.5 * box_height))),
            font=weight_font,
            fill=(196, 56, 79) if slot.missing else tuple(int(channel) for channel in lever_theme.weight_text_rgb),
            stroke_width_px=int(render_defaults["label_stroke_width_px"]),
        )
        weight_bbox = [
            round(float(min(box_bbox[0], label_bbox[0])), 3),
            round(float(min(box_bbox[1], label_bbox[1])), 3),
            round(float(max(box_bbox[2], label_bbox[2])), 3),
            round(float(max(box_bbox[3], label_bbox[3])), 3),
        ]
        weight_id = "missing_weight_marker" if slot.missing else f"lever_weight_{int(index)}"
        spec = LeverWeightPlacement(
            weight_id=str(weight_id),
            side=str(slot.side),
            distance_units=int(slot.distance_units),
            value=int(slot.value) if slot.value is not None else None,
            missing=bool(slot.missing),
            relevant=bool(slot.relevant),
            bbox_px=list(weight_bbox),
        )
        weight_specs.append(spec)
        scene_entities.append(
            {
                "entity_id": str(weight_id),
                "entity_type": "physics_missing_weight_marker" if slot.missing else "physics_lever_weight",
                "bbox_px": list(weight_bbox),
                "meta": {
                    "side": str(slot.side),
                    "distance_units": int(slot.distance_units),
                    "value": int(slot.value) if slot.value is not None else None,
                    "missing": bool(slot.missing),
                    "relevant_to_task": bool(slot.relevant),
                },
            }
        )
        max_distance_units = max(int(max_distance_units), int(slot.distance_units))
        if bool(slot.relevant):
            relevant_weight_bboxes.append(list(weight_bbox))
            relevant_weight_ids.append(str(weight_id))
            if bool(slot.missing):
                target_weight_bboxes.append(list(weight_bbox))
            else:
                known_weight_bboxes.append(list(weight_bbox))
        if bool(slot.missing):
            placeholder_bbox_px = list(weight_bbox)

    render_map = {
        "accent_color_name": str(accent_color_name),
        "technical_diagram_frame_mode": str(getattr(diagram_style, "frame_mode", "none")),
        "beam_bbox_px": list(beam_bbox_px),
        "fulcrum_bbox_px": list(fulcrum_bbox_px),
        "weight_bboxes_px": {spec.weight_id: list(spec.bbox_px) for spec in weight_specs},
        "relevant_weight_ids": list(relevant_weight_ids),
        "witness_entity_ids": list(relevant_weight_ids),
        "known_weight_bboxes_px": [list(bbox) for bbox in known_weight_bboxes],
        "target_weight_bboxes_px": [list(bbox) for bbox in target_weight_bboxes],
        "annotation_bbox_set_map_px": {
            "known_weights": [list(bbox) for bbox in known_weight_bboxes],
            "target_weight": [list(bbox) for bbox in target_weight_bboxes],
        },
        "beam_center_px": [round(float(center_x), 3), round(float(beam_center_y), 3)],
        "max_distance_units": int(max_distance_units),
    }
    if placeholder_bbox_px is not None:
        render_map["missing_weight_marker_bbox_px"] = list(placeholder_bbox_px)

    return RenderedLeverScene(
        image=canvas,
        beam_bbox_px=list(beam_bbox_px),
        fulcrum_bbox_px=list(fulcrum_bbox_px),
        weight_specs=list(weight_specs),
        placeholder_bbox_px=list(placeholder_bbox_px) if placeholder_bbox_px is not None else None,
        relevant_weight_bboxes=list(relevant_weight_bboxes),
        relevant_weight_ids=list(relevant_weight_ids),
        known_weight_bboxes=list(known_weight_bboxes),
        target_weight_bboxes=list(target_weight_bboxes),
        render_map=render_map,
        scene_entities=list(scene_entities),
        max_distance_units=int(max_distance_units),
    )
