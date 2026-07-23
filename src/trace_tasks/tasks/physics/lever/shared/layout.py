"""Layout projection helpers for lever-balance diagrams."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.shared.render_variation import resolve_layout_jitter

from .state import LeverWeightSlot


def beam_center_x(
    rng,
    *,
    scene_variant: str,
    canvas_width: int,
    fulcrum_offset_px: int,
) -> float:
    """Return the horizontal fulcrum center for one scene variant."""

    center_x = 0.5 * float(canvas_width)
    if str(scene_variant) != "offset_fulcrum":
        return float(center_x)
    direction = -1.0 if rng.random() < 0.5 else 1.0
    return float(center_x + (direction * float(fulcrum_offset_px)))


def lever_content_bbox(
    *,
    render_defaults: Mapping[str, Any],
    scene_variant: str,
    placements: Sequence[LeverWeightSlot],
) -> list[float]:
    """Return a conservative bbox before whole-diagram placement offset."""

    canvas_width = int(render_defaults["canvas_width"])
    scene_rng = spawn_rng(
        int(render_defaults.get("instance_seed", 0)),
        f"physics_lever.scene_layout.{str(scene_variant)}",
    )
    center_x = beam_center_x(
        scene_rng,
        scene_variant=str(scene_variant),
        canvas_width=int(canvas_width),
        fulcrum_offset_px=int(render_defaults["fulcrum_offset_px"]),
    )
    beam_center_y = float(render_defaults["beam_center_y_px"])
    beam_width = float(render_defaults["beam_width_px"])
    beam_height = float(render_defaults["beam_height_px"])
    beam_bbox_px = [
        float(center_x - (0.5 * beam_width)),
        float(beam_center_y - (0.5 * beam_height)),
        float(center_x + (0.5 * beam_width)),
        float(beam_center_y + (0.5 * beam_height)),
    ]
    fulcrum_width = float(render_defaults["fulcrum_width_px"])
    fulcrum_height = float(render_defaults["fulcrum_height_px"])
    fulcrum_bbox_px = [
        float(center_x - (0.5 * fulcrum_width)),
        float(beam_bbox_px[3]),
        float(center_x + (0.5 * fulcrum_width)),
        float(beam_bbox_px[3] + fulcrum_height),
    ]
    left = min(float(beam_bbox_px[0]), float(fulcrum_bbox_px[0]))
    top = min(float(beam_bbox_px[1] - 36.0), float(fulcrum_bbox_px[1]))
    right = max(float(beam_bbox_px[2]), float(fulcrum_bbox_px[2]))
    bottom = max(float(beam_bbox_px[3] + 50.0), float(fulcrum_bbox_px[3]))
    slot_spacing = float(render_defaults["slot_spacing_px"])
    box_width = float(render_defaults["weight_box_width_px"])
    box_height = float(render_defaults["weight_box_height_px"])
    for slot in placements:
        sign = -1.0 if str(slot.side) == "left" else 1.0
        weight_center_x = float(center_x + (sign * float(slot.distance_units) * slot_spacing))
        box_left = float(weight_center_x - (0.5 * box_width) - 8.0)
        box_top = float(beam_bbox_px[1] - float(render_defaults["weight_box_gap_px"]) - box_height - 8.0)
        box_right = float(weight_center_x + (0.5 * box_width) + 8.0)
        box_bottom = float(beam_bbox_px[1] - float(render_defaults["weight_box_gap_px"]) + 8.0)
        left = min(left, box_left)
        top = min(top, box_top)
        right = max(right, box_right)
        bottom = max(bottom, box_bottom)
    return [round(float(left), 3), round(float(top), 3), round(float(right), 3), round(float(bottom), 3)]


def resolve_lever_layout_placement(
    *,
    render_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    scene_variant: str,
    placements: Sequence[LeverWeightSlot],
    namespace: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve whole-diagram placement before annotation projection."""

    canvas_width = int(render_defaults["canvas_width"])
    canvas_height = int(render_defaults["canvas_height"])
    content_bbox = lever_content_bbox(
        render_defaults=render_defaults,
        scene_variant=str(scene_variant),
        placements=placements,
    )
    content_left, content_top, content_right, content_bottom = [float(value) for value in content_bbox]
    jitter = resolve_layout_jitter(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace=str(namespace),
    )
    min_margin = int(jitter.get("min_margin_px", 18))
    requested_dx = int(jitter.get("requested_dx_px", 0))
    requested_dy = int(jitter.get("requested_dy_px", 0))
    min_dx = int(math.ceil(float(min_margin) - float(content_left)))
    max_dx = int(math.floor(float(canvas_width) - float(min_margin) - float(content_right)))
    min_dy = int(math.ceil(float(min_margin) - float(content_top)))
    max_dy = int(math.floor(float(canvas_height) - float(min_margin) - float(content_bottom)))
    if int(min_dx) > int(max_dx):
        min_dx = 0
        max_dx = 0
    if int(min_dy) > int(max_dy):
        min_dy = 0
        max_dy = 0
    if not bool(jitter.get("enabled", False)):
        requested_dx = 0
        requested_dy = 0
    dx = max(int(min_dx), min(int(max_dx), int(requested_dx)))
    dy = max(int(min_dy), min(int(max_dy), int(requested_dy)))

    adjusted = dict(render_defaults)
    adjusted["layout_offset_x_px"] = int(dx)
    adjusted["layout_offset_y_px"] = int(dy)

    content_width = round(float(content_right) - float(content_left), 3)
    content_height = round(float(content_bottom) - float(content_top), 3)
    final_bbox = [
        round(float(content_left) + float(dx), 3),
        round(float(content_top) + float(dy), 3),
        round(float(content_right) + float(dx), 3),
        round(float(content_bottom) + float(dy), 3),
    ]
    placement = dict(jitter)
    placement.update(
        {
            "mode": "whole_lever_diagram_offset",
            "content_bbox_px": list(content_bbox),
            "content_size_px": [float(content_width), float(content_height)],
            "final_content_bbox_px": list(final_bbox),
            "canvas_size_px": [int(canvas_width), int(canvas_height)],
            "free_space_px": [
                round(float(canvas_width) - float(content_width), 3),
                round(float(canvas_height) - float(content_height), 3),
            ],
            "available_offset_x_px": [int(min_dx), int(max_dx)],
            "available_offset_y_px": [int(min_dy), int(max_dy)],
            "sampled_offset_px": [int(requested_dx), int(requested_dy)],
            "final_offset_px": [int(dx), int(dy)],
            "default_origin_px": [round(float(content_left), 3), round(float(content_top), 3)],
            "final_origin_px": [
                round(float(content_left) + float(dx), 3),
                round(float(content_top) + float(dy), 3),
            ],
            "dx_px": int(dx),
            "dy_px": int(dy),
        }
    )
    return adjusted, placement
