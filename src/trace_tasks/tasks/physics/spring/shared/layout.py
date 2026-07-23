"""Layout placement helpers for the spring physics scene."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Tuple

from trace_tasks.tasks.shared.render_variation import resolve_layout_jitter


def spring_content_bbox(render_defaults: Mapping[str, Any], *, scene_variant: str) -> List[float]:
    """Return a conservative spring-diagram bbox before whole-scene placement."""

    card_left = float(render_defaults["card_left_px"])
    card_top = float(render_defaults["card_top_px"])
    card_width = float(render_defaults["card_width_px"])
    card_height = float(render_defaults["card_height_px"])
    card_gap = float(render_defaults["card_gap_px"])
    stagger_offset = float(render_defaults["stagger_offset_y_px"]) if str(scene_variant) == "staggered_springs" else 0.0
    right_left = float(card_left + card_width + card_gap)
    right_top = float(card_top + stagger_offset)
    max_column_extra_bottom = (
        34.0
        + float(render_defaults["support_height_px"])
        + float(render_defaults["anchor_y_gap_px"])
        + float(render_defaults["ruler_top_gap_px"])
        + (float(render_defaults["ruler_value_max"]) * float(render_defaults["ruler_unit_px"]))
        + float(render_defaults["weight_box_height_px"])
    )
    left = float(card_left - 14.0)
    top = float(min(card_top, right_top) - 14.0)
    right = float(right_left + card_width + 14.0)
    bottom = max(
        float(card_top + card_height),
        float(right_top + card_height),
        float(card_top + max_column_extra_bottom),
        float(right_top + max_column_extra_bottom),
    ) + 14.0
    return [round(left, 3), round(top, 3), round(right, 3), round(bottom, 3)]


def resolve_spring_layout_placement(
    *,
    render_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    scene_variant: str,
    namespace: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve whole-spring-diagram placement before rendering and annotation projection."""

    canvas_width = int(render_defaults["canvas_width"])
    canvas_height = int(render_defaults["canvas_height"])
    content_bbox = spring_content_bbox(render_defaults, scene_variant=str(scene_variant))
    content_left, content_top, content_right, content_bottom = [float(value) for value in content_bbox]
    jitter = resolve_layout_jitter(
        params,
        rendering_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.spring_layout",
    )
    min_margin = int(jitter.get("min_margin_px", 14))
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
    adjusted["card_left_px"] = int(render_defaults["card_left_px"]) + int(dx)
    adjusted["card_top_px"] = int(render_defaults["card_top_px"]) + int(dy)
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
            "mode": "whole_spring_diagram_offset",
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
            "final_origin_px": [round(float(content_left) + float(dx), 3), round(float(content_top) + float(dy), 3)],
            "dx_px": int(dx),
            "dy_px": int(dy),
        }
    )
    return adjusted, placement

__all__ = ["resolve_spring_layout_placement", "spring_content_bbox"]
