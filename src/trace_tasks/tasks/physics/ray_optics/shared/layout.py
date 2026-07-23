"""Layout helpers for ray-optics diagrams."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Tuple

from trace_tasks.tasks.shared.render_variation import resolve_layout_jitter, resolve_render_int

from .state import RayOpticsTaskDefaults


RENDER_DEFAULT_KEYS: tuple[str, ...] = (
    "canvas_width",
    "canvas_height",
    "board_left_px",
    "board_top_px",
    "board_cols",
    "board_rows",
    "cell_size_px",
    "board_grid_width_px",
    "board_outline_width_px",
    "mirror_width_px",
    "mirror_padding_px",
    "ray_width_px",
    "ray_head_length_px",
    "ray_head_width_px",
    "target_radius_px",
    "source_radius_px",
    "bounce_radius_px",
    "target_font_size_px",
    "source_font_size_px",
    "label_stroke_width_px",
)


def resolve_board_render_defaults(
    *,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    fallback_defaults: RayOpticsTaskDefaults,
    instance_seed: int | None,
    namespace: str,
) -> Dict[str, Any]:
    """Return resolved render defaults for one optics board."""

    return {
        key: resolve_render_int(
            params,
            rendering_defaults,
            key,
            int(getattr(fallback_defaults, key)),
            instance_seed=instance_seed,
            namespace=str(namespace),
        )
        for key in RENDER_DEFAULT_KEYS
    }


def optics_content_bbox(render_defaults: Mapping[str, Any]) -> List[float]:
    """Return the conservative full-board bbox before layout offset."""

    board_left = float(render_defaults["board_left_px"])
    board_top = float(render_defaults["board_top_px"])
    board_cols = int(render_defaults["board_cols"])
    board_rows = int(render_defaults["board_rows"])
    cell_size = float(render_defaults["cell_size_px"])
    board_right = float(board_left + (float(board_cols) * cell_size))
    board_bottom = float(board_top + (float(board_rows) * cell_size))
    return [
        round(float(board_left - (0.95 * cell_size)), 3),
        round(float(board_top - (0.65 * cell_size)), 3),
        round(float(board_right + (0.70 * cell_size)), 3),
        round(float(board_bottom + (0.60 * cell_size)), 3),
    ]


def resolve_optics_layout_placement(
    *,
    render_defaults: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    params: Mapping[str, Any],
    instance_seed: int,
    namespace: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve whole-board placement before rendering and annotation projection."""

    canvas_width = int(render_defaults["canvas_width"])
    canvas_height = int(render_defaults["canvas_height"])
    content_bbox = optics_content_bbox(render_defaults)
    content_left, content_top, content_right, content_bottom = [
        float(value) for value in content_bbox
    ]
    jitter = resolve_layout_jitter(
        params,
        rendering_defaults,
        instance_seed=int(instance_seed),
        namespace=f"{namespace}.optics_layout",
    )
    min_margin = int(jitter.get("min_margin_px", 8))
    requested_dx = int(jitter.get("requested_dx_px", 0))
    requested_dy = int(jitter.get("requested_dy_px", 0))
    min_dx = int(math.ceil(float(min_margin) - float(content_left)))
    max_dx = int(
        math.floor(float(canvas_width) - float(min_margin) - float(content_right))
    )
    min_dy = int(math.ceil(float(min_margin) - float(content_top)))
    max_dy = int(
        math.floor(float(canvas_height) - float(min_margin) - float(content_bottom))
    )
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
    adjusted["board_left_px"] = int(render_defaults["board_left_px"]) + int(dx)
    adjusted["board_top_px"] = int(render_defaults["board_top_px"]) + int(dy)
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
            "mode": "whole_ray_optics_board_offset",
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
            "default_origin_px": [
                round(float(content_left), 3),
                round(float(content_top), 3),
            ],
            "final_origin_px": [
                round(float(content_left) + float(dx), 3),
                round(float(content_top) + float(dy), 3),
            ],
            "dx_px": int(dx),
            "dy_px": int(dy),
        }
    )
    return adjusted, placement


__all__ = [
    "RENDER_DEFAULT_KEYS",
    "optics_content_bbox",
    "resolve_board_render_defaults",
    "resolve_optics_layout_placement",
]
