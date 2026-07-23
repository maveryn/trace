"""Layout resolution for rectangular cell-board renderings."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import (
    resolve_required_float_bounds,
    resolve_required_int_bounds,
)

from .state import BoardLayout


def resolve_tile_size(
    *,
    rng,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> tuple[int, int, dict[str, Any]]:
    """Sample one board-uniform tile size with rectangular aspect jitter."""

    short_min, short_max = resolve_required_int_bounds(
        params,
        rendering_defaults,
        min_key="short_side_px_min",
        max_key="short_side_px_max",
        fallback_min=28,
        fallback_max=56,
        context="cell-board tile size",
    )
    aspect_min, aspect_max = resolve_required_float_bounds(
        params,
        rendering_defaults,
        min_key="aspect_ratio_min",
        max_key="aspect_ratio_max",
        fallback_min=1.0,
        fallback_max=2.0,
        context="cell-board tile aspect",
    )
    short_side = int(rng.randint(int(short_min), int(short_max)))
    aspect = float(rng.uniform(float(aspect_min), float(aspect_max)))
    long_side = min(
        int(round(float(short_side) * aspect)),
        int(float(short_side) * float(aspect_max)),
    )
    long_side = max(int(short_side), int(long_side))
    orientation = "wide" if bool(rng.randint(0, 1)) else "tall"
    if orientation == "wide":
        tile_w = int(long_side)
        tile_h = int(short_side)
    else:
        tile_w = int(short_side)
        tile_h = int(long_side)
    return max(1, tile_w), max(1, tile_h), {
        "short_side_px": int(short_side),
        "aspect_ratio": float(aspect),
        "orientation": str(orientation),
    }


def resolve_board_layout(
    *,
    rng,
    rows: int,
    cols: int,
    tile_width_px: int,
    tile_height_px: int,
    coordinate_labels: bool,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> BoardLayout:
    """Resolve a jittered canvas around one rectangular board."""

    padding_min, padding_max = resolve_required_float_bounds(
        params,
        rendering_defaults,
        min_key="outer_padding_fraction_min",
        max_key="outer_padding_fraction_max",
        fallback_min=0.08,
        fallback_max=0.12,
        context="cell-board outer padding",
    )
    jitter_min, jitter_max = resolve_required_float_bounds(
        params,
        rendering_defaults,
        min_key="placement_jitter_fraction_min",
        max_key="placement_jitter_fraction_max",
        fallback_min=0.02,
        fallback_max=0.06,
        context="cell-board placement jitter",
    )
    board_w = int(cols) * int(tile_width_px)
    board_h = int(rows) * int(tile_height_px)
    short_side = max(1, min(int(tile_width_px), int(tile_height_px)))
    label_font_size = max(14, int(round(float(short_side) * 0.48)))
    label_gutter_x = max(label_font_size + 18, int(tile_width_px * 0.70))
    label_gutter_y = max(label_font_size + 18, int(tile_height_px * 0.70))
    if not bool(coordinate_labels):
        label_gutter_x = 0
        label_gutter_y = 0

    padding_fraction = float(rng.uniform(float(padding_min), float(padding_max)))
    jitter_fraction = float(rng.uniform(float(jitter_min), float(jitter_max)))
    padding_x = max(16, int(round(float(board_w) * padding_fraction)))
    padding_y = max(16, int(round(float(board_h) * padding_fraction)))
    slack_x = max(0, int(round(float(board_w) * jitter_fraction)))
    slack_y = max(0, int(round(float(board_h) * jitter_fraction)))
    offset_x = int(rng.randint(0, slack_x)) if slack_x > 0 else 0
    offset_y = int(rng.randint(0, slack_y)) if slack_y > 0 else 0
    origin_x = int(label_gutter_x) + int(padding_x) + int(offset_x)
    origin_y = int(label_gutter_y) + int(padding_y) + int(offset_y)
    canvas_w = int(label_gutter_x) + int(board_w) + (2 * int(padding_x)) + slack_x
    canvas_h = int(label_gutter_y) + int(board_h) + (2 * int(padding_y)) + slack_y
    return BoardLayout(
        rows=int(rows),
        cols=int(cols),
        canvas_width_px=int(canvas_w),
        canvas_height_px=int(canvas_h),
        tile_width_px=int(tile_width_px),
        tile_height_px=int(tile_height_px),
        board_origin_x_px=int(origin_x),
        board_origin_y_px=int(origin_y),
        board_width_px=int(board_w),
        board_height_px=int(board_h),
        coordinate_labels=bool(coordinate_labels),
        label_font_size_px=int(label_font_size) if coordinate_labels else 0,
    )


__all__ = ["resolve_board_layout", "resolve_tile_size"]
