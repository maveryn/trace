"""Layout helpers for counterfactual-board renderings."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.shared.config_defaults import (
    group_default,
    resolve_required_float_bounds,
    resolve_required_int_bounds,
)

from .state import BoardLayout, CELL_BOARD_KIND, LINE_BOARD_KIND


def resolve_board_layout(
    *,
    rng,
    rows: int,
    cols: int,
    board_kind: str,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
) -> BoardLayout:
    """Resolve canvas size, board bbox, and placement jitter for one board."""

    unit_min, unit_max = resolve_required_int_bounds(
        params,
        rendering_defaults,
        min_key="unit_size_px_min",
        max_key="unit_size_px_max",
        fallback_min=42,
        fallback_max=60,
        context="counterfactual-board unit size",
    )
    padding_min, padding_max = resolve_required_float_bounds(
        params,
        rendering_defaults,
        min_key="outer_padding_fraction_min",
        max_key="outer_padding_fraction_max",
        fallback_min=0.08,
        fallback_max=0.14,
        context="counterfactual-board outer padding",
    )
    jitter_min, jitter_max = resolve_required_float_bounds(
        params,
        rendering_defaults,
        min_key="placement_jitter_fraction_min",
        max_key="placement_jitter_fraction_max",
        fallback_min=0.03,
        fallback_max=0.08,
        context="counterfactual-board placement jitter",
    )
    unit_size = int(rng.randint(int(unit_min), int(unit_max)))
    if str(board_kind) == CELL_BOARD_KIND:
        board_w = int(cols) * int(unit_size)
        board_h = int(rows) * int(unit_size)
    elif str(board_kind) == LINE_BOARD_KIND:
        board_w = max(1, int(cols) - 1) * int(unit_size)
        board_h = max(1, int(rows) - 1) * int(unit_size)
    else:
        raise ValueError(f"unsupported board kind: {board_kind!r}")

    padding_fraction = float(rng.uniform(float(padding_min), float(padding_max)))
    jitter_fraction = float(rng.uniform(float(jitter_min), float(jitter_max)))
    padding_x = max(20, int(round(float(board_w) * padding_fraction)))
    padding_y = max(20, int(round(float(board_h) * padding_fraction)))
    slack_x = max(0, int(round(float(board_w) * jitter_fraction)))
    slack_y = max(0, int(round(float(board_h) * jitter_fraction)))
    offset_x = int(rng.randint(0, slack_x)) if slack_x else 0
    offset_y = int(rng.randint(0, slack_y)) if slack_y else 0
    origin_x = int(padding_x) + int(offset_x)
    origin_y = int(padding_y) + int(offset_y)
    canvas_w = int(board_w) + (2 * int(padding_x)) + int(slack_x)
    canvas_h = int(board_h) + (2 * int(padding_y)) + int(slack_y)
    line_thickness = int(
        params.get(
            "line_annotation_thickness_px",
            group_default(rendering_defaults, "line_annotation_thickness_px", 28),
        )
    )
    return BoardLayout(
        rows=int(rows),
        cols=int(cols),
        board_kind=str(board_kind),
        canvas_width_px=int(canvas_w),
        canvas_height_px=int(canvas_h),
        board_bbox_px=(
            float(origin_x),
            float(origin_y),
            float(origin_x + board_w),
            float(origin_y + board_h),
        ),
        unit_size_px=int(unit_size),
        line_annotation_thickness_px=max(24, int(line_thickness)),
        placement_meta={
            "unit_size_px": int(unit_size),
            "padding_fraction": float(padding_fraction),
            "jitter_fraction": float(jitter_fraction),
            "offset_px": [int(offset_x), int(offset_y)],
            "slack_px": [int(slack_x), int(slack_y)],
        },
    )


__all__ = ["resolve_board_layout"]
