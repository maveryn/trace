"""Default resolution helpers for word-search puzzle tasks."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from trace_tasks.tasks.puzzles.shared.unit_size_jitter import (
    resolve_puzzle_unit_size_scale,
    scale_puzzle_px,
)
from trace_tasks.tasks.shared.color_distance import coerce_rgb
from trace_tasks.tasks.shared.config_defaults import (
    group_default,
)

from .state import WordSearchRenderParams


def get_int_param(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: int,
) -> int:
    """Resolve one integer parameter from params/defaults."""

    return int(params.get(str(key), group_default(defaults, str(key), int(fallback))))


def get_int_range(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    min_key: str,
    max_key: str,
    fallback_min: int,
    fallback_max: int,
) -> tuple[int, int]:
    """Resolve one inclusive integer range from params/defaults."""

    low = get_int_param(params, defaults, str(min_key), int(fallback_min))
    high = get_int_param(params, defaults, str(max_key), int(fallback_max))
    if low > high:
        raise ValueError(f"{min_key} must be <= {max_key}")
    return int(low), int(high)


def resolve_render_params(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    instance_seed: int,
) -> WordSearchRenderParams:
    """Resolve word-search render parameters with unit-size jitter."""

    unit_scale, unit_meta = resolve_puzzle_unit_size_scale(
        params,
        defaults,
        instance_seed=int(instance_seed),
        namespace="puzzles.word_search.unit_size",
    )
    return WordSearchRenderParams(
        canvas_width=int(defaults.get("canvas_width", 1180)),
        canvas_height=int(defaults.get("canvas_height", 880)),
        cell_size_px=scale_puzzle_px(
            defaults.get("cell_size_px", 58), unit_scale, min_px=42
        ),
        header_size_px=scale_puzzle_px(
            defaults.get("header_size_px", 46), unit_scale, min_px=32
        ),
        panel_padding_px=scale_puzzle_px(
            defaults.get("panel_padding_px", 28), unit_scale, min_px=18
        ),
        panel_corner_radius_px=scale_puzzle_px(
            defaults.get("panel_corner_radius_px", 18),
            unit_scale,
            min_px=8,
        ),
        grid_line_width_px=scale_puzzle_px(
            defaults.get("grid_line_width_px", 2), unit_scale, min_px=1
        ),
        option_panel_width_px=scale_puzzle_px(
            defaults.get("option_panel_width_px", 270), unit_scale, min_px=190
        ),
        option_panel_height_px=scale_puzzle_px(
            defaults.get("option_panel_height_px", 58), unit_scale, min_px=42
        ),
        option_gap_px=scale_puzzle_px(
            defaults.get("option_gap_px", 12), unit_scale, min_px=8
        ),
        option_font_size_px=scale_puzzle_px(
            defaults.get("option_font_size_px", 20), unit_scale, min_px=15
        ),
        letter_font_size_px=scale_puzzle_px(
            defaults.get("letter_font_size_px", 24), unit_scale, min_px=18
        ),
        index_font_size_px=scale_puzzle_px(
            defaults.get("index_font_size_px", 17), unit_scale, min_px=13
        ),
        panel_fill_rgb=coerce_rgb(defaults.get("panel_fill_rgb"), (250, 251, 253)),
        grid_fill_rgb=coerce_rgb(defaults.get("grid_fill_rgb"), (255, 255, 255)),
        header_fill_rgb=coerce_rgb(defaults.get("header_fill_rgb"), (239, 243, 248)),
        grid_line_rgb=coerce_rgb(defaults.get("grid_line_rgb"), (93, 102, 116)),
        text_rgb=coerce_rgb(defaults.get("text_rgb"), (26, 31, 39)),
        text_stroke_rgb=coerce_rgb(defaults.get("text_stroke_rgb"), (255, 255, 255)),
        option_fill_rgb=coerce_rgb(defaults.get("option_fill_rgb"), (255, 250, 224)),
        option_border_rgb=coerce_rgb(defaults.get("option_border_rgb"), (54, 96, 168)),
        option_text_rgb=coerce_rgb(defaults.get("option_text_rgb"), (26, 31, 39)),
        unit_size_jitter=dict(unit_meta),
    )


def resize_canvas_to_content(render_params, *, dataset, rng) -> WordSearchRenderParams:
    """Shrink canvas around the grid plus any side-panel content."""

    cell = int(render_params.cell_size_px)
    header = int(render_params.header_size_px)
    padding = int(render_params.panel_padding_px)
    rows = int(dataset.rows)
    cols = int(dataset.cols)
    grid_w = int(header + cols * cell)
    grid_h = int(header + rows * cell)
    option_count = len(dataset.option_specs)
    if option_count == 4:
        option_columns, option_rows = 2, 2
    elif option_count == 6:
        option_columns, option_rows = 3, 2
    elif option_count > 0:
        option_columns = min(3, int(option_count))
        option_rows = (int(option_count) + option_columns - 1) // option_columns
    else:
        option_columns, option_rows = 0, 0
    option_grid_w = int(
        option_columns * int(render_params.option_panel_width_px)
        + max(0, option_columns - 1) * int(render_params.option_gap_px)
    )
    option_grid_h = int(
        option_rows * int(render_params.option_panel_height_px)
        + max(0, option_rows - 1) * int(render_params.option_gap_px)
    )
    option_gap_y = int(render_params.option_gap_px) * 2 if option_count else 0
    content_w = max(int(grid_w), int(option_grid_w))
    content_h = int(grid_h + option_gap_y + option_grid_h)
    panel_w = int(content_w + 2 * padding)
    panel_h = int(content_h + 2 * padding)
    margin = 34
    slack_x = int(rng.randrange(28, 91))
    slack_y = int(rng.randrange(24, 81))
    return replace(
        render_params,
        canvas_width=min(
            int(render_params.canvas_width), max(520, panel_w + 2 * margin + slack_x)
        ),
        canvas_height=min(
            int(render_params.canvas_height), max(460, panel_h + 2 * margin + slack_y)
        ),
    )


__all__ = [
    "get_int_param",
    "get_int_range",
    "resize_canvas_to_content",
    "resolve_render_params",
]
