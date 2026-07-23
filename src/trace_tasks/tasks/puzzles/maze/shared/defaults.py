"""Config and visual-style defaults for maze puzzle rendering."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from trace_tasks.core.seed import spawn_rng
from trace_tasks.tasks.puzzles.shared.unit_size_jitter import (
    resolve_puzzle_unit_size_scale,
    scale_puzzle_px,
)
from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.font_assets import (
    font_asset_version,
    get_font_family_record,
    sample_font_family,
)

from .state import Color, MazeExitRenderParams, SCENE_NAMESPACE

def to_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)
def normalize_rgb(value: Any, fallback: Color) -> Color:
    if isinstance(value, (list, tuple)) and len(value) >= 3:
        return (
            max(0, min(255, to_int(value[0], fallback[0]))),
            max(0, min(255, to_int(value[1], fallback[1]))),
            max(0, min(255, to_int(value[2], fallback[2]))),
        )
    return tuple(int(channel) for channel in fallback)
def rgb_option(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: Color, *, seed: int) -> Color:
    raw = params.get(str(key), group_default(defaults, str(key), fallback))
    options = params.get(f"{key}_options", group_default(defaults, f"{key}_options", None))
    if isinstance(options, list) and options:
        rng = spawn_rng(int(seed), f"{SCENE_NAMESPACE}.render.{key}")
        return normalize_rgb(options[int(rng.randrange(len(options)))], fallback)
    return normalize_rgb(raw, fallback)
def rgb_palette(params: Mapping[str, Any], defaults: Mapping[str, Any], key: str, fallback: Sequence[Color]) -> Tuple[Color, ...]:
    raw = params.get(str(key), group_default(defaults, str(key), list(fallback)))
    if not isinstance(raw, list) or not raw:
        raw = list(fallback)
    palette = tuple(normalize_rgb(value, fallback[0]) for value in raw)
    return palette if palette else tuple(fallback)
def string_option(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    key: str,
    fallback: str,
    *,
    seed: int,
    allowed: Sequence[str] | None = None,
) -> str:
    raw = str(params.get(str(key), group_default(defaults, str(key), str(fallback)))).strip() or str(fallback)
    options = params.get(f"{key}_options", group_default(defaults, f"{key}_options", None))
    if isinstance(options, list) and options:
        candidates = [str(value).strip() for value in options if str(value).strip()]
        if candidates:
            rng = spawn_rng(int(seed), f"{SCENE_NAMESPACE}.render.{key}")
            raw = str(candidates[int(rng.randrange(len(candidates)))])
    if allowed is not None and raw not in set(map(str, allowed)):
        return str(fallback)
    return str(raw)
def resolve_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    instance_seed: int,
) -> MazeExitRenderParams:
    """Resolve scene-level render dimensions, style colors, and stroke widths.

    This helper is intentionally task-neutral: it reads only scene render
    defaults and explicit render params, then returns immutable drawing
    parameters consumed by the maze renderer.
    """

    unit_scale, unit_meta = resolve_puzzle_unit_size_scale(
        params,
        render_defaults,
        instance_seed=int(instance_seed),
        namespace="puzzles.maze.unit_size",
    )
    wall_default = to_int(params.get("wall_stroke_width_px", group_default(render_defaults, "wall_stroke_width_px", 6)), 6)
    wall_min = to_int(params.get("wall_stroke_width_min_px", group_default(render_defaults, "wall_stroke_width_min_px", wall_default)), wall_default)
    wall_max = to_int(params.get("wall_stroke_width_max_px", group_default(render_defaults, "wall_stroke_width_max_px", wall_default)), wall_default)
    if int(wall_min) > int(wall_max):
        raise ValueError("wall_stroke_width_min_px must be <= wall_stroke_width_max_px")
    wall_rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.render.wall_width")
    wall_width = int(wall_rng.randint(max(2, int(wall_min)), max(2, int(wall_max))))
    outer_wall_default = to_int(params.get("outer_wall_stroke_width_px", group_default(render_defaults, "outer_wall_stroke_width_px", 9)), 9)
    outer_wall_min = to_int(
        params.get("outer_wall_stroke_width_min_px", group_default(render_defaults, "outer_wall_stroke_width_min_px", outer_wall_default)),
        outer_wall_default,
    )
    outer_wall_max = to_int(
        params.get("outer_wall_stroke_width_max_px", group_default(render_defaults, "outer_wall_stroke_width_max_px", outer_wall_default)),
        outer_wall_default,
    )
    if int(outer_wall_min) > int(outer_wall_max):
        raise ValueError("outer_wall_stroke_width_min_px must be <= outer_wall_stroke_width_max_px")
    outer_wall_rng = spawn_rng(int(instance_seed), f"{SCENE_NAMESPACE}.render.outer_wall_width")
    outer_wall_width = int(outer_wall_rng.randint(max(3, int(outer_wall_min)), max(3, int(outer_wall_max))))
    return MazeExitRenderParams(
        canvas_width=max(760, to_int(params.get("canvas_width", group_default(render_defaults, "canvas_width", 1200)), 1200)),
        canvas_height=max(620, to_int(params.get("canvas_height", group_default(render_defaults, "canvas_height", 900)), 900)),
        scene_margin_left_px=max(76, to_int(params.get("scene_margin_left_px", group_default(render_defaults, "scene_margin_left_px", 104)), 104)),
        scene_margin_right_px=max(76, to_int(params.get("scene_margin_right_px", group_default(render_defaults, "scene_margin_right_px", 104)), 104)),
        scene_margin_top_px=max(70, to_int(params.get("scene_margin_top_px", group_default(render_defaults, "scene_margin_top_px", 90)), 90)),
        scene_margin_bottom_px=max(70, to_int(params.get("scene_margin_bottom_px", group_default(render_defaults, "scene_margin_bottom_px", 90)), 90)),
        wall_stroke_width_px=max(2, scale_puzzle_px(wall_width, unit_scale, min_px=2)),
        wall_stroke_width_min_px=max(2, int(wall_min)),
        wall_stroke_width_max_px=max(2, int(wall_max)),
        outer_wall_stroke_width_px=max(3, scale_puzzle_px(outer_wall_width, unit_scale, min_px=3)),
        outer_wall_stroke_width_min_px=max(3, int(outer_wall_min)),
        outer_wall_stroke_width_max_px=max(3, int(outer_wall_max)),
        exit_marker_radius_px=max(13, scale_puzzle_px(to_int(params.get("exit_marker_radius_px", group_default(render_defaults, "exit_marker_radius_px", 27)), 27), unit_scale, min_px=13)),
        exit_marker_shape=string_option(
            params,
            render_defaults,
            "exit_marker_shape",
            "circle",
            seed=int(instance_seed),
            allowed=("circle", "square", "tab"),
        ),
        exit_label_font_size_px=max(14, scale_puzzle_px(to_int(params.get("exit_label_font_size_px", group_default(render_defaults, "exit_label_font_size_px", 28)), 28), unit_scale, min_px=14)),
        start_font_size_px=max(11, scale_puzzle_px(to_int(params.get("start_font_size_px", group_default(render_defaults, "start_font_size_px", 20)), 20), unit_scale, min_px=11)),
        panel_fill_rgb=rgb_option(params, render_defaults, "panel_fill_rgb", (248, 249, 252), seed=int(instance_seed)),
        floor_fill_rgb=rgb_option(params, render_defaults, "floor_fill_rgb", (255, 255, 255), seed=int(instance_seed)),
        wall_color_rgb=rgb_option(params, render_defaults, "wall_color_rgb", (44, 52, 66), seed=int(instance_seed)),
        border_color_rgb=rgb_option(params, render_defaults, "border_color_rgb", (96, 105, 118), seed=int(instance_seed)),
        text_color_rgb=rgb_option(params, render_defaults, "text_color_rgb", (24, 29, 36), seed=int(instance_seed)),
        text_stroke_rgb=rgb_option(params, render_defaults, "text_stroke_rgb", (255, 255, 255), seed=int(instance_seed)),
        start_fill_rgb=rgb_option(params, render_defaults, "start_fill_rgb", (213, 239, 222), seed=int(instance_seed)),
        start_outline_rgb=rgb_option(params, render_defaults, "start_outline_rgb", (44, 122, 84), seed=int(instance_seed)),
        exit_outline_rgb=rgb_option(params, render_defaults, "exit_outline_rgb", (42, 48, 58), seed=int(instance_seed)),
        exit_palette=rgb_palette(
            params,
            render_defaults,
            "exit_palette",
            ((242, 201, 80), (91, 158, 217), (224, 116, 92), (119, 184, 129), (165, 132, 211), (229, 151, 185), (95, 183, 178), (198, 145, 83)),
        ),
        subtle_grid_rgb=rgb_option(params, render_defaults, "subtle_grid_rgb", (229, 232, 238), seed=int(instance_seed)),
        unit_size_scale=float(unit_scale),
        unit_size_jitter=dict(unit_meta),
    )
def sample_maze_font(*, instance_seed: int, params: Mapping[str, Any], render_defaults: Mapping[str, Any]) -> str:
    """Sample one role-aware font family for all visible maze text."""

    return sample_font_family(
        role="readout",
        instance_seed=int(instance_seed),
        namespace=f"{SCENE_NAMESPACE}.label_font",
        params={**dict(render_defaults), **dict(params)},
    )
def font_trace_record(font_family: str) -> Dict[str, Any]:
    """Build trace metadata for the sampled maze label font."""

    return {
        **get_font_family_record(str(font_family)).to_trace(),
        "source": "global_font_pool",
        "font_asset_version": font_asset_version(),
        "selection_scope": "maze_start_and_exit_labels",
        "include_tags": [],
        "exclude_tags": [],
    }
def maze_style_trace(render_params: MazeExitRenderParams) -> Dict[str, Any]:
    """Record resolved non-semantic maze style parameters."""

    return {
        "panel_fill_rgb": [int(value) for value in render_params.panel_fill_rgb],
        "floor_fill_rgb": [int(value) for value in render_params.floor_fill_rgb],
        "wall_color_rgb": [int(value) for value in render_params.wall_color_rgb],
        "border_color_rgb": [int(value) for value in render_params.border_color_rgb],
        "text_color_rgb": [int(value) for value in render_params.text_color_rgb],
        "text_stroke_rgb": [int(value) for value in render_params.text_stroke_rgb],
        "start_fill_rgb": [int(value) for value in render_params.start_fill_rgb],
        "start_outline_rgb": [int(value) for value in render_params.start_outline_rgb],
        "exit_outline_rgb": [int(value) for value in render_params.exit_outline_rgb],
        "exit_palette_rgb": [[int(value) for value in color] for color in render_params.exit_palette],
        "subtle_grid_rgb": [int(value) for value in render_params.subtle_grid_rgb],
        "wall_stroke_width_px": int(render_params.wall_stroke_width_px),
        "outer_wall_stroke_width_px": int(render_params.outer_wall_stroke_width_px),
        "exit_marker_radius_px": int(render_params.exit_marker_radius_px),
        "exit_marker_shape": str(render_params.exit_marker_shape),
    }


__all__ = [
    "font_trace_record",
    "maze_style_trace",
    "normalize_rgb",
    "resolve_render_params",
    "sample_maze_font",
    "to_int",
]
