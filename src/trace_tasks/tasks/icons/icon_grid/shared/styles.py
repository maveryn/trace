"""Style and render-parameter helpers for icon-grid scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence, Tuple

from ....shared.config_defaults import group_default
from ...shared.icon_style import icon_palette_meets_distance_constraints, sample_icon_palette
from ...shared.icon_task_rendering import resolve_icon_render_params, resolve_icon_rgb_param

from .defaults import IconGridDefaults


def _resolve_rgb(
    params: Mapping[str, Any],
    key: str,
    fallback: Sequence[int],
    *,
    render_defaults: Mapping[str, Any],
    instance_seed: int,
) -> Tuple[int, int, int]:
    return resolve_icon_rgb_param(
        params=params,
        key=str(key),
        fallback=fallback,
        render_defaults=render_defaults,
        instance_seed=int(instance_seed),
    )


def resolve_icon_grid_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    defaults: IconGridDefaults,
    instance_seed: int,
) -> Dict[str, Any]:
    """Resolve render params for one visible icon grid."""

    render_params = resolve_icon_render_params(
        params=params,
        render_defaults=render_defaults,
        fallback_defaults=defaults,
        instance_seed=int(instance_seed),
    )
    for key in (
        "grid_cell_max_size_px",
        "grid_cell_padding_px",
        "grid_line_width_px",
        "grid_border_width_px",
    ):
        render_params[key] = int(params.get(key, group_default(render_defaults, key, getattr(defaults, key))))
    for key in ("grid_line_rgb", "cell_fill_rgb", "alternate_cell_fill_rgb"):
        render_params[key] = _resolve_rgb(
            params,
            key,
            getattr(defaults, key),
            render_defaults=render_defaults,
            instance_seed=int(instance_seed),
        )
    raw_rotations = params.get(
        "rotation_candidates_degrees",
        group_default(render_defaults, "rotation_candidates_degrees", list(defaults.rotation_candidates_degrees)),
    )
    if not isinstance(raw_rotations, (list, tuple)):
        raise ValueError("rotation_candidates_degrees must be a sequence")
    rotations = tuple(int(value) % 360 for value in raw_rotations)
    if not rotations:
        raise ValueError("rotation_candidates_degrees resolved no values")
    render_params["rotation_candidates_degrees"] = rotations
    return render_params


def sample_icon_grid_palette(rng, render_params: Mapping[str, Any]) -> Tuple[Tuple[int, int, int], ...]:
    """Sample a palette separated from the grid and panel chrome colors."""

    palette_size = int(rng.randint(int(render_params["palette_size_min"]), int(render_params["palette_size_max"])))
    anchor_colors = (
        tuple(int(v) for v in render_params["background_color_rgb"]),
        tuple(int(v) for v in render_params["panel_fill_rgb"]),
        tuple(int(v) for v in render_params["panel_border_rgb"]),
        tuple(int(v) for v in render_params["grid_line_rgb"]),
        tuple(int(v) for v in render_params["cell_fill_rgb"]),
        tuple(int(v) for v in render_params["alternate_cell_fill_rgb"]),
    )
    palette = tuple(
        tuple(int(channel) for channel in color)
        for color in sample_icon_palette(
            rng,
            palette_size=int(palette_size),
            channel_min=int(render_params["color_channel_min"]),
            channel_max=int(render_params["color_channel_max"]),
            anchor_colors=anchor_colors,
            min_color_distance=float(render_params["min_color_distance"]),
            distance_space=str(render_params["color_distance_space"]),
        )
    )
    if not icon_palette_meets_distance_constraints(
        palette=palette,
        anchor_colors=anchor_colors,
        min_color_distance=float(render_params["min_color_distance"]),
        distance_space=str(render_params["color_distance_space"]),
    ):
        raise ValueError("sampled icon-grid palette did not satisfy distance constraints")
    return palette


def icon_grid_style_trace(
    *,
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...],
) -> Dict[str, Any]:
    """Return trace metadata for rendered icon-grid style choices."""

    return {
        "sampled_palette_rgb": [list(color) for color in sampled_palette_rgb],
        "rotation_candidates_degrees": [
            int(value) for value in render_params["rotation_candidates_degrees"]
        ],
        "grid_cell_max_size_px": int(render_params["grid_cell_max_size_px"]),
        "grid_cell_padding_px": int(render_params["grid_cell_padding_px"]),
        "grid_line_width_px": int(render_params["grid_line_width_px"]),
        "grid_border_width_px": int(render_params["grid_border_width_px"]),
        "grid_line_rgb": [int(value) for value in render_params["grid_line_rgb"]],
        "cell_fill_rgb": [int(value) for value in render_params["cell_fill_rgb"]],
        "alternate_cell_fill_rgb": [int(value) for value in render_params["alternate_cell_fill_rgb"]],
    }


__all__ = [
    "icon_grid_style_trace",
    "resolve_icon_grid_render_params",
    "sample_icon_grid_palette",
]
