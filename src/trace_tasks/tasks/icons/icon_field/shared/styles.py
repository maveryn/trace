"""Style and render-parameter helpers for icon-field scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Tuple

from ....shared.config_defaults import group_default
from ...shared.icon_style import icon_palette_meets_distance_constraints, sample_icon_palette
from ...shared.icon_task_rendering import icon_render_style_trace, resolve_icon_render_params

from .defaults import IconFieldDefaults


def resolve_icon_field_render_params(
    params: Mapping[str, Any],
    *,
    render_defaults: Mapping[str, Any],
    defaults: IconFieldDefaults,
    instance_seed: int,
) -> Dict[str, Any]:
    """Resolve render params for an icon-field scene."""

    render_params = resolve_icon_render_params(
        params=params,
        render_defaults=render_defaults,
        fallback_defaults=defaults,
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
    render_params["placement_mode"] = str(
        params.get("placement_mode", group_default(render_defaults, "placement_mode", "scatter"))
    )
    return render_params


def sample_icon_field_palette(rng, render_params: Mapping[str, Any]) -> Tuple[Tuple[int, int, int], ...]:
    """Sample a palette separated from the panel chrome colors."""

    palette_size = int(rng.randint(int(render_params["palette_size_min"]), int(render_params["palette_size_max"])))
    anchor_colors = (
        tuple(int(v) for v in render_params["background_color_rgb"]),
        tuple(int(v) for v in render_params["panel_fill_rgb"]),
        tuple(int(v) for v in render_params["panel_border_rgb"]),
        tuple(int(v) for v in render_params["header_text_rgb"]),
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
        raise ValueError("sampled icon-field palette did not satisfy distance constraints")
    return palette


def icon_field_style_trace(
    *,
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Tuple[Tuple[int, int, int], ...],
) -> Dict[str, Any]:
    """Return trace metadata for rendered icon-field style choices."""

    style = icon_render_style_trace(
        render_params=render_params,
        sampled_palette_rgb=sampled_palette_rgb,
    )
    style["rotation_candidates_degrees"] = [
        int(value) for value in render_params["rotation_candidates_degrees"]
    ]
    style["placement_mode"] = str(render_params.get("placement_mode", "scatter"))
    return style


__all__ = [
    "icon_field_style_trace",
    "resolve_icon_field_render_params",
    "sample_icon_field_palette",
]
