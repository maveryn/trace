"""Render parameter helpers for Venn-field icon scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from ....shared.config_defaults import group_default
from ...shared.icon_task_rendering import (
    icon_render_style_trace,
    resolve_icon_render_params,
    resolve_icon_rgb_param,
)

from .defaults import VennFieldDefaults


def resolve_venn_render_params(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback_defaults: VennFieldDefaults,
    instance_seed: int,
) -> Dict[str, Any]:
    """Resolve shared icon rendering plus Venn circle chrome."""

    render_params = resolve_icon_render_params(
        params=params,
        render_defaults=render_defaults,
        fallback_defaults=fallback_defaults,
        instance_seed=int(instance_seed),
    )
    for key, fallback in (
        ("venn_boundary_margin_px", fallback_defaults.venn_boundary_margin_px),
        ("venn_fill_alpha", fallback_defaults.venn_fill_alpha),
        ("venn_outline_width_px", fallback_defaults.venn_outline_width_px),
    ):
        render_params[str(key)] = int(
            params.get(str(key), group_default(render_defaults, str(key), fallback))
        )
    for key, fallback in (
        ("venn_left_fill_rgb", fallback_defaults.venn_left_fill_rgb),
        ("venn_right_fill_rgb", fallback_defaults.venn_right_fill_rgb),
        ("venn_left_outline_rgb", fallback_defaults.venn_left_outline_rgb),
        ("venn_right_outline_rgb", fallback_defaults.venn_right_outline_rgb),
    ):
        render_params[str(key)] = resolve_icon_rgb_param(
            params=params,
            render_defaults=render_defaults,
            key=str(key),
            fallback=fallback,
            instance_seed=int(instance_seed),
        )
    return render_params


def venn_style_trace(
    *,
    render_params: Mapping[str, Any],
    sampled_palette_rgb,
) -> Dict[str, Any]:
    """Serialize Venn-field style metadata."""

    return {
        **icon_render_style_trace(
            render_params=render_params, sampled_palette_rgb=sampled_palette_rgb
        ),
        "venn_left_fill_rgb": [
            int(value) for value in render_params["venn_left_fill_rgb"]
        ],
        "venn_right_fill_rgb": [
            int(value) for value in render_params["venn_right_fill_rgb"]
        ],
        "venn_left_outline_rgb": [
            int(value) for value in render_params["venn_left_outline_rgb"]
        ],
        "venn_right_outline_rgb": [
            int(value) for value in render_params["venn_right_outline_rgb"]
        ],
        "venn_fill_alpha": int(render_params["venn_fill_alpha"]),
        "venn_outline_width_px": int(render_params["venn_outline_width_px"]),
        "venn_boundary_margin_px": int(render_params["venn_boundary_margin_px"]),
    }


__all__ = ["resolve_venn_render_params", "venn_style_trace"]
