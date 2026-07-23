"""Style and render-parameter helpers for wallpaper-panel scenes."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from ....shared.config_defaults import group_default
from ...shared.icon_task_rendering import resolve_icon_cell_render_params

from .rendering import wallpaper_canvas_params


def resolve_wallpaper_render_params(
    *,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    fallback_defaults: Any,
    instance_seed: int,
    include_reference_panel: bool,
) -> Dict[str, Any]:
    """Resolve render parameters shared by wallpaper-panel tasks."""

    canvas_params = wallpaper_canvas_params(params)
    render_params = resolve_icon_cell_render_params(
        params=canvas_params,
        render_defaults=render_defaults,
        fallback_defaults=fallback_defaults,
        instance_seed=int(instance_seed),
    )
    if bool(include_reference_panel):
        render_params["reference_panel_width_px"] = int(
            params.get(
                "reference_panel_width_px",
                group_default(render_defaults, "reference_panel_width_px", fallback_defaults.reference_panel_width_px),
            )
        )
    render_params["option_panel_gap_px"] = int(
        params.get(
            "option_panel_gap_px",
            group_default(render_defaults, "option_panel_gap_px", fallback_defaults.option_panel_gap_px),
        )
    )
    render_params["lattice_rows"] = int(
        params.get("lattice_rows", group_default(render_defaults, "lattice_rows", fallback_defaults.lattice_rows))
    )
    render_params["lattice_cols"] = int(
        params.get("lattice_cols", group_default(render_defaults, "lattice_cols", fallback_defaults.lattice_cols))
    )
    return render_params


__all__ = ["resolve_wallpaper_render_params"]
