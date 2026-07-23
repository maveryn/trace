"""Render-parameter resolution for organic-structure notation scenes."""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping

from ....shared.config_defaults import group_default
from ....shared.render_variation import resolve_render_int
from ...shared.common import get_int_param as _get_int
from ...shared.unit_size_jitter import resolve_symbolic_unit_size_scale, scale_symbolic_px

from .state import OrganicRenderParams


def resolve_render_params(
    params: Mapping[str, Any],
    defaults: Mapping[str, Any],
    *,
    instance_seed: int,
    sampling_scope: str,
) -> OrganicRenderParams:
    """Resolve scene-level organic render parameters from config and params."""

    unit_scale, unit_meta = resolve_symbolic_unit_size_scale(
        params,
        defaults,
        instance_seed=int(instance_seed),
        namespace=f"{sampling_scope}.unit_size",
        fallback_min=0.5,
        fallback_max=1.0,
    )
    effective_scale_min = float(
        params.get(
            "organic_effective_unit_size_scale_min",
            group_default(defaults, "organic_effective_unit_size_scale_min", 0.85),
        )
    )
    unit_scale = max(float(unit_scale), float(effective_scale_min))
    unit_meta = dict(unit_meta)
    unit_meta["effective_scale"] = float(unit_scale)
    unit_meta["effective_scale_min"] = float(effective_scale_min)
    canvas_width = int(params.get("canvas_width", group_default(defaults, "canvas_width", 1180)))
    canvas_height = int(params.get("canvas_height", group_default(defaults, "canvas_height", 820)))
    bond_width = resolve_render_int(
        params,
        defaults,
        "organic_bond_width_px",
        4,
        instance_seed=int(instance_seed),
        namespace=f"{sampling_scope}.bond_width",
    )
    render_params = OrganicRenderParams(
        canvas_width=canvas_width,
        canvas_height=canvas_height,
        panel_padding_px=scale_symbolic_px(_get_int(params, defaults, "panel_padding_px", 34), unit_scale, min_px=18),
        panel_corner_radius_px=scale_symbolic_px(_get_int(params, defaults, "panel_corner_radius_px", 18), unit_scale, min_px=8),
        panel_border_width_px=max(1, scale_symbolic_px(_get_int(params, defaults, "panel_border_width_px", 2), unit_scale, min_px=1)),
        bond_width_px=max(2, scale_symbolic_px(bond_width, unit_scale, min_px=2)),
        bond_gap_px=max(6, scale_symbolic_px(_get_int(params, defaults, "organic_bond_gap_px", 10), unit_scale, min_px=6)),
        structure_width_px=scale_symbolic_px(_get_int(params, defaults, "organic_structure_width_px", 860), unit_scale, min_px=620),
        structure_height_px=scale_symbolic_px(_get_int(params, defaults, "organic_structure_height_px", 540), unit_scale, min_px=400),
        unit_size_jitter=dict(unit_meta),
        panel_fill_rgb=(253, 253, 249),
        panel_border_rgb=(80, 86, 92),
        bond_rgb=(28, 30, 34),
        annotation_rgb=(160, 164, 168),
    )
    return replace(
        render_params,
        panel_fill_rgb=(253, 252, 247),
        panel_border_rgb=(88, 88, 88),
        bond_rgb=(24, 25, 27),
        annotation_rgb=(160, 164, 168),
    )


__all__ = ["resolve_render_params"]
