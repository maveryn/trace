"""Default resolution helpers for lever-balance rendering."""

from __future__ import annotations

from typing import Any, Dict, Mapping

from trace_tasks.tasks.shared.config_defaults import group_default
from trace_tasks.tasks.shared.render_variation import resolve_render_int

from .state import LeverTaskDefaults


def resolve_render_defaults(
    *,
    params: Mapping[str, Any],
    rendering_defaults: Mapping[str, Any],
    defaults: LeverTaskDefaults,
    instance_seed: int,
    namespace: str,
) -> Dict[str, Any]:
    """Resolve numeric render defaults before layout projection."""

    resolved = {
        key: (
            params.get(key, group_default(rendering_defaults, key, getattr(defaults, key)))
            if key == "distance_support"
            else resolve_render_int(
                params,
                rendering_defaults,
                key,
                int(getattr(defaults, key)),
                instance_seed=int(instance_seed),
                namespace=str(namespace),
            )
        )
        for key in (
            "canvas_width",
            "canvas_height",
            "beam_width_px",
            "beam_height_px",
            "beam_corner_radius_px",
            "beam_center_y_px",
            "fulcrum_width_px",
            "fulcrum_height_px",
            "fulcrum_offset_px",
            "slot_spacing_px",
            "distance_support",
            "weight_box_width_px",
            "weight_box_height_px",
            "weight_box_gap_px",
            "weight_font_size_px",
            "distance_font_size_px",
            "label_stroke_width_px",
            "texture_line_width_px",
            "texture_spacing_px",
        )
    }
    resolved["instance_seed"] = int(instance_seed)
    return dict(resolved)
