"""Output metadata helpers for mirror-grid icon scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, Tuple

from .styles import mirror_grid_style_trace


def mirror_grid_render_spec(
    *,
    common_ids: Mapping[str, Any],
    render_params: Mapping[str, Any],
    sampled_palette_rgb: Sequence[Tuple[int, int, int]],
    panel_geometry: Mapping[str, Any],
) -> dict[str, Any]:
    """Return shared render-spec metadata for one mirror-grid image."""

    return {
        **dict(common_ids),
        "canvas_size": [int(render_params["canvas_width"]), int(render_params["canvas_height"])],
        "coord_space": "pixel",
        "panel_geometry": dict(panel_geometry),
        "style": mirror_grid_style_trace(
            render_params=render_params,
            sampled_palette_rgb=tuple(tuple(int(channel) for channel in color) for color in sampled_palette_rgb),
        ),
    }


__all__ = ["mirror_grid_render_spec"]
