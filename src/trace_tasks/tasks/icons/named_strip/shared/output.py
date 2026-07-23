"""Trace output helpers for named-strip icon scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ...shared.icon_task_rendering import icon_render_style_trace

from .defaults import SCENE_ID


def named_strip_render_spec(
    *,
    render_params: Mapping[str, Any],
    panel_geometry: Mapping[str, Any],
    sampled_palette_rgb: Sequence[Sequence[int]],
    cell_box_width_px: int,
    cell_box_height_px: int,
    fill_style_support: Sequence[str],
) -> dict[str, Any]:
    """Return render metadata for one named-strip scene."""

    return {
        "canvas_size": list(panel_geometry["canvas_size"]),
        "coord_space": "pixel",
        "scene_id": SCENE_ID,
        "panel_geometry": dict(panel_geometry),
        "style": {
            **icon_render_style_trace(render_params=render_params, sampled_palette_rgb=sampled_palette_rgb),
            "cell_box_width_px": int(cell_box_width_px),
            "cell_box_height_px": int(cell_box_height_px),
            "cell_padding_px": int(render_params["cell_padding_px"]),
            "cell_icon_padding_px": int(render_params["cell_icon_padding_px"]),
            "cell_corner_radius_px": int(render_params["cell_corner_radius_px"]),
            "cell_border_rgb": list(render_params["cell_border_rgb"]),
            "named_icon_fill_style_support": [str(value) for value in fill_style_support],
        },
    }


def named_strip_render_map(*, icons: Sequence[Any], selected_instance_ids: Sequence[str]) -> dict[str, Any]:
    """Return bbox render-map entries for one named-strip scene."""

    return {
        "image_id": "img0",
        "object_bboxes_px": {
            str(icon.instance_id): [int(value) for value in icon.bbox_xyxy]
            for icon in icons
        },
        "selected_run_instance_ids": [str(value) for value in selected_instance_ids],
    }


__all__ = ["named_strip_render_map", "named_strip_render_spec"]
