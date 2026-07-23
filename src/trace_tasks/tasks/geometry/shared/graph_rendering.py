"""Shared graph-paper rendering helpers for geometry tasks.

This module centralizes graph-style normalization, forced graph-paper
background selection, pixel-to-graph coordinate projection, and deterministic
render scaling helpers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping

from ....core.visual.background import coerce_grid_style_spec, compute_grid_axis_origin
from ...shared.geometry_primitives import Point

FALLBACK_GRAPH_STYLE: Dict[str, Any] = {
    "kind": "grid",
    "base_color": [255, 255, 255],
    "line_color": [224, 229, 236],
    "spacing": 24,
    "outer_margin_px": 16,
    "line_width": 1,
    "major_every": 0,
    "major_line_color": [202, 209, 220],
    "major_line_width": 1,
    "axis_enabled": True,
    "axis_color": [118, 128, 146],
    "axis_line_width": 2,
    "axis_arrows_enabled": True,
    "axis_arrow_size": 10,
    "center_point_enabled": True,
    "center_point_color": [102, 112, 130],
    "center_point_radius": 2,
    "color_variation_enabled": True,
    "base_color_jitter": [0, 0],
    "line_color_jitter": [-5, 5],
    "major_line_darken_range": [6, 14],
    "axis_darken_range": [32, 48],
    "center_point_darken_extra_range": [8, 14],
    "origin_label_darken_extra_range": [6, 12],
    "axis_scale_labels_enabled": True,
    "axis_scale_label_max_abs": 0,
    "origin_label_enabled": False,
    "origin_label_text": "0",
    "origin_label_color": [88, 98, 116],
    "supersample_scale": 1,
    "scene_supersample_scale": 3,
}


def _coerce_graph_style(
    spec: Mapping[str, Any],
    *,
    fallback_style: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Normalize one graph-style config to deterministic, valid values."""
    fallback = dict(fallback_style or FALLBACK_GRAPH_STYLE)
    style = coerce_grid_style_spec(spec, fallback_style=fallback)
    scene_supersample_value = fallback["scene_supersample_scale"]
    if isinstance(spec, Mapping) and "scene_supersample_scale" in spec:
        scene_supersample_value = spec["scene_supersample_scale"]
    style["scene_supersample_scale"] = max(
        1,
        min(4, int(scene_supersample_value)),
    )
    if isinstance(spec, Mapping):
        for key in ("style_variants", "style_variant_weights"):
            if key in spec and isinstance(spec.get(key), Mapping):
                style[key] = dict(spec[key])
    return style


def resolve_graph_style_from_params(
    params: Mapping[str, Any],
    *,
    default_background_config: Mapping[str, Any],
    fallback_style: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Resolve graph-paper style from defaults and optional task overrides."""
    fallback = dict(fallback_style or FALLBACK_GRAPH_STYLE)
    styles = default_background_config.get("styles", {}) if isinstance(default_background_config, Mapping) else {}
    base_spec = styles.get("graph_paper", {}) if isinstance(styles, Mapping) else {}
    style = _coerce_graph_style(base_spec if isinstance(base_spec, Mapping) else {}, fallback_style=fallback)

    visual = params.get("visual", {})
    if isinstance(visual, Mapping):
        background = visual.get("background", {})
        if isinstance(background, Mapping):
            user_styles = background.get("styles", {})
            if isinstance(user_styles, Mapping):
                graph_spec = user_styles.get("graph_paper")
                if "graph_paper" in user_styles and isinstance(graph_spec, Mapping):
                    style = _coerce_graph_style(graph_spec, fallback_style=fallback)
    return style


def enforce_graph_paper_background(params: Mapping[str, Any], *, graph_style: Mapping[str, Any]) -> Dict[str, Any]:
    """Return params with graph-paper background forced on for deterministic layouts."""
    forced: Dict[str, Any] = dict(params)
    visual = forced.get("visual", {})
    visual_dict = dict(visual) if isinstance(visual, Mapping) else {}
    background = visual_dict.get("background", {})
    background_dict = dict(background) if isinstance(background, Mapping) else {}
    background_dict.update(
        {
            "enabled": True,
            "style_name": "graph_paper",
            "styles": {"graph_paper": dict(graph_style)},
            "weights": {"graph_paper": 1.0},
        }
    )
    visual_dict["background"] = background_dict
    forced["visual"] = visual_dict
    return forced


def scaled_graph_style_for_scene(graph_style: Mapping[str, Any], *, scene_scale: int) -> Dict[str, Any]:
    """Scale graph-style line geometry into high-resolution scene render space."""
    scaled = dict(graph_style)
    scale = max(1, int(scene_scale))
    for key in (
        "spacing",
        "outer_margin_px",
        "line_width",
        "major_line_width",
        "axis_line_width",
        "axis_arrow_size",
        "center_point_radius",
    ):
        if key in scaled:
            min_value = 0 if str(key) == "outer_margin_px" else 1
            scaled[key] = max(int(min_value), int(scaled[key]) * scale)
    origin_pixel = scaled.get("origin_pixel")
    if isinstance(origin_pixel, (list, tuple)) and len(origin_pixel) >= 2:
        scaled["origin_pixel"] = [
            max(0, int(origin_pixel[0]) * int(scale)),
            max(0, int(origin_pixel[1]) * int(scale)),
        ]
    scaled["supersample_scale"] = 1
    return scaled


def scale_point(point: Point, scale: int) -> Point:
    """Scale one `(x, y)` point from canonical pixels into render pixels."""
    return (float(point[0]) * float(scale), float(point[1]) * float(scale))


def graph_units_to_pixel(
    point: tuple[int, int],
    *,
    origin: Point | None = None,
    graph_origin: Point | None = None,
    spacing: int,
) -> Point:
    """Project one integer graph-unit point into canonical pixel coordinates."""

    resolved_origin = origin if origin is not None else graph_origin
    if resolved_origin is None:
        raise ValueError("graph_units_to_pixel requires origin or graph_origin")
    return (
        float(resolved_origin[0]) + (float(point[0]) * float(spacing)),
        float(resolved_origin[1]) - (float(point[1]) * float(spacing)),
    )


def pixel_point_to_graph_units(
    point: Point,
    *,
    origin: Point,
    spacing: int,
    tol: float = 1e-6,
) -> List[int]:
    """Project one pixel-space point to centered integer graph coordinates."""
    spacing_px = max(1, int(spacing))
    gx_raw = (float(point[0]) - float(origin[0])) / float(spacing_px)
    gy_raw = (float(origin[1]) - float(point[1])) / float(spacing_px)
    gx = int(round(gx_raw))
    gy = int(round(gy_raw))
    if abs(gx_raw - float(gx)) > float(tol) or abs(gy_raw - float(gy)) > float(tol):
        raise ValueError("point is not aligned to graph-paper lattice for graph-unit annotation")
    return [int(gx), int(gy)]


def build_graph_coordinate_frame(
    *,
    canvas_size: int,
    spacing: int,
    target_cells: int,
    outer_margin_px: int = 0,
    origin_fraction_x: float = 0.5,
    origin_fraction_y: float = 0.5,
) -> Dict[str, Any]:
    """Build graph-unit coordinate-frame metadata for trace payloads."""
    spacing_px = max(1, int(spacing))
    size_px = int(canvas_size)
    inset_px = max(0, int(outer_margin_px))
    origin = compute_grid_axis_origin(
        canvas_size=size_px,
        spacing=spacing_px,
        inset=int(inset_px),
        x_fraction=float(origin_fraction_x),
        y_fraction=float(origin_fraction_y),
    )
    left = max(0, int(inset_px))
    top = max(0, int(inset_px))
    right = max(int(left), int(size_px) - 1 - int(inset_px))
    bottom = max(int(top), int(size_px) - 1 - int(inset_px))
    x_neg_steps = max(0, int((int(origin[0]) - int(left)) // int(spacing_px)))
    x_pos_steps = max(0, int((int(right) - int(origin[0])) // int(spacing_px)))
    y_neg_steps = max(0, int((int(origin[1]) - int(top)) // int(spacing_px)))
    y_pos_steps = max(0, int((int(bottom) - int(origin[1])) // int(spacing_px)))
    full_cells_x = max(0, int(x_neg_steps + x_pos_steps))
    full_cells_y = max(0, int(y_neg_steps + y_pos_steps))
    partial_cells = bool(
        ((int(origin[0]) - int(left)) % int(spacing_px))
        or ((int(right) - int(origin[0])) % int(spacing_px))
        or ((int(origin[1]) - int(top)) % int(spacing_px))
        or ((int(bottom) - int(origin[1])) % int(spacing_px))
    )
    return {
        "coord_space": "graph_unit",
        "origin_pixel": [int(origin[0]), int(origin[1])],
        "origin_fraction_x": float(origin_fraction_x),
        "origin_fraction_y": float(origin_fraction_y),
        "outer_margin_px": int(inset_px),
        "spacing_px": int(spacing_px),
        "target_cells_x": int(target_cells),
        "target_cells_y": int(target_cells),
        "full_cells_x": int(full_cells_x),
        "full_cells_y": int(full_cells_y),
        "partial_edge_cells": bool(partial_cells),
        "x_positive": "right",
        "y_positive": "up",
    }


def graph_paper_grid_from_frame(frame: Mapping[str, Any]) -> Dict[str, Any]:
    """Project graph-paper grid fields from one graph coordinate-frame mapping."""
    out = {
        "target_cells_x": int(frame["target_cells_x"]),
        "target_cells_y": int(frame["target_cells_y"]),
        "full_cells_x": int(frame["full_cells_x"]),
        "full_cells_y": int(frame["full_cells_y"]),
        "partial_edge_cells": bool(frame["partial_edge_cells"]),
        "spacing_px": int(frame["spacing_px"]),
    }
    for key in (
        "graph_panel_bbox_px",
        "graph_content_bbox_px",
        "graph_origin_px",
        "graph_spacing_px",
        "scene_bbox_px",
        "layout_placement",
    ):
        if key in frame:
            value = frame[key]
            out[str(key)] = dict(value) if isinstance(value, Mapping) else value
    return out
