"""Bounded graph-paper panel layout helpers for geometry scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple


BBox = Tuple[int, int, int, int]


@dataclass(frozen=True)
class GraphPanelLayout:
    """Resolved graph-paper panel geometry in canonical pixel coordinates."""

    canvas_width: int
    canvas_height: int
    graph_cells: int
    graph_spacing: int
    panel_bbox_px: BBox
    content_bbox_px: BBox
    graph_origin_px: Tuple[int, int]
    local_graph_origin_px: Tuple[int, int]
    panel_frame_width_px: int
    panel_padding_px: int
    layout_placement: Dict[str, Any]

    @property
    def panel_size_px(self) -> Tuple[int, int]:
        return (
            int(self.panel_bbox_px[2]) - int(self.panel_bbox_px[0]),
            int(self.panel_bbox_px[3]) - int(self.panel_bbox_px[1]),
        )

    @property
    def content_size_px(self) -> Tuple[int, int]:
        return (
            int(self.content_bbox_px[2]) - int(self.content_bbox_px[0]),
            int(self.content_bbox_px[3]) - int(self.content_bbox_px[1]),
        )

    def to_metadata(self) -> Dict[str, Any]:
        """Return trace-ready panel placement metadata."""

        return {
            "layout_placement": dict(self.layout_placement),
            "graph_panel_bbox_px": [int(value) for value in self.panel_bbox_px],
            "graph_content_bbox_px": [int(value) for value in self.content_bbox_px],
            "graph_origin_px": [int(self.graph_origin_px[0]), int(self.graph_origin_px[1])],
            "graph_spacing_px": int(self.graph_spacing),
            "scene_bbox_px": [int(value) for value in self.panel_bbox_px],
        }


def _coerce_bool(value: Any, fallback: bool) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
    return bool(fallback)


def _coerce_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except Exception:
        return int(fallback)


def _coerce_fraction(value: Any, fallback: float) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(fallback)
    return max(0.0, min(1.0, float(parsed)))


def _defaulted(
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    key: str,
    fallback: Any,
) -> Any:
    if str(key) in params and params.get(str(key)) is not None:
        return params.get(str(key))
    return render_defaults.get(str(key), fallback)


def resolve_graph_panel_layout(
    *,
    canvas_width: int,
    canvas_height: int,
    graph_cells: int,
    params: Mapping[str, Any],
    render_defaults: Mapping[str, Any],
    min_spacing_px: int = 4,
    origin_fraction_x: float = 0.5,
    origin_fraction_y: float = 0.5,
) -> GraphPanelLayout:
    """Resolve a bounded graph-paper panel inside the final canvas.

    The graph-paper content is sized so its lattice boundaries align with the
    panel content bbox. The panel nearly fills the canvas by default; this keeps
    source sampling helpers, which still constrain against the full canvas,
    inside the panel because the free area is smaller than one graph cell.
    """

    width = max(64, int(canvas_width))
    height = max(64, int(canvas_height))
    cells = max(2, int(graph_cells))
    frame_width = max(1, _coerce_int(_defaulted(params, render_defaults, "graph_panel_frame_width_px", 2), 2))
    padding = max(0, _coerce_int(_defaulted(params, render_defaults, "graph_panel_padding_px", 4), 4))
    outer_margin = max(0, _coerce_int(_defaulted(params, render_defaults, "graph_panel_outer_margin_px", 12), 12))
    chrome = int(frame_width) + int(padding)
    available_side = max(8, min(int(width), int(height)) - (2 * int(outer_margin)) - (2 * int(chrome)) - 1)

    explicit_spacing = _defaulted(params, render_defaults, "graph_spacing_px", None)
    if explicit_spacing is not None:
        spacing = max(int(min_spacing_px), _coerce_int(explicit_spacing, int(min_spacing_px)))
        max_spacing = max(int(min_spacing_px), int(available_side) // int(cells))
        spacing = min(int(spacing), int(max_spacing))
    else:
        spacing = max(int(min_spacing_px), int(available_side) // int(cells))
    if int(spacing) < int(min_spacing_px):
        raise ValueError("no feasible graph panel spacing for current canvas and graph_cells")

    content_side = (int(cells) * int(spacing)) + 1
    panel_side = int(content_side) + (2 * int(chrome))
    if int(panel_side) > min(int(width), int(height)):
        raise ValueError("resolved graph panel does not fit inside canvas")

    free_x = max(0, int(width) - int(panel_side))
    free_y = max(0, int(height) - int(panel_side))
    centered_origin_x = int(round(float(free_x) * 0.5))
    centered_origin_y = int(round(float(free_y) * 0.5))
    placement_enabled = _coerce_bool(
        _defaulted(params, render_defaults, "graph_panel_placement_enabled", True),
        True,
    )
    fraction_x = (
        _coerce_fraction(_defaulted(params, render_defaults, "graph_panel_fraction_x", 0.5), 0.5)
        if bool(placement_enabled)
        else 0.5
    )
    fraction_y = (
        _coerce_fraction(_defaulted(params, render_defaults, "graph_panel_fraction_y", 0.5), 0.5)
        if bool(placement_enabled)
        else 0.5
    )
    panel_x = max(0, min(int(free_x), int(round(float(free_x) * float(fraction_x)))))
    panel_y = max(0, min(int(free_y), int(round(float(free_y) * float(fraction_y)))))
    panel_bbox = (
        int(panel_x),
        int(panel_y),
        int(panel_x) + int(panel_side),
        int(panel_y) + int(panel_side),
    )
    content_left = int(panel_x) + int(chrome)
    content_top = int(panel_y) + int(chrome)
    content_bbox = (
        int(content_left),
        int(content_top),
        int(content_left) + int(content_side),
        int(content_top) + int(content_side),
    )

    origin_step_x = max(0, min(int(cells), int(round(float(cells) * float(origin_fraction_x)))))
    origin_step_y = max(0, min(int(cells), int(round(float(cells) * float(origin_fraction_y)))))
    local_origin = (int(origin_step_x) * int(spacing), int(origin_step_y) * int(spacing))
    graph_origin = (
        int(content_left) + int(local_origin[0]),
        int(content_top) + int(local_origin[1]),
    )
    panel_size = (int(panel_side), int(panel_side))
    content_size = (int(content_side), int(content_side))
    layout_placement = {
        "mode": "fractional_free_area",
        "enabled": bool(placement_enabled),
        "canvas_size_px": [int(width), int(height)],
        "panel_size_px": [int(panel_size[0]), int(panel_size[1])],
        "content_size_px": [int(content_size[0]), int(content_size[1])],
        "free_space_px": [int(free_x), int(free_y)],
        "requested_outer_margin_px": int(outer_margin),
        "sampled_fractions": {"x": float(fraction_x), "y": float(fraction_y)},
        "centered_origin_px": [int(centered_origin_x), int(centered_origin_y)],
        "final_origin_px": [int(panel_x), int(panel_y)],
        "dx_from_center_px": int(panel_x) - int(centered_origin_x),
        "dy_from_center_px": int(panel_y) - int(centered_origin_y),
    }
    return GraphPanelLayout(
        canvas_width=int(width),
        canvas_height=int(height),
        graph_cells=int(cells),
        graph_spacing=int(spacing),
        panel_bbox_px=tuple(int(value) for value in panel_bbox),
        content_bbox_px=tuple(int(value) for value in content_bbox),
        graph_origin_px=(int(graph_origin[0]), int(graph_origin[1])),
        local_graph_origin_px=(int(local_origin[0]), int(local_origin[1])),
        panel_frame_width_px=int(frame_width),
        panel_padding_px=int(padding),
        layout_placement=dict(layout_placement),
    )


def graph_coordinate_frame_from_panel_layout(
    layout: GraphPanelLayout,
    *,
    origin_fraction_x: float,
    origin_fraction_y: float,
) -> Dict[str, Any]:
    """Build graph-unit frame metadata from a resolved panel layout."""

    cells = int(layout.graph_cells)
    spacing = int(layout.graph_spacing)
    content_left, content_top, content_right, content_bottom = layout.content_bbox_px
    origin_x, origin_y = layout.graph_origin_px
    x_neg_steps = max(0, int((int(origin_x) - int(content_left)) // int(spacing)))
    x_pos_steps = max(0, int((int(content_right) - 1 - int(origin_x)) // int(spacing)))
    y_neg_steps = max(0, int((int(origin_y) - int(content_top)) // int(spacing)))
    y_pos_steps = max(0, int((int(content_bottom) - 1 - int(origin_y)) // int(spacing)))
    frame = {
        "coord_space": "graph_unit",
        "origin_pixel": [int(origin_x), int(origin_y)],
        "origin_fraction_x": float(origin_fraction_x),
        "origin_fraction_y": float(origin_fraction_y),
        "resolved_origin_fraction_x": float(x_neg_steps) / float(max(1, cells)),
        "resolved_origin_fraction_y": float(y_neg_steps) / float(max(1, cells)),
        "outer_margin_px": 0,
        "spacing_px": int(spacing),
        "target_cells_x": int(cells),
        "target_cells_y": int(cells),
        "full_cells_x": int(x_neg_steps + x_pos_steps),
        "full_cells_y": int(y_neg_steps + y_pos_steps),
        "partial_edge_cells": False,
        "x_positive": "right",
        "y_positive": "up",
        "graph_panel_bbox_px": [int(value) for value in layout.panel_bbox_px],
        "graph_content_bbox_px": [int(value) for value in layout.content_bbox_px],
        "graph_origin_px": [int(origin_x), int(origin_y)],
        "graph_spacing_px": int(spacing),
        "scene_bbox_px": [int(value) for value in layout.panel_bbox_px],
        "layout_placement": dict(layout.layout_placement),
    }
    return frame
