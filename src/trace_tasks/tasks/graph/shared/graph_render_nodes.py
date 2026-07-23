"""Node glyph and label helpers for node-link graph rendering."""

from __future__ import annotations

import math
from typing import Sequence, Tuple

from PIL import ImageDraw, ImageFont

from ...shared.text_rendering import fit_font_to_box
from .graph_render_types import BBox, GraphRenderParams, Point


def _regular_polygon_points(*, center: Point, radius: int, sides: int, rotation_degrees: float) -> Tuple[Point, ...]:
    """Return polygon vertices for one regular polygon node glyph."""

    cx = float(center[0])
    cy = float(center[1])
    start = math.radians(float(rotation_degrees))
    return tuple(
        (
            int(round(cx + (float(radius) * math.cos(start + ((2.0 * math.pi * index) / float(sides)))))),
            int(round(cy + (float(radius) * math.sin(start + ((2.0 * math.pi * index) / float(sides)))))),
        )
        for index in range(int(sides))
    )


def _draw_node_shape(
    draw: ImageDraw.ImageDraw,
    *,
    center: Point,
    radius: int,
    node_shape_variant: str,
    fill_rgb: Sequence[int],
    outline_rgb: Sequence[int],
    outline_width: int,
) -> BBox:
    """Draw one node glyph and return its bounding box."""

    bbox = (
        int(center[0] - radius),
        int(center[1] - radius),
        int(center[0] + radius),
        int(center[1] + radius),
    )
    shape = str(node_shape_variant)
    fill = tuple(int(value) for value in fill_rgb)
    outline = tuple(int(value) for value in outline_rgb)
    if shape == "rounded_square":
        draw.rounded_rectangle(
            bbox,
            radius=max(4, int(round(float(radius) * 0.35))),
            fill=fill,
            outline=outline,
            width=int(outline_width),
        )
    elif shape == "hexagon":
        draw.polygon(
            _regular_polygon_points(center=center, radius=int(radius), sides=6, rotation_degrees=30.0),
            fill=fill,
            outline=outline,
            width=int(outline_width),
        )
    else:
        draw.ellipse(
            bbox,
            fill=fill,
            outline=outline,
            width=int(outline_width),
        )
    return bbox

def _node_label_box(
    *,
    radius: int,
    node_shape_variant: str,
    outline_width: int,
) -> Tuple[float, float]:
    """Return one conservative text-fit box inside a node glyph."""

    inner_diameter = max(4.0, float((2 * int(radius)) - (2 * max(1, int(outline_width))) - 6))
    shape = str(node_shape_variant)
    if shape == "rounded_square":
        scale = 0.84
    elif shape == "hexagon":
        scale = 0.74
    else:
        scale = 0.68
    side = float(inner_diameter) * float(scale)
    return (float(side), float(side))


def _resolve_node_label_font(
    draw: ImageDraw.ImageDraw,
    *,
    node_labels: Sequence[str],
    render_params: GraphRenderParams,
) -> Tuple[ImageFont.ImageFont, int]:
    """Resolve one fitted node-label font and its stroke width.

    We fit against the longest rendered label so the whole graph uses one
    stable font size while still accommodating multi-character labels such as
    `10` inside compact node glyphs.
    """

    sample_label = max((str(label) for label in node_labels), key=len, default="A")
    max_width, max_height = _node_label_box(
        radius=int(render_params.node_radius_px),
        node_shape_variant=str(render_params.node_shape_variant),
        outline_width=int(render_params.node_border_width_px),
    )
    fitted_font = fit_font_to_box(
        draw,
        text=str(sample_label),
        max_width=float(max_width),
        max_height=float(max_height),
        bold=True,
        font_family=str(render_params.font_family or ""),
        min_size_px=10,
        max_size_px=int(render_params.label_font_size_px),
        fill_ratio=0.94,
    )
    stroke_width = max(1, int(round(float(getattr(fitted_font, "size", render_params.label_font_size_px)) * 0.10)))
    return fitted_font, int(stroke_width)


def _resolve_effective_node_radius_px(
    *,
    node_labels: Sequence[str],
    render_params: GraphRenderParams,
) -> int:
    """Return a node radius large enough for the longest visible label."""

    base_radius = int(render_params.node_radius_px)
    longest = max((len(str(label)) for label in node_labels), default=1)
    if int(longest) <= 2:
        return int(base_radius)
    # Named labels need materially more room than letter/number labels. Keep the
    # growth bounded so dense graphs still fit after layout quality checks.
    return int(max(base_radius, min(38, base_radius + ((int(longest) - 2) * 5))))


__all__ = [
    "_draw_node_shape",
    "_node_label_box",
    "_resolve_effective_node_radius_px",
    "_resolve_node_label_font",
]
