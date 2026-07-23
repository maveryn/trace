"""Games-domain semantic marker wrappers."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from PIL import ImageDraw

from ...shared.marker_legibility import (
    MARKER_LEGIBILITY_POLICY_VERSION,
    SEMANTIC_MARKER_MIN_CONTRAST_RATIO,
    SEMANTIC_MARKER_MIN_LAB_DISTANCE,
    SemanticMarkerStyle,
    draw_semantic_bbox_marker,
    draw_semantic_ellipse_marker,
    draw_semantic_line_marker,
    draw_semantic_polygon_marker,
    resolve_semantic_marker_style,
)


def draw_optional_marker_x(
    draw: ImageDraw.ImageDraw,
    bbox: Sequence[float],
    *,
    enabled: bool = False,
    outer_rgb: Sequence[int] = (30, 30, 30),
    inner_rgb: Sequence[int] = (255, 221, 34),
    width: int = 4,
    inset_fraction: float = 0.26,
    marker_kind: str = "marker_x",
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any] | None:
    """Optionally draw a high-contrast X marker over one bbox."""

    if not bool(enabled):
        return None
    x0, y0, x1, y1 = [float(value) for value in bbox]
    inset = max(1.0, float(inset_fraction) * min(float(x1 - x0), float(y1 - y0)))
    points_a = ((float(x0 + inset), float(y0 + inset)), (float(x1 - inset), float(y1 - inset)))
    points_b = ((float(x1 - inset), float(y0 + inset)), (float(x0 + inset), float(y1 - inset)))
    outer_width = max(1, int(width) + 2)
    inner_width = max(1, int(width))
    outer = tuple(int(value) for value in outer_rgb[:3]) + (255,)
    inner = tuple(int(value) for value in inner_rgb[:3]) + (255,)
    draw.line(points_a, fill=outer, width=outer_width)
    draw.line(points_b, fill=outer, width=outer_width)
    draw.line(points_a, fill=inner, width=inner_width)
    draw.line(points_b, fill=inner, width=inner_width)
    return {
        "marker_kind": str(marker_kind),
        "bbox_px": [float(x0), float(y0), float(x1), float(y1)],
        "width_px": int(inner_width),
        "outer_rgb": [int(value) for value in outer[:3]],
        "inner_rgb": [int(value) for value in inner[:3]],
        **dict(extra_metadata or {}),
    }


__all__ = [
    "MARKER_LEGIBILITY_POLICY_VERSION",
    "SEMANTIC_MARKER_MIN_CONTRAST_RATIO",
    "SEMANTIC_MARKER_MIN_LAB_DISTANCE",
    "SemanticMarkerStyle",
    "draw_optional_marker_x",
    "draw_semantic_bbox_marker",
    "draw_semantic_ellipse_marker",
    "draw_semantic_line_marker",
    "draw_semantic_polygon_marker",
    "resolve_semantic_marker_style",
]
