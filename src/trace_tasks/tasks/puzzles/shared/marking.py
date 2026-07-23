"""Puzzles-domain semantic marker wrappers."""

from __future__ import annotations

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

__all__ = [
    "MARKER_LEGIBILITY_POLICY_VERSION",
    "SEMANTIC_MARKER_MIN_CONTRAST_RATIO",
    "SEMANTIC_MARKER_MIN_LAB_DISTANCE",
    "SemanticMarkerStyle",
    "draw_semantic_bbox_marker",
    "draw_semantic_ellipse_marker",
    "draw_semantic_line_marker",
    "draw_semantic_polygon_marker",
    "resolve_semantic_marker_style",
]
