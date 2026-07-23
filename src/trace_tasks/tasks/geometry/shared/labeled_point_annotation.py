"""Geometry wrappers for graph-derived pixel-point annotation helpers."""

from __future__ import annotations

from typing import Any, Dict

from ...shared.graph_point_annotation import (
    empty_graph_point_set_annotation_artifacts as _base_empty_graph_point_set_annotation_artifacts,
    graph_point_annotation_artifacts as _base_graph_point_annotation_artifacts,
    graph_point_set_annotation_artifacts as _base_graph_point_set_annotation_artifacts,
    labeled_grid_point_annotation_artifacts,
)


def _as_pixel_point_set_artifacts(artifacts: Dict[str, Any]) -> Dict[str, Any]:
    """Expose graph-derived Geometry annotation as public pixel point sets."""

    projected = dict(artifacts.get("projected_annotation") or {})
    pixel_points = projected.get("pixel_point_set")
    if pixel_points is None:
        pixel_path = projected.get("pixel_point_sequence")
        pixel_points = list(pixel_path) if isinstance(pixel_path, list) else []
    point_set = [[float(point[0]), float(point[1])] for point in pixel_points if isinstance(point, (list, tuple)) and len(point) == 2]

    witness_symbolic = dict(artifacts.get("witness_symbolic") or {})

    pixel_projected: Dict[str, Any] = {
        "type": "point_set",
        "point_set": [list(point) for point in point_set],
        "pixel_point_set": [list(point) for point in point_set],
    }
    if "pixel_point_map" in projected:
        pixel_projected["pixel_point_map"] = dict(projected["pixel_point_map"])
    if "pixel_point_sequence" in projected:
        pixel_projected["pixel_point_sequence"] = [list(point) for point in projected["pixel_point_sequence"]]

    return {
        "annotation_type": "point_set",
        "annotation_value": [list(point) for point in point_set],
        "required_labels": list(artifacts.get("required_labels", [])),
        "witness_symbolic": witness_symbolic,
        "projected_annotation": pixel_projected,
    }


def graph_point_set_annotation_artifacts(**kwargs: Any) -> Dict[str, Any]:
    """Build Geometry public annotation as pixel points for a graph-point set witness."""

    return _as_pixel_point_set_artifacts(_base_graph_point_set_annotation_artifacts(**kwargs))


def empty_graph_point_set_annotation_artifacts(*, witness_type: str) -> Dict[str, Any]:
    """Build an empty Geometry pixel point-set annotation payload."""

    return _as_pixel_point_set_artifacts(
        _base_empty_graph_point_set_annotation_artifacts(witness_type=str(witness_type))
    )


def graph_point_annotation_artifacts(**kwargs: Any) -> Dict[str, Any]:
    """Build Geometry public annotation as a singleton pixel point set."""

    return _as_pixel_point_set_artifacts(_base_graph_point_annotation_artifacts(**kwargs))


__all__ = [
    "empty_graph_point_set_annotation_artifacts",
    "graph_point_annotation_artifacts",
    "graph_point_set_annotation_artifacts",
    "labeled_grid_point_annotation_artifacts",
]
