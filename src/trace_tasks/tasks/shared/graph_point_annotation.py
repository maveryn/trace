"""Cross-domain helpers for projecting graph-paper points into pixel annotation."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def pixel_point_to_graph_units(
    point: Sequence[float],
    *,
    origin: Sequence[float],
    spacing: int,
    tol: float = 1e-6,
) -> list[int]:
    """Project one pixel-space point to integer graph-paper coordinates."""

    spacing_px = max(1, int(spacing))
    gx_raw = (float(point[0]) - float(origin[0])) / float(spacing_px)
    gy_raw = (float(origin[1]) - float(point[1])) / float(spacing_px)
    gx = int(round(gx_raw))
    gy = int(round(gy_raw))
    if abs(gx_raw - float(gx)) > float(tol) or abs(gy_raw - float(gy)) > float(tol):
        raise ValueError("point is not aligned to graph-paper lattice for graph-unit annotation")
    return [int(gx), int(gy)]


def labeled_grid_point_annotation_artifacts(
    *,
    points_by_label: Mapping[str, Sequence[float]],
    graph_origin: Sequence[float],
    graph_spacing: int,
    witness_type: str,
    ordered_labels: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Build public pixel annotation/projection payloads for labeled graph points."""

    if ordered_labels is None:
        labels = [str(label) for label in points_by_label.keys()]
    else:
        labels = [str(label) for label in ordered_labels]
    if not labels:
        raise ValueError("labeled annotation requires at least one label")

    pixel_map: Dict[str, list[float]] = {}
    pixel_set: list[list[float]] = []
    for label in labels:
        if str(label) not in points_by_label:
            raise ValueError(f"missing point for annotation label: {label}")
        point = points_by_label[str(label)]
        if not isinstance(point, Sequence) or len(point) != 2:
            raise ValueError(f"invalid point for annotation label: {label}")
        pixel_point = [float(point[0]), float(point[1])]
        pixel_point_to_graph_units(
            pixel_point,
            origin=(float(graph_origin[0]), float(graph_origin[1])),
            spacing=int(graph_spacing),
        )
        pixel_map[str(label)] = list(pixel_point)
        pixel_set.append(list(pixel_point))

    return {
        "annotation_type": "point_set",
        "annotation_value": list(pixel_set),
        "required_labels": list(labels),
        "witness_symbolic": {
            "type": str(witness_type),
            "count": len(labels),
        },
        "projected_annotation": {
            "type": "point_set",
            "point_set": list(pixel_set),
            "pixel_point_map": dict(pixel_map),
            "pixel_point_set": list(pixel_set),
            "pixel_point_sequence": list(pixel_set),
        },
    }


def graph_point_set_annotation_artifacts(
    *,
    points_by_label: Mapping[str, Sequence[float]],
    graph_origin: Sequence[float],
    graph_spacing: int,
    witness_type: str,
    ordered_labels: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Build public pixel point-set annotation while preserving projections."""

    labeled = labeled_grid_point_annotation_artifacts(
        points_by_label=points_by_label,
        graph_origin=graph_origin,
        graph_spacing=graph_spacing,
        witness_type=witness_type,
        ordered_labels=ordered_labels,
    )
    projected = dict(labeled["projected_annotation"])
    point_set = projected.get("point_set", [])
    if not isinstance(point_set, list) or not point_set:
        raise ValueError("point_set annotation requires at least one graph-paper point")
    return {
        "annotation_type": "point_set",
        "annotation_value": list(point_set),
        "required_labels": list(labeled.get("required_labels", [])),
        "witness_symbolic": dict(labeled["witness_symbolic"]),
        "projected_annotation": projected,
    }


def empty_graph_point_set_annotation_artifacts(*, witness_type: str) -> Dict[str, Any]:
    """Build one empty pixel point-set annotation payload."""

    return {
        "annotation_type": "point_set",
        "annotation_value": [],
        "required_labels": [],
        "witness_symbolic": {
            "type": str(witness_type),
            "count": 0,
        },
        "projected_annotation": {
            "type": "point_set",
            "point_set": [],
            "pixel_point_set": [],
        },
    }


def graph_point_annotation_artifacts(
    *,
    points_by_label: Mapping[str, Sequence[float]],
    graph_origin: Sequence[float],
    graph_spacing: int,
    witness_type: str,
    ordered_labels: Sequence[str] | None = None,
) -> Dict[str, Any]:
    """Build singleton public pixel point-set annotation while preserving projections."""

    labeled = labeled_grid_point_annotation_artifacts(
        points_by_label=points_by_label,
        graph_origin=graph_origin,
        graph_spacing=graph_spacing,
        witness_type=witness_type,
        ordered_labels=ordered_labels,
    )
    projected = dict(labeled["projected_annotation"])
    point_set = projected.get("point_set", [])
    if not isinstance(point_set, list) or len(point_set) != 1:
        raise ValueError("point_set annotation requires exactly one graph-paper point")
    point = point_set[0]
    if not isinstance(point, list) or len(point) != 2:
        raise ValueError("point_set annotation must contain one [x, y] pixel pair")
    return {
        "annotation_type": "point_set",
        "annotation_value": [list(point)],
        "required_labels": list(labeled.get("required_labels", [])),
        "witness_symbolic": dict(labeled["witness_symbolic"]),
        "projected_annotation": projected,
    }


__all__ = [
    "empty_graph_point_set_annotation_artifacts",
    "graph_point_annotation_artifacts",
    "graph_point_set_annotation_artifacts",
    "labeled_grid_point_annotation_artifacts",
    "pixel_point_to_graph_units",
]
