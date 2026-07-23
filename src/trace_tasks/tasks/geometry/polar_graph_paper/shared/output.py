"""Output helpers for polar graph paper tasks."""

from __future__ import annotations

from typing import Any


def point_annotation_from_render(render_map: dict[str, Any]) -> list[float]:
    point = render_map["point_p"]
    return [float(point[0]), float(point[1])]


def point_map_annotation_from_render(render_map: dict[str, Any], labels: tuple[str, ...]) -> dict[str, list[float]]:
    annotation: dict[str, list[float]] = {}
    for label in labels:
        point = render_map[f"point_{label.lower()}"]
        annotation[str(label)] = [float(point[0]), float(point[1])]
    return annotation


def point_set_annotation_from_render(render_map: dict[str, Any], labels: tuple[str, ...]) -> list[list[float]]:
    points_by_label = render_map["points_by_label"]
    return [
        [float(points_by_label[str(label)][0]), float(points_by_label[str(label)][1])]
        for label in labels
    ]


def projected_point_annotation(annotation: list[float]) -> dict[str, Any]:
    return {"type": "point", "value": list(annotation)}


def projected_point_set_annotation(annotation: list[list[float]]) -> dict[str, Any]:
    point_set = [list(point) for point in annotation]
    return {
        "type": "point_set",
        "value": point_set,
        "point_set": point_set,
        "pixel_point_set": point_set,
    }


def projected_point_map_annotation(annotation: dict[str, list[float]]) -> dict[str, Any]:
    point_map = {str(key): list(value) for key, value in annotation.items()}
    return {
        "type": "point_map",
        "value": point_map,
        "point_map": point_map,
        "pixel_point_map": point_map,
    }


def point_witness(label: str) -> dict[str, Any]:
    return {
        "type": "point",
        "labels": [str(label)],
        "count": 1,
    }


def point_map_witness(labels: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "point_map",
        "labels": [str(label) for label in labels],
        "count": len(tuple(labels)),
    }


def point_set_witness(labels: tuple[str, ...]) -> dict[str, Any]:
    return {
        "type": "point_set",
        "labels": [str(label) for label in labels],
        "count": len(tuple(labels)),
    }
