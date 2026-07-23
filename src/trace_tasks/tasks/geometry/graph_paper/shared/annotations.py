"""Annotation projection helpers for graph-paper objectives."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.geometry.shared.annotation_values import (
    bbox_set_annotation_artifacts,
    keyed_point_annotation_artifacts,
    point_set_annotation_artifacts,
)

from .state import BBox, Point


def _round_point(point: Sequence[float]) -> list[float]:
    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def _round_box(box: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in box]


def scalar_bbox_artifacts(box: BBox) -> tuple[list[float], dict[str, Any]]:
    """Build scalar bbox annotation value and projection payload."""

    value = _round_box(box)
    center = [
        round((value[0] + value[2]) / 2.0, 3),
        round((value[1] + value[3]) / 2.0, 3),
    ]
    return value, {
        "type": "bbox",
        "bbox": list(value),
        "pixel_bbox": list(value),
        "point": center,
        "pixel_point": center,
    }


def scalar_point_artifacts(point: Point) -> tuple[list[float], dict[str, Any]]:
    """Build scalar point annotation value and projection payload."""

    value = _round_point(point)
    return value, {
        "type": "point",
        "point": list(value),
        "pixel_point": list(value),
    }


def scalar_segment_artifacts(
    start: Point, end: Point
) -> tuple[list[list[float]], dict[str, Any]]:
    """Build scalar segment annotation value and projection payload."""

    value = [_round_point(start), _round_point(end)]
    return value, {
        "type": "segment",
        "segment": list(value),
        "pixel_segment": list(value),
    }


def point_set_artifacts(
    points: Sequence[Point],
) -> tuple[list[list[float]], dict[str, Any]]:
    """Build unordered point-set annotation artifacts."""

    artifacts = point_set_annotation_artifacts(points)
    return list(artifacts.value), dict(artifacts.projected_annotation)


def point_map_artifacts(
    points: Mapping[str, Point],
) -> tuple[dict[str, list[float]], dict[str, Any]]:
    """Build keyed point-map annotation artifacts."""

    artifacts = keyed_point_annotation_artifacts(points)
    return dict(artifacts.value), dict(artifacts.projected_annotation)


def bbox_set_artifacts(
    boxes: Sequence[BBox],
) -> tuple[list[list[float]], dict[str, Any]]:
    """Build unordered bbox-set annotation artifacts."""

    artifacts = bbox_set_annotation_artifacts(boxes)
    return list(artifacts.value), dict(artifacts.projected_annotation)
