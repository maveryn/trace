"""Annotation projection helpers for style-legend marker witnesses."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .state import Point, RenderedStyleLegend, point


def center_points_for_markers(rendered: RenderedStyleLegend, marker_ids: Sequence[str]) -> list[Point]:
    points: list[Point] = []
    for marker_id in marker_ids:
        bbox = rendered.point_bboxes_px[str(marker_id)]
        points.append(point(0.5 * (float(bbox[0]) + float(bbox[2])), 0.5 * (float(bbox[1]) + float(bbox[3]))))
    return points


def projected_point_payload(value: Sequence[float]) -> dict[str, Any]:
    point_value = [float(value[0]), float(value[1])]
    return {
        "type": "point",
        "point": list(point_value),
        "pixel_point": list(point_value),
    }


def projected_point_set_payload(values: Sequence[Sequence[float]]) -> dict[str, Any]:
    point_values = [[float(point_value[0]), float(point_value[1])] for point_value in values]
    return {
        "type": "point_set",
        "point_set": [list(point_value) for point_value in point_values],
        "pixel_point_set": [list(point_value) for point_value in point_values],
    }
