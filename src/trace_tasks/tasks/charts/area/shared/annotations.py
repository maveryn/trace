"""Annotation payload helpers for the area chart scene."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.area.shared.state import RenderedAreaPanel


def point_set_artifacts(points: Sequence[Sequence[float]]) -> tuple[TypedValue, dict[str, Any], dict[str, Any]]:
    annotation_points = [[round(float(point[0]), 3), round(float(point[1]), 3)] for point in points]
    annotation_gt = TypedValue(type="point_set", value=annotation_points)
    witness_symbolic = {"type": "point_set", "count": len(annotation_points)}
    projected_annotation = {
        "type": "point_set",
        "point_set": annotation_points,
        "pixel_point_set": annotation_points,
    }
    return annotation_gt, witness_symbolic, projected_annotation


def points_for_pairs(
    panel: RenderedAreaPanel,
    pairs: Sequence[tuple[str, str]],
) -> list[list[float]]:
    trace_by_pair: dict[tuple[str, str], Mapping[str, Any]] = {
        (str(trace["series_label"]), str(trace["x_label"])): trace
        for trace in panel.point_traces
    }
    points: list[list[float]] = []
    for series, label in pairs:
        key = (str(series), str(label))
        if key not in trace_by_pair:
            raise KeyError(f"missing rendered area point for {key!r}")
        point = trace_by_pair[key]["mark_center_px"]
        points.append([round(float(point[0]), 3), round(float(point[1]), 3)])
    return points
