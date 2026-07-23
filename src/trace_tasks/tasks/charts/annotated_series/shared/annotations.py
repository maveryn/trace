"""Annotation payload helpers for annotated-series chart tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.base import TypedValue
from trace_tasks.tasks.charts.shared.chart_scene_types import RenderedChartScene


def mark_centers_by_label(rendered_scene: RenderedChartScene) -> dict[str, list[float]]:
    centers: dict[str, list[float]] = {}
    for mark in rendered_scene.mark_traces:
        centers[str(mark["label"])] = [
            round(float(mark["mark_center_px"][0]), 3),
            round(float(mark["mark_center_px"][1]), 3),
        ]
    return centers


def point_for_label(rendered_scene: RenderedChartScene, label: str) -> list[float]:
    centers = mark_centers_by_label(rendered_scene)
    if label not in centers:
        raise KeyError(f"Missing rendered mark for label {label!r}")
    return centers[label]


def point_set_for_labels(rendered_scene: RenderedChartScene, labels: Sequence[str]) -> list[list[float]]:
    centers = mark_centers_by_label(rendered_scene)
    return [centers[str(label)] for label in labels if str(label) in centers]


def keyed_points_for_labels(
    rendered_scene: RenderedChartScene,
    role_to_label: Mapping[str, str],
) -> dict[str, list[float]]:
    centers = mark_centers_by_label(rendered_scene)
    points: dict[str, list[float]] = {}
    for role, label in role_to_label.items():
        if label not in centers:
            raise KeyError(f"Missing rendered mark for label {label!r}")
        points[str(role)] = centers[label]
    return points


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


def keyed_point_artifacts(
    points_by_role: Mapping[str, Sequence[float]],
    labels_by_role: Mapping[str, str],
) -> tuple[TypedValue, dict[str, Any], dict[str, Any]]:
    annotation_map = {
        str(role): [round(float(point[0]), 3), round(float(point[1]), 3)]
        for role, point in points_by_role.items()
    }
    annotation_gt = TypedValue(type="point_map", value=annotation_map)
    witness_symbolic = {
        "type": "object_key_map",
        "keys": dict(labels_by_role),
    }
    projected_annotation = {
        "type": "point_map",
        "point_map": annotation_map,
        "pixel_point_map": annotation_map,
    }
    return annotation_gt, witness_symbolic, projected_annotation
