"""Annotation helpers for combo-mark chart tasks."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.combo_mark.shared.state import ComboScene
from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    point_annotation_artifacts,
    segment_set_annotation_artifacts,
)


def combo_annotation_artifacts(
    scene: ComboScene,
    *,
    indices: Iterable[int],
    mode: str,
    mark_role: str = "",
) -> tuple[AnnotationArtifacts, list[str]]:
    """Project task-selected combo marks using the scene-level annotation contract."""

    selected_indices = tuple(int(idx) for idx in indices)
    if str(mode) == "paired_mark_map":
        if len(selected_indices) != 1:
            raise ValueError("paired_mark_map expects exactly one selected category")
        idx = int(selected_indices[0])
        points = {
            "primary_mark": [float(scene.primary_points[idx][0]), float(scene.primary_points[idx][1])],
            "line_mark": [float(scene.line_points[idx][0]), float(scene.line_points[idx][1])],
        }
        labels = [f"{scene.primary_name}:{scene.labels[idx]}", f"{scene.line_name}:{scene.labels[idx]}"]
        return keyed_point_artifacts(points), labels
    if str(mode) == "single_mark_map":
        if len(selected_indices) != 1:
            raise ValueError("single_mark_map expects exactly one selected category")
        idx = int(selected_indices[0])
        if str(mark_role) == "primary":
            point = scene.primary_points[idx]
            labels = [f"{scene.primary_name}:{scene.labels[idx]}"]
        elif str(mark_role) == "line":
            point = scene.line_points[idx]
            labels = [f"{scene.line_name}:{scene.labels[idx]}"]
        else:
            raise ValueError(f"unsupported single mark role: {mark_role}")
        return keyed_point_artifacts({"answer_mark": [float(point[0]), float(point[1])]}), labels
    if str(mode) == "single_mark_point":
        if len(selected_indices) != 1:
            raise ValueError("single_mark_point expects exactly one selected category")
        idx = int(selected_indices[0])
        if str(mark_role) == "primary":
            point = scene.primary_points[idx]
            labels = [f"{scene.primary_name}:{scene.labels[idx]}"]
        elif str(mark_role) == "line":
            point = scene.line_points[idx]
            labels = [f"{scene.line_name}:{scene.labels[idx]}"]
        else:
            raise ValueError(f"unsupported single mark role: {mark_role}")
        return point_annotation_artifacts([float(point[0]), float(point[1])]), labels
    if str(mode) == "mark_pair_set":
        pairs = [
            [
                [float(scene.primary_points[int(idx)][0]), float(scene.primary_points[int(idx)][1])],
                [float(scene.line_points[int(idx)][0]), float(scene.line_points[int(idx)][1])],
            ]
            for idx in selected_indices
        ]
        labels = [f"{scene.primary_name}+{scene.line_name}:{scene.labels[int(idx)]}" for idx in selected_indices]
        return segment_set_annotation_artifacts(pairs), labels
    raise ValueError(f"unsupported combo annotation mode: {mode}")


def keyed_point_artifacts(points: Mapping[str, Sequence[float]]) -> AnnotationArtifacts:
    value = {
        str(key): [round(float(point[0]), 3), round(float(point[1]), 3)]
        for key, point in points.items()
    }
    projected = {
        "type": "point_map",
        "point_map": dict(value),
        "pixel_point_map": dict(value),
    }
    return AnnotationArtifacts(
        annotation_type="point_map",
        value=dict(value),
        annotation_gt=TypedValue(type="point_map", value=dict(value)),
        projected_annotation=projected,
    )


__all__ = ["combo_annotation_artifacts", "keyed_point_artifacts"]
