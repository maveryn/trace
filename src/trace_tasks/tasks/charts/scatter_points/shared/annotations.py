"""Annotation helpers for scatter-point chart scenes."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
    point_set_annotation_artifacts,
)

from .state import Dataset, ScatterPointsRenderResult


def point_set_annotation_for_ids(
    *,
    dataset: Dataset,
    rendered: ScatterPointsRenderResult,
) -> tuple[AnnotationArtifacts, dict[str, Any]]:
    """Project the selected point ids into public point-set annotations."""

    point_ids = [str(point_id) for point_id in dataset.query.annotation_point_ids]
    points = [list(rendered.rendered_scene.point_centers[str(point_id)]) for point_id in point_ids]
    base = point_set_annotation_artifacts(points)
    projected = {
        **dict(base.projected_annotation),
        "point_ids": list(point_ids),
    }
    artifacts = AnnotationArtifacts(
        annotation_type=str(base.annotation_type),
        value=list(base.value),
        annotation_gt=base.annotation_gt,
        projected_annotation=dict(projected),
    )
    witness_symbolic = {
        "type": "scatter_point_set",
        "point_ids": list(point_ids),
    }
    return artifacts, witness_symbolic


def bbox_annotation_for_ids(
    *,
    dataset: Dataset,
    rendered: ScatterPointsRenderResult,
) -> tuple[AnnotationArtifacts, dict[str, Any]]:
    """Project the selected point ids into one cluster bbox annotation."""

    point_ids = [str(point_id) for point_id in dataset.query.annotation_point_ids]
    boxes = [list(rendered.rendered_scene.point_bboxes[str(point_id)]) for point_id in point_ids]
    if not boxes:
        raise ValueError("scatter-points bbox annotation requires at least one point id")
    x0 = min(float(box[0]) for box in boxes)
    y0 = min(float(box[1]) for box in boxes)
    x1 = max(float(box[2]) for box in boxes)
    y1 = max(float(box[3]) for box in boxes)
    base = bbox_annotation_artifacts([x0, y0, x1, y1])
    projected = {
        **dict(base.projected_annotation),
        "point_ids": list(point_ids),
    }
    artifacts = AnnotationArtifacts(
        annotation_type=str(base.annotation_type),
        value=list(base.value),
        annotation_gt=base.annotation_gt,
        projected_annotation=dict(projected),
    )
    witness_symbolic = {
        "type": "scatter_point_cluster_bbox",
        "point_ids": list(point_ids),
    }
    return artifacts, witness_symbolic


__all__ = ["bbox_annotation_for_ids", "point_set_annotation_for_ids"]
