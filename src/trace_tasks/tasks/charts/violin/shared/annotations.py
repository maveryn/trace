"""Annotation helpers for violin chart tasks."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.charts.shared.cartesian.annotations import projected_mark_annotation
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, bbox_annotation_artifacts


def bbox_artifacts_for_label(rendered_scene: Any, answer_label: str) -> AnnotationArtifacts:
    """Return scalar bbox annotation artifacts for one selected violin."""

    projection = projected_mark_annotation(rendered_scene, [str(answer_label)])
    bboxes = [list(bbox) for bbox in projection["bbox_set"]]
    if len(bboxes) != 1:
        raise ValueError(f"expected one violin bbox for label {answer_label!r}, got {len(bboxes)}")
    return bbox_annotation_artifacts(bboxes[0])


def label_centers(rendered_scene: Any) -> dict[str, list[float]]:
    """Return visible label centers keyed by label text."""

    return {
        str(mark["label"]): list(mark["label_center_px"])
        for mark in rendered_scene.mark_traces
    }


__all__ = ["bbox_artifacts_for_label", "label_centers"]
