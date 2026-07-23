"""Annotation helpers for histogram chart scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.charts.shared.cartesian.annotations import projected_mark_annotation
from trace_tasks.tasks.charts.shared.chart_scene_types import RenderedChartScene
from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
    bbox_set_annotation_artifacts,
)


def bbox_map_for_labels(
    rendered_scene: RenderedChartScene,
    labels: Sequence[str],
) -> dict[str, list[float]]:
    """Return bar bboxes keyed by visible x-axis label."""

    requested = [str(label) for label in labels]
    projection = projected_mark_annotation(rendered_scene, requested)
    bboxes = [list(bbox) for bbox in projection["bbox_set"]]
    return {
        str(label): [round(float(value), 3) for value in bbox]
        for label, bbox in zip(requested, bboxes)
    }


def bbox_set_artifacts_for_labels(
    label_to_bbox: Mapping[str, Sequence[float]],
    labels: Sequence[str],
) -> tuple[AnnotationArtifacts, dict[str, Any]]:
    """Build bbox-set artifacts for a homogeneous set of histogram bars."""

    ordered_labels = [str(label) for label in labels]
    bboxes = [label_to_bbox[str(label)] for label in ordered_labels]
    artifacts = bbox_set_annotation_artifacts(bboxes)
    witness_symbolic = {
        "type": "object_set",
        "labels": list(ordered_labels),
    }
    return artifacts, witness_symbolic


def bbox_artifacts_for_label(
    label_to_bbox: Mapping[str, Sequence[float]],
    label: str,
) -> tuple[AnnotationArtifacts, dict[str, Any]]:
    """Build scalar bbox artifacts for one histogram bar."""

    resolved_label = str(label)
    artifacts = bbox_annotation_artifacts(label_to_bbox[resolved_label])
    witness_symbolic = {
        "type": "object",
        "label": str(resolved_label),
    }
    return artifacts, witness_symbolic


__all__ = [
    "bbox_artifacts_for_label",
    "bbox_map_for_labels",
    "bbox_set_artifacts_for_labels",
]
