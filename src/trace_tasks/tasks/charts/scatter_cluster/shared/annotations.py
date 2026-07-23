"""Annotation helpers for scatter-cluster chart scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from trace_tasks.core.types import TypedValue

from .state import ScatterClusterDataset, ScatterClusterRenderResult


@dataclass(frozen=True)
class ScatterClusterAnnotationBundle:
    annotation_type: str
    annotation_gt: TypedValue
    projected_annotation: dict[str, Any]
    annotation_refs: list[str]
    annotation_cluster_labels: list[str]
    annotation_point_ids: list[str]


def _point_ids_for_clusters(dataset: ScatterClusterDataset, cluster_labels: list[str]) -> list[str]:
    selected = set(str(label) for label in cluster_labels)
    return [
        str(point.point_id)
        for cluster in dataset.clusters
        if str(cluster.cluster_label) in selected
        for point in cluster.points
    ]


def cluster_bbox_annotation(
    *,
    dataset: ScatterClusterDataset,
    rendered: ScatterClusterRenderResult,
    cluster_label: str,
) -> ScatterClusterAnnotationBundle:
    rendered_scene = rendered.rendered_scene
    label = str(cluster_label)
    bbox = list(rendered_scene.cluster_bboxes[label])
    point_ids = _point_ids_for_clusters(dataset, [label])
    return ScatterClusterAnnotationBundle(
        annotation_type="bbox",
        annotation_gt=TypedValue(type="bbox", value=list(bbox)),
        projected_annotation={
            "type": "bbox",
            "bbox": list(bbox),
            "pixel_bbox": list(bbox),
            "point_ids": list(point_ids),
            "cluster_labels": [label],
            "cluster_bboxes": {label: list(bbox)},
            "cluster_envelope_bboxes": dict(rendered_scene.cluster_envelope_bboxes),
        },
        annotation_refs=[label],
        annotation_cluster_labels=[label],
        annotation_point_ids=list(point_ids),
    )


def centroid_option_point_annotation(
    *,
    dataset: ScatterClusterDataset,
    rendered: ScatterClusterRenderResult,
    selected_option_label: str,
) -> ScatterClusterAnnotationBundle:
    rendered_scene = rendered.rendered_scene
    option_label = str(selected_option_label)
    point = list(rendered_scene.option_centers_px[option_label])
    return ScatterClusterAnnotationBundle(
        annotation_type="point",
        annotation_gt=TypedValue(type="point", value=list(point)),
        projected_annotation={
            "type": "point",
            "point": list(point),
            "pixel_point": list(point),
            "option_labels": [option_label],
            "option_bboxes": dict(rendered_scene.option_bboxes),
            "option_centers_px": dict(rendered_scene.option_centers_px),
        },
        annotation_refs=[option_label],
        annotation_cluster_labels=[],
        annotation_point_ids=[],
    )
