"""Annotation projection helpers for dots-and-boxes games tasks."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
    bbox_set_annotation_artifacts,
    segment_set_annotation_artifacts,
)


def dots_and_boxes_annotation_artifacts(
    *,
    annotation_kind: str,
    annotation_entity_ids: Sequence[str],
    render_map: Mapping[str, Any],
) -> AnnotationArtifacts:
    """Project dots-and-boxes witnesses into the requested annotation type."""

    if str(annotation_kind) == "box":
        bboxes = [
            list(render_map["box_bboxes_px"][str(box_id)])
            for box_id in annotation_entity_ids
        ]
        return bbox_set_annotation_artifacts(bboxes)
    if str(annotation_kind) == "single_box":
        if len(tuple(annotation_entity_ids)) != 1:
            raise ValueError("single_box annotation requires exactly one box id")
        bbox = list(render_map["box_bboxes_px"][str(tuple(annotation_entity_ids)[0])])
        return bbox_annotation_artifacts(bbox)
    if str(annotation_kind) == "edge_point_pair":
        point_pairs = [
            [list(point) for point in render_map["edge_point_pairs_px"][str(edge_id)]]
            for edge_id in annotation_entity_ids
        ]
        return segment_set_annotation_artifacts(point_pairs)
    raise ValueError(f"unsupported dots-and-boxes annotation kind: {annotation_kind}")


__all__ = ["dots_and_boxes_annotation_artifacts"]
