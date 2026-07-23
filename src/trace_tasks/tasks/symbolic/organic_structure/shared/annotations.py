"""Annotation projection helpers for organic-structure scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from ....shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_set_annotation_artifacts,
    point_set_annotation_artifacts,
    segment_set_annotation_artifacts,
)


def bond_segment_set(segments_by_item: Mapping[str, Sequence[Sequence[float]]], item_ids: Sequence[str]) -> AnnotationArtifacts:
    """Build a segment-set annotation from selected bond ids."""

    return segment_set_annotation_artifacts([segments_by_item[str(item_id)] for item_id in item_ids])


def atom_point_set(points_by_item: Mapping[str, Sequence[float]], item_ids: Sequence[str]) -> AnnotationArtifacts:
    """Build a point-set annotation from selected atom ids."""

    return point_set_annotation_artifacts([points_by_item[str(item_id)] for item_id in item_ids])


def ring_bbox_set(bboxes_by_item: Mapping[str, Sequence[float]], item_ids: Sequence[str]) -> AnnotationArtifacts:
    """Build a bbox-set annotation from selected ring ids."""

    return bbox_set_annotation_artifacts([bboxes_by_item[str(item_id)] for item_id in item_ids])


def witness_payload(artifacts: AnnotationArtifacts) -> dict[str, Any]:
    """Return compact trace witness payload for one annotation artifact."""

    return {"type": str(artifacts.annotation_type), "value": artifacts.value}


__all__ = ["atom_point_set", "bond_segment_set", "ring_bbox_set", "witness_payload"]
