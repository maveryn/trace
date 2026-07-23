"""Annotation projection helpers for cell-board coordinates and paths."""

from __future__ import annotations

from typing import Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_set_annotation_artifacts,
    segment_set_annotation_artifacts,
)
from trace_tasks.tasks.shared.bbox_projection import BBox, bbox_center

from .topology import Coord, cell_id, sort_coords


def coord_bbox_set_annotation(
    *,
    coords: Sequence[Coord],
    bbox_map: Mapping[str, BBox],
) -> AnnotationArtifacts:
    """Project unordered board cells as public bbox-set annotation."""

    bboxes = [bbox_map[cell_id(coord)] for coord in sort_coords(coords)]
    return bbox_set_annotation_artifacts(bboxes)


def coord_path_segment_set_annotation(
    *,
    path: Sequence[Coord],
    bbox_map: Mapping[str, BBox],
) -> AnnotationArtifacts:
    """Project adjacent board-cell path steps as public segment-set annotation."""

    ordered = [(int(row), int(col)) for row, col in path]
    segments = []
    for first, second in zip(ordered, ordered[1:]):
        first_center = bbox_center(bbox_map[cell_id(first)])
        second_center = bbox_center(bbox_map[cell_id(second)])
        segments.append([list(first_center), list(second_center)])
    return segment_set_annotation_artifacts(segments)


def coord_pair_segment_set_annotation(
    *,
    coord_pairs: Sequence[tuple[Coord, Coord]],
    bbox_map: Mapping[str, BBox],
) -> AnnotationArtifacts:
    """Project paired board-cell centers as public segment-set annotation."""

    segments = []
    for first, second in coord_pairs:
        first_center = bbox_center(bbox_map[cell_id(first)])
        second_center = bbox_center(bbox_map[cell_id(second)])
        segments.append([list(first_center), list(second_center)])
    return segment_set_annotation_artifacts(segments)


__all__ = [
    "coord_bbox_set_annotation",
    "coord_pair_segment_set_annotation",
    "coord_path_segment_set_annotation",
]
