"""Annotation projection helpers for Snake cells."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, bbox_set_annotation_artifacts


def cell_bbox_set(render_map: Mapping[str, Any], *, cell_ids: Sequence[str]) -> AnnotationArtifacts:
    """Project visible grid-cell ids to a bbox-set annotation."""

    cell_bboxes = render_map.get("cell_bboxes_px", {})
    if not isinstance(cell_bboxes, Mapping):
        raise ValueError("Snake render map is missing cell_bboxes_px")
    bboxes: list[list[float]] = []
    for cell_id in cell_ids:
        if str(cell_id) not in cell_bboxes:
            raise ValueError(f"unknown Snake annotation cell: {cell_id}")
        bboxes.append([float(value) for value in cell_bboxes[str(cell_id)]])
    return bbox_set_annotation_artifacts(bboxes)


__all__ = ["cell_bbox_set"]
