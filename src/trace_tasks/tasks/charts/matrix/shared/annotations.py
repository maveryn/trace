"""Annotation projection helpers for matrix chart tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.charts.shared.grid.annotations import bbox_refs
from trace_tasks.tasks.charts.shared.grid.geometry import bboxes_for_ids

from .state import RenderedMatrix


@dataclass(frozen=True)
class MatrixAnnotationBundle:
    annotation_gt: TypedValue
    projected_annotation: Dict[str, Any]
    annotation_refs: List[Dict[str, Any]]
    annotation_cell_ids: List[str]
    annotation_type: str


def matrix_bbox_set_bundle(rendered_scene: RenderedMatrix, cell_ids: Sequence[str]) -> MatrixAnnotationBundle:
    """Return bbox-set annotations for the task-selected matrix cells."""

    normalized_ids = [str(cell_id) for cell_id in cell_ids]
    bboxes = bboxes_for_ids(rendered_scene.cell_bbox_map, normalized_ids, missing="error")
    refs = bbox_refs(ids=normalized_ids, boxes=bboxes, role="cell", id_key="id", bbox_key="bbox")
    return MatrixAnnotationBundle(
        annotation_gt=TypedValue(type="bbox_set", value=list(bboxes)),
        projected_annotation={
            "bbox_set": list(bboxes),
            "entries": [dict(ref) for ref in refs],
            "cell_ids": list(normalized_ids),
        },
        annotation_refs=[dict(ref) for ref in refs],
        annotation_cell_ids=list(normalized_ids),
        annotation_type="bbox_set",
    )


__all__ = ["MatrixAnnotationBundle", "matrix_bbox_set_bundle"]
