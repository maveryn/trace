"""Annotation projection helpers for color-gradient puzzle tasks."""

from __future__ import annotations

from typing import Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
)


def item_bbox_annotation(
    item_bboxes: Mapping[str, Sequence[float]],
    item_id: str,
) -> AnnotationArtifacts:
    """Build scalar-bbox annotation artifacts for one rendered item."""

    selected = str(item_id)
    if selected not in item_bboxes:
        raise KeyError(f"missing item bbox for {selected}")
    return bbox_annotation_artifacts(item_bboxes[selected])


__all__ = ["item_bbox_annotation"]
