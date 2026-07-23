"""Annotation projection helpers for cyclic-order puzzle tasks."""

from __future__ import annotations

from typing import Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, bbox_annotation_artifacts


def option_bbox_annotation(
    option_bboxes: Mapping[str, Sequence[float]],
    option_choice: str,
) -> AnnotationArtifacts:
    """Build scalar-bbox annotation artifacts for one selected option image."""

    selected = str(option_choice)
    if selected not in option_bboxes:
        raise KeyError(f"missing option bbox for {selected}")
    return bbox_annotation_artifacts(option_bboxes[selected])


__all__ = ["option_bbox_annotation"]
