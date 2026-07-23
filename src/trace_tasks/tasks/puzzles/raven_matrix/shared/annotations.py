"""Annotation projection helpers for Raven-matrix option witnesses."""

from __future__ import annotations

from typing import Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
)


def selected_option_bbox_annotation(
    option_cell_bbox_map: Mapping[str, Sequence[float]],
    option_panel_id: str,
) -> AnnotationArtifacts:
    """Project the selected option cell as a scalar bbox annotation."""

    key = str(option_panel_id)
    if key not in option_cell_bbox_map:
        raise RuntimeError(f"missing Raven option cell bbox: {key!r}")
    return bbox_annotation_artifacts(option_cell_bbox_map[key])


__all__ = ["selected_option_bbox_annotation"]
