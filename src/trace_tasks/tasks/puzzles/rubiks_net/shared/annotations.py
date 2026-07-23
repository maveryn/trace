"""Annotation projection helpers for Rubik option-panel witnesses."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
)


def selected_option_bbox_annotation(
    option_panel_bbox_map: Mapping[str, Sequence[float]],
    option_panel_id: str,
) -> AnnotationArtifacts:
    """Project the selected option panel as a scalar bbox annotation."""

    key = str(option_panel_id)
    if key not in option_panel_bbox_map:
        raise RuntimeError(f"missing Rubik option panel bbox: {key!r}")
    return bbox_annotation_artifacts(option_panel_bbox_map[key])


__all__ = ["selected_option_bbox_annotation"]
