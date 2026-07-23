"""Annotation helpers for measuring-tool scenes."""

from __future__ import annotations

from typing import Mapping, Sequence

from trace_tasks.tasks.geometry.shared.annotation_values import (
    PixelAnnotationArtifacts,
    keyed_point_annotation_artifacts,
)
from trace_tasks.tasks.shared.annotation_artifacts import segment_annotation_artifacts


def measuring_tool_point_annotation(
    points: Mapping[str, Sequence[float]],
    *,
    roles: Sequence[str],
) -> PixelAnnotationArtifacts:
    """Build role-bound pixel point annotation for a visible measurement."""

    return keyed_point_annotation_artifacts(points, roles=tuple(str(role) for role in roles))


def measuring_tool_segment_annotation(
    points: Mapping[str, Sequence[float]],
    *,
    start_role: str,
    end_role: str,
) -> PixelAnnotationArtifacts:
    """Build scalar segment annotation for one measured visual length."""

    artifacts = segment_annotation_artifacts(
        [points[str(start_role)], points[str(end_role)]]
    )
    return PixelAnnotationArtifacts(
        annotation_type=artifacts.annotation_type,
        value=artifacts.value,
        projected_annotation=artifacts.projected_annotation,
    )


__all__ = ["measuring_tool_point_annotation", "measuring_tool_segment_annotation"]
