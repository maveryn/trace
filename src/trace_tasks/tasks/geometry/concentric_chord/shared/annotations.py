"""Annotation projection helpers for concentric-chord diagrams."""

from __future__ import annotations

from trace_tasks.tasks.geometry.shared.annotation_values import (
    PixelAnnotationArtifacts,
    keyed_point_annotation_artifacts,
)

from .state import RenderedConcentricChordScene


def concentric_chord_annotation(rendered: RenderedConcentricChordScene) -> PixelAnnotationArtifacts:
    """Build the public keyed-point annotation from rendered construction points."""

    return keyed_point_annotation_artifacts(
        rendered.annotation_keyed_points,
        roles=rendered.annotation_roles,
    )


__all__ = ["concentric_chord_annotation"]
