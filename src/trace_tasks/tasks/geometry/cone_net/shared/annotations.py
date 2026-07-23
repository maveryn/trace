"""Annotation projection helpers for cone-net diagrams."""

from __future__ import annotations

from trace_tasks.tasks.geometry.shared.annotation_values import (
    PixelAnnotationArtifacts,
    keyed_point_annotation_artifacts,
)

from .state import RenderedConeNetScene


def cone_net_annotation(rendered: RenderedConeNetScene) -> PixelAnnotationArtifacts:
    """Build the public keyed-point annotation from rendered construction points."""

    return keyed_point_annotation_artifacts(
        rendered.annotation_keyed_points,
        roles=rendered.annotation_roles,
    )


__all__ = ["cone_net_annotation"]
