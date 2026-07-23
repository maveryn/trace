"""Annotation projection helpers for circle-pair tangent scenes."""

from __future__ import annotations

from trace_tasks.tasks.geometry.shared.vector2d import point_to_list

from .state import RenderedPairTangentScene


def keyed_point_annotation(rendered: RenderedPairTangentScene) -> dict[str, list[float]]:
    """Return keyed point annotation from final rendered witness positions."""

    return {
        str(key): point_to_list(value)
        for key, value in rendered.annotation_keyed_points.items()
    }


__all__ = ["keyed_point_annotation"]
