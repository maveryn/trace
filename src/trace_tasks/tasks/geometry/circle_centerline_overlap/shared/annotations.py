"""Annotation projection helpers for circle-centerline-overlap scenes."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.geometry.shared.vector2d import point_to_list

from .state import RenderedCenterlineOverlapScene


def keyed_point_annotation(rendered: RenderedCenterlineOverlapScene) -> dict[str, list[float]]:
    """Return keyed point annotation from final rendered witness positions."""

    return {
        str(key): point_to_list(value)
        for key, value in rendered.annotation_keyed_points.items()
    }


def segment_annotation(
    rendered: RenderedCenterlineOverlapScene,
    endpoint_keys: Sequence[str],
) -> list[list[float]]:
    """Return the requested visual segment endpoints after final layout projection."""

    endpoints = tuple(str(key) for key in endpoint_keys)
    if len(endpoints) != 2:
        raise ValueError("segment annotation requires exactly two endpoint keys")
    return [point_to_list(rendered.annotation_keyed_points[key]) for key in endpoints]


__all__ = ["keyed_point_annotation", "segment_annotation"]
