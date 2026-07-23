"""Annotation helpers for solid-revolution diagrams."""

from __future__ import annotations

from typing import Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_to_list

from .state import RenderedSolidRevolutionScene


def bbox_map_annotation(
    rendered: RenderedSolidRevolutionScene,
    keys: Sequence[str],
) -> tuple[TypedValue, dict[str, object]]:
    """Build role-bound bbox annotation artifacts for one rendered scene."""

    bbox_map = {
        str(key): bbox_to_list(rendered.annotation_bboxes[str(key)])
        for key in tuple(str(value) for value in keys)
    }
    projected_annotation = {
        "type": "bbox_map",
        "bbox_map": dict(bbox_map),
        "pixel_bbox_map": dict(bbox_map),
    }
    return TypedValue(type="bbox_map", value=dict(bbox_map)), projected_annotation


def annotation_roles_metadata(annotation_value: Mapping[str, object]) -> list[str]:
    """Return annotation keys in prompt-facing order."""

    return [str(key) for key in annotation_value.keys()]


__all__ = ["annotation_roles_metadata", "bbox_map_annotation"]
