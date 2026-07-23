"""Annotation helpers for solid cross-section diagrams."""

from __future__ import annotations

from typing import Mapping

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_to_list

from .state import RenderedSolidCrossSectionScene

CROSS_SECTION_ANNOTATION_ROLE = "cross_section"


def cross_section_bbox_annotation(rendered: RenderedSolidCrossSectionScene) -> tuple[TypedValue, dict[str, object]]:
    """Build scalar bbox annotation artifacts around the marked cross-section."""

    bbox = bbox_to_list(rendered.annotation_bboxes[CROSS_SECTION_ANNOTATION_ROLE])
    projected_annotation = {
        "type": "bbox",
        "bbox": list(bbox),
        "pixel_bbox": list(bbox),
    }
    return TypedValue(type="bbox", value=list(bbox)), projected_annotation


def annotation_roles_metadata(projected_annotation: Mapping[str, object]) -> list[str]:
    """Return prompt-facing annotation roles for trace metadata."""

    if projected_annotation.get("type") == "bbox":
        return [CROSS_SECTION_ANNOTATION_ROLE]
    return []


__all__ = [
    "annotation_roles_metadata",
    "cross_section_bbox_annotation",
    "CROSS_SECTION_ANNOTATION_ROLE",
]
