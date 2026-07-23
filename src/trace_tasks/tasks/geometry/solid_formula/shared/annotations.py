"""Annotation helpers for solid-formula diagrams."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_to_list

from .state import RenderedSolidFormulaScene


def solid_bbox_annotation(rendered: RenderedSolidFormulaScene) -> tuple[TypedValue, dict[str, object]]:
    """Build scalar bbox annotation artifacts around the rendered solid."""

    solid_bbox = rendered.render_map.get("solid", {}).get("bbox")
    if not isinstance(solid_bbox, Sequence) or len(solid_bbox) != 4:
        raise ValueError("rendered solid_formula scene is missing render_map.solid.bbox")
    bbox = bbox_to_list(tuple(float(value) for value in solid_bbox))
    projected_annotation = {
        "type": "bbox",
        "bbox": list(bbox),
        "pixel_bbox": list(bbox),
    }
    return TypedValue(type="bbox", value=list(bbox)), projected_annotation


def annotation_roles_metadata(projected_annotation: Mapping[str, object]) -> list[str]:
    """Return prompt-facing annotation roles for trace metadata."""

    if projected_annotation.get("type") == "bbox":
        return ["solid"]
    return []


__all__ = ["annotation_roles_metadata", "solid_bbox_annotation"]
