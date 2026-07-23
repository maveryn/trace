"""Annotation helpers for Braille-cell scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .....core.types import TypedValue
from ....shared.annotation_artifacts import AnnotationArtifacts, point_set_annotation_artifacts


def round_bbox(bbox: Sequence[float]) -> list[float]:
    """Round one bbox to stable JSON-friendly floats."""

    return [round(float(value), 3) for value in bbox[:4]]


def round_point(point: Sequence[float]) -> list[float]:
    """Round one point to stable JSON-friendly floats."""

    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def round_bbox_map(mapping: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Round a string-keyed bbox map."""

    return {str(key): round_bbox(bbox) for key, bbox in mapping.items()}


def round_point_map(mapping: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Round a string-keyed point map."""

    return {str(key): round_point(point) for key, point in mapping.items()}


def dot_point_set(dot_centers: Mapping[str, Sequence[float]], dot_ids: Sequence[str]) -> AnnotationArtifacts:
    """Build a point-set annotation from selected dot center ids."""

    return point_set_annotation_artifacts([dot_centers[str(dot_id)] for dot_id in dot_ids])


def role_bbox_map(item_bboxes: Mapping[str, Sequence[float]], roles: Mapping[str, str]) -> AnnotationArtifacts:
    """Build a role-keyed bbox-map annotation from selected item ids."""

    keyed = {str(role): round_bbox(item_bboxes[str(item_id)]) for role, item_id in roles.items()}
    projected = {
        "type": "bbox_map",
        "bbox_map": dict(keyed),
        "pixel_bbox_map": dict(keyed),
    }
    return AnnotationArtifacts(
        annotation_type="bbox_map",
        value=dict(keyed),
        annotation_gt=TypedValue(type="bbox_map", value=dict(keyed)),
        projected_annotation=projected,
    )


def witness_payload(artifacts: AnnotationArtifacts) -> dict[str, Any]:
    """Return the compact witness payload for one annotation artifact."""

    return {"type": str(artifacts.annotation_type), "value": artifacts.value}


__all__ = [
    "dot_point_set",
    "role_bbox_map",
    "round_bbox",
    "round_bbox_map",
    "round_point",
    "round_point_map",
    "witness_payload",
]
