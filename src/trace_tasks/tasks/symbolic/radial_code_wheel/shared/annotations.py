"""Annotation helpers for radial code-wheel scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from .....core.types import TypedValue
from ....shared.annotation_artifacts import AnnotationArtifacts


def round_bbox(bbox: Sequence[float]) -> list[float]:
    """Round one bbox to stable JSON-friendly floats."""

    return [round(float(value), 3) for value in bbox[:4]]


def round_bbox_map(mapping: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Round a string-keyed bbox map."""

    return {str(key): round_bbox(bbox) for key, bbox in mapping.items()}


def round_point(point: Sequence[float]) -> list[float]:
    """Round one point to stable JSON-friendly floats."""

    return [round(float(point[0]), 3), round(float(point[1]), 3)]


def round_point_map(mapping: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Round a string-keyed point map."""

    return {str(key): round_point(point) for key, point in mapping.items()}


def role_point_map(item_points: Mapping[str, Sequence[float]], roles: Mapping[str, str]) -> AnnotationArtifacts:
    """Build a role-keyed point-map annotation from selected item ids."""

    keyed = {str(role): round_point(item_points[str(item_id)]) for role, item_id in roles.items()}
    projected = {
        "type": "point_map",
        "point_map": dict(keyed),
        "pixel_point_map": dict(keyed),
    }
    return AnnotationArtifacts(
        annotation_type="point_map",
        value=dict(keyed),
        annotation_gt=TypedValue(type="point_map", value=dict(keyed)),
        projected_annotation=projected,
    )


def witness_payload(artifacts: AnnotationArtifacts) -> dict[str, Any]:
    """Return the compact witness payload for one annotation artifact."""

    return {"type": str(artifacts.annotation_type), "value": artifacts.value}


__all__ = ["role_point_map", "round_bbox", "round_bbox_map", "round_point", "round_point_map", "witness_payload"]
