"""Annotation helpers for the boxplot chart scene."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts
from trace_tasks.tasks.shared.annotation_artifacts import bbox_annotation_artifacts
from trace_tasks.tasks.shared.annotation_artifacts import point_annotation_artifacts


def keyed_point_artifacts(
    role_to_point: Mapping[str, Sequence[float]],
    role_to_label: Mapping[str, str],
) -> tuple[AnnotationArtifacts, dict[str, Any]]:
    """Build keyed point annotations and symbolic role metadata."""

    keyed_points = {
        str(role): [round(float(point[0]), 3), round(float(point[1]), 3)]
        for role, point in role_to_point.items()
    }
    projected_annotation = {
        "type": "point_map",
        "point_map": dict(keyed_points),
        "pixel_point_map": dict(keyed_points),
    }
    artifacts = AnnotationArtifacts(
        annotation_type="point_map",
        value=dict(keyed_points),
        annotation_gt=TypedValue(type="point_map", value=dict(keyed_points)),
        projected_annotation=projected_annotation,
    )
    witness_symbolic = {
        "type": "object_key_map",
        "keys": {str(role): str(label) for role, label in role_to_label.items()},
    }
    return artifacts, witness_symbolic


def scalar_point_artifacts(
    role_to_point: Mapping[str, Sequence[float]],
    role_to_label: Mapping[str, str],
) -> tuple[AnnotationArtifacts, dict[str, Any]]:
    """Build scalar point annotation for a single selected boxplot witness."""

    if len(role_to_point) != 1:
        raise ValueError("scalar boxplot point annotation requires exactly one point")
    role, point = next(iter(role_to_point.items()))
    label = str(role_to_label[str(role)])
    artifacts = point_annotation_artifacts(point)
    witness_symbolic = {
        "type": "object_key",
        "role": str(role),
        "label": label,
    }
    return artifacts, witness_symbolic


def scalar_bbox_artifacts(
    role_to_bbox: Mapping[str, Sequence[float]],
    role_to_label: Mapping[str, str],
) -> tuple[AnnotationArtifacts, dict[str, Any]]:
    """Build scalar bbox annotation for a single selected boxplot rectangle."""

    if len(role_to_bbox) != 1:
        raise ValueError("scalar boxplot bbox annotation requires exactly one bbox")
    role, bbox = next(iter(role_to_bbox.items()))
    label = str(role_to_label[str(role)])
    artifacts = bbox_annotation_artifacts(bbox)
    witness_symbolic = {
        "type": "object_key",
        "role": str(role),
        "label": label,
    }
    return artifacts, witness_symbolic


__all__ = ["keyed_point_artifacts", "scalar_bbox_artifacts", "scalar_point_artifacts"]
