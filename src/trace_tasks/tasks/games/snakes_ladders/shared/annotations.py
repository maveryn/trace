"""Annotation projection helpers for Snakes and Ladders scenes."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.shared.annotation_artifacts import AnnotationArtifacts, bbox_annotation_artifacts, bbox_set_annotation_artifacts


def _bbox(render_map: Mapping[str, Any], entity_id: str) -> list[float]:
    return [round(float(value), 3) for value in render_map["entity_bboxes_px"][str(entity_id)][:4]]


def bbox_map_annotation_artifacts(
    render_map: Mapping[str, Any],
    role_entity_ids: Mapping[str, str],
) -> tuple[AnnotationArtifacts, dict[str, Any]]:
    """Build a role-bound bbox map for start/end square witnesses."""

    values = {str(role): _bbox(render_map, str(entity_id)) for role, entity_id in role_entity_ids.items()}
    return (
        AnnotationArtifacts(
            annotation_type="bbox_map",
            value=dict(values),
            annotation_gt=TypedValue(type="bbox_map", value=dict(values)),
            projected_annotation={
                "type": "bbox_map",
                "bbox_map": dict(values),
                "pixel_bbox_map": dict(values),
            },
        ),
        {"type": "object_map", "ids": dict(role_entity_ids)},
    )


def bbox_annotation_for_entity(
    render_map: Mapping[str, Any],
    entity_id: str,
) -> tuple[AnnotationArtifacts, dict[str, Any]]:
    """Build scalar bbox annotation for one square witness."""

    artifacts = bbox_annotation_artifacts(_bbox(render_map, str(entity_id)))
    return artifacts, {"type": "object", "id": str(entity_id)}


def bbox_set_annotation_for_entities(
    render_map: Mapping[str, Any],
    entity_ids: Sequence[str],
) -> tuple[AnnotationArtifacts, dict[str, Any]]:
    """Build bbox-set annotation for counted square witnesses."""

    ids = [str(entity_id) for entity_id in entity_ids]
    artifacts = bbox_set_annotation_artifacts([_bbox(render_map, entity_id) for entity_id in ids])
    return artifacts, {"type": "object_set", "ids": list(ids)}


__all__ = [
    "bbox_annotation_for_entity",
    "bbox_map_annotation_artifacts",
    "bbox_set_annotation_for_entities",
]
