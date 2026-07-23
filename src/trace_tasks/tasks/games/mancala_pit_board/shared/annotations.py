"""Annotation projection helpers for Mancala pit-board tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from trace_tasks.core.types import TypedValue
from trace_tasks.tasks.shared.annotation_artifacts import bbox_annotation_artifacts

from .state import RenderedMancalaScene


@dataclass(frozen=True)
class MancalaAnnotationBundle:
    """Task-bound annotation plus symbolic witness ids."""

    annotation_gt: TypedValue
    projected_annotation: Mapping[str, Any]
    witness_symbolic: Mapping[str, Any]
    entity_ids: Mapping[str, Any]


def _round_bbox(bbox: Sequence[float]) -> list[float]:
    return [round(float(value), 3) for value in bbox[:4]]


def pit_bbox_set_annotation(
    *,
    rendered: RenderedMancalaScene,
    pit_ids: Sequence[str],
    role_name: str,
) -> MancalaAnnotationBundle:
    """Project an unordered bbox set for selected visible pits."""

    pit_bboxes = rendered.render_map["pit_bboxes_px"]
    values = [_round_bbox(pit_bboxes[str(pit_id)]) for pit_id in pit_ids]
    return MancalaAnnotationBundle(
        annotation_gt=TypedValue(type="bbox_set", value=[list(value) for value in values]),
        projected_annotation={
            "type": "bbox_set",
            "bbox_set": [list(value) for value in values],
            "pixel_bbox_set": [list(value) for value in values],
        },
        witness_symbolic={"type": "bbox_set", "ids": [str(pit_id) for pit_id in pit_ids]},
        entity_ids={str(role_name): [str(pit_id) for pit_id in pit_ids]},
    )


def pit_bbox_annotation(
    *,
    rendered: RenderedMancalaScene,
    pit_id: str,
    role_name: str,
) -> MancalaAnnotationBundle:
    """Project one selected visible pit to a scalar bbox annotation."""

    pit_bbox = _round_bbox(rendered.render_map["pit_bboxes_px"][str(pit_id)])
    artifacts = bbox_annotation_artifacts(pit_bbox)
    return MancalaAnnotationBundle(
        annotation_gt=artifacts.annotation_gt,
        projected_annotation=dict(artifacts.projected_annotation),
        witness_symbolic={"type": artifacts.annotation_type, "ids": [str(pit_id)]},
        entity_ids={str(role_name): str(pit_id)},
    )


def keyed_pit_bbox_annotation(
    *,
    rendered: RenderedMancalaScene,
    role_pit_ids: Mapping[str, str],
) -> MancalaAnnotationBundle:
    """Project bbox-map roles from task-selected pit ids."""

    pit_bboxes = rendered.render_map["pit_bboxes_px"]
    values = {
        str(role): _round_bbox(pit_bboxes[str(pit_id)])
        for role, pit_id in dict(role_pit_ids).items()
    }
    ids = {str(role): str(pit_id) for role, pit_id in dict(role_pit_ids).items()}
    return MancalaAnnotationBundle(
        annotation_gt=TypedValue(type="bbox_map", value=dict(values)),
        projected_annotation={
            "type": "bbox_map",
            "bbox_map": dict(values),
            "pixel_bbox_map": dict(values),
        },
        witness_symbolic={"type": "bbox_map", "ids": dict(ids)},
        entity_ids=dict(ids),
    )


__all__ = [
    "MancalaAnnotationBundle",
    "keyed_pit_bbox_annotation",
    "pit_bbox_annotation",
    "pit_bbox_set_annotation",
]
