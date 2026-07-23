"""Annotation projection helpers for slot-machine tasks."""

from __future__ import annotations

from typing import Sequence

from trace_tasks.tasks.shared.annotation_artifacts import (
    AnnotationArtifacts,
    bbox_annotation_artifacts,
    segment_annotation_artifacts,
    segment_set_annotation_artifacts,
)

from .rendering import RenderedSlotMachineScene
from .state import payline_entity_id


def payline_segment_set_annotation(
    rendered_scene: RenderedSlotMachineScene,
    payline_keys: Sequence[str],
) -> AnnotationArtifacts:
    """Build a segment-set annotation for selected conceptual paylines."""

    segments = [
        rendered_scene.render_map["payline_segments_px"][payline_entity_id(str(payline_key))]
        for payline_key in payline_keys
    ]
    return segment_set_annotation_artifacts(segments)


def payline_segment_annotation(
    rendered_scene: RenderedSlotMachineScene,
    payline_key: str,
) -> AnnotationArtifacts:
    """Build a scalar segment annotation for one conceptual payline."""

    segment = rendered_scene.render_map["payline_segments_px"][payline_entity_id(str(payline_key))]
    return segment_annotation_artifacts(segment)


def option_bbox_annotation(
    rendered_scene: RenderedSlotMachineScene,
    option_label: str,
) -> AnnotationArtifacts:
    """Build a bbox annotation for one selected third-reel option."""

    bboxes = dict(rendered_scene.render_map.get("option_bboxes_px", {}))
    label = str(option_label)
    if label not in bboxes:
        raise ValueError(f"missing rendered slot-machine option bbox for {label}")
    return bbox_annotation_artifacts(bboxes[label])


__all__ = ["option_bbox_annotation", "payline_segment_annotation", "payline_segment_set_annotation"]
