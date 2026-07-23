"""Annotation helpers for printed map scene packages."""

from __future__ import annotations

from typing import Any, Dict, Sequence

from trace_tasks.tasks.pages.shared.diagram.common import projected_diagram_bbox_sequence_annotation

from .state import RenderedMapScene


def annotation_bbox_source(rendered_scene: RenderedMapScene) -> Dict[str, Sequence[float]]:
    """Return all bbox maps addressable by map route annotations."""

    return {
        **dict(rendered_scene.landmark_bbox_map),
        **dict(rendered_scene.zone_label_bbox_map),
    }


def bbox_sequence_for_ids(rendered_scene: RenderedMapScene, bbox_ids: Sequence[str]) -> list[list[float]]:
    """Project ordered bbox ids into prompt-facing bbox-sequence values."""

    projection = projected_diagram_bbox_sequence_annotation(
        annotation_bbox_source(rendered_scene),
        [str(item) for item in bbox_ids],
    )
    return [
        [round(float(value), 3) for value in bbox]
        for bbox in projection["bbox_sequence"]
    ]


def projected_annotation(rendered_scene: RenderedMapScene, bbox_ids: Sequence[str]) -> Dict[str, Any]:
    """Return the trace projection payload for one ordered map route."""

    projection = projected_diagram_bbox_sequence_annotation(
        annotation_bbox_source(rendered_scene),
        [str(item) for item in bbox_ids],
    )
    return {
        **dict(projection),
        "bbox_sequence": bbox_sequence_for_ids(rendered_scene, bbox_ids),
    }


__all__ = [
    "annotation_bbox_source",
    "bbox_sequence_for_ids",
    "projected_annotation",
]
