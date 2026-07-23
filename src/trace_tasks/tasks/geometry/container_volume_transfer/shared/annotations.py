"""Annotation helpers for container volume-transfer diagrams."""

from __future__ import annotations

from typing import Dict, Sequence

from trace_tasks.tasks.geometry.shared.measurement_rendering import bbox_to_list

from .state import RenderedScene


CONTAINER_BBOX_ANNOTATION_KEYS = (
    "source_container_bbox",
    "target_container_bbox",
)


def annotation_bbox_map(rendered: RenderedScene, keys: Sequence[str]) -> Dict[str, list[float]]:
    return {str(key): bbox_to_list(rendered.annotation_bboxes[str(key)]) for key in keys}


__all__ = ["CONTAINER_BBOX_ANNOTATION_KEYS", "annotation_bbox_map"]
