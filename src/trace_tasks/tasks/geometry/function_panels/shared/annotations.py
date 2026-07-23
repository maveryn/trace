"""Annotation primitives for function-panel scenes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from trace_tasks.core.types import TypedValue

from .state import RenderedIntersectionScene, RenderedPropertyScene


@dataclass(frozen=True)
class PanelAnnotationArtifacts:
    """Bound annotation value plus projected trace payload."""

    annotation_gt: TypedValue
    projected_annotation: Mapping[str, Any]
    witness_symbolic: Mapping[str, Any]


def selected_panel_bbox_annotation(rendered: RenderedPropertyScene, *, label: str) -> PanelAnnotationArtifacts:
    """Return scalar bbox annotation for one selected panel."""

    bbox = [int(value) for value in rendered.panel_bboxes[str(label)]]
    return PanelAnnotationArtifacts(
        annotation_gt=TypedValue(type="bbox", value=bbox),
        projected_annotation={
            "bbox": list(bbox),
            "panel_bbox_by_label": {str(key): list(value) for key, value in rendered.panel_bboxes.items()},
            "plot_bbox_by_label": {str(key): list(value) for key, value in rendered.plot_bboxes.items()},
        },
        witness_symbolic={"selected_panel_label": str(label), "annotation_kind": "selected_panel_bbox"},
    )


def intersection_panel_annotation(rendered: RenderedIntersectionScene, *, label: str) -> PanelAnnotationArtifacts:
    """Return scalar bbox annotation for the selected intersection panel."""

    panel_bbox = [int(value) for value in rendered.panel_bboxes[str(label)]]
    point_bboxes = [[int(coord) for coord in bbox] for bbox in rendered.intersection_point_bboxes[str(label)]]
    return PanelAnnotationArtifacts(
        annotation_gt=TypedValue(type="bbox", value=panel_bbox),
        projected_annotation={
            "bbox": list(panel_bbox),
            "panel_bbox_by_label": {str(key): list(item) for key, item in rendered.panel_bboxes.items()},
            "plot_bbox_by_label": {str(key): list(item) for key, item in rendered.plot_bboxes.items()},
            "intersection_point_bboxes_by_label": {
                str(key): [[int(coord) for coord in bbox] for bbox in item]
                for key, item in rendered.intersection_point_bboxes.items()
            },
        },
        witness_symbolic={
            "selected_panel_label": str(label),
            "intersection_point_count": int(len(point_bboxes)),
            "selected_panel_intersection_point_bboxes": point_bboxes,
            "annotation_kind": "selected_panel_bbox",
        },
    )
