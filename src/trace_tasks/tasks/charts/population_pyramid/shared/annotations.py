"""Annotation projection helpers for population-pyramid scenes."""

from __future__ import annotations

from typing import Any

from .rendering import PopulationPyramidRenderResult
from .state import PopulationPyramidDataset


def annotation_payload(*, dataset: PopulationPyramidDataset, rendered: PopulationPyramidRenderResult) -> dict[str, Any]:
    """Project bound row ids onto the task-selected bar scope without changing the answer trace."""

    row_ids = [str(row_id) for row_id in dataset.query.annotation_row_ids]
    bar_scope = str(dataset.query.params.get("annotation_bar_scope", "row"))
    if bar_scope == "left":
        box_by_row_id = rendered.rendered_scene.left_bar_bboxes_px
    elif bar_scope == "right":
        box_by_row_id = rendered.rendered_scene.right_bar_bboxes_px
    elif bar_scope == "row":
        box_by_row_id = rendered.rendered_scene.row_bar_bboxes_px
    else:
        raise ValueError(f"unsupported population-pyramid annotation bar scope: {bar_scope}")
    boxes = [list(box_by_row_id[str(row_id)]) for row_id in row_ids]
    row_id_to_label = {str(row.row_id): str(row.label) for row in dataset.rows}
    labels = [str(row_id_to_label[row_id]) for row_id in row_ids]
    annotation_type = str(dataset.query.annotation_type)
    if annotation_type == "bbox":
        if len(boxes) != 1:
            raise ValueError("bbox annotation requires exactly one population-pyramid row")
        value = list(boxes[0])
        projected_annotation = {
            "type": "bbox",
            "bbox": list(value),
            "pixel_bbox": list(value),
            "row_ids": list(row_ids),
            "row_labels": list(labels),
            "annotation_bar_scope": str(bar_scope),
        }
    elif annotation_type == "bbox_set":
        value = [list(box) for box in boxes]
        projected_annotation = {
            "type": "bbox_set",
            "bbox_set": [list(box) for box in boxes],
            "pixel_bbox_set": [list(box) for box in boxes],
            "row_ids": list(row_ids),
            "row_labels": list(labels),
            "annotation_bar_scope": str(bar_scope),
        }
    else:
        raise ValueError(f"unsupported population-pyramid annotation type: {annotation_type}")
    return {
        "type": str(annotation_type),
        "value": value,
        "projected_annotation": dict(projected_annotation),
        "row_ids": list(row_ids),
        "row_labels": list(labels),
    }
