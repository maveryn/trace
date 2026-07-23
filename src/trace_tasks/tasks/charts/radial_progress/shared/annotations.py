"""Annotation projection helpers for radial-progress charts."""

from __future__ import annotations

from typing import Any

from .state import ProgressDataset, RadialProgressRenderResult


def annotation_payload(*, dataset: ProgressDataset, rendered: RadialProgressRenderResult) -> dict[str, Any]:
    """Project task-owned item ids into the public annotation contract."""

    rendered_scene = rendered.rendered_scene
    item_ids = [str(value) for value in dataset.question.annotation_item_ids]
    item_by_id = {str(item.item_id): item for item in dataset.items}
    labels = [str(item_by_id[item_id].label) for item_id in item_ids]
    boxes = [list(rendered_scene.item_bboxes_px[str(item_id)]) for item_id in item_ids]
    annotation_type = str(dataset.question.annotation_type)
    if annotation_type == "bbox":
        if len(boxes) != 1:
            raise ValueError("scalar bbox annotation requires exactly one radial-progress item")
        box = list(boxes[0])
        return {
            "type": "bbox",
            "value": list(box),
            "item_ids": list(item_ids),
            "labels": list(labels),
            "projected_annotation": {
                "type": "bbox",
                "bbox": list(box),
                "pixel_bbox": list(box),
                "item_ids": list(item_ids),
                "item_labels": list(labels),
            },
        }
    if annotation_type == "bbox_set":
        return {
            "type": "bbox_set",
            "value": [list(box) for box in boxes],
            "item_ids": list(item_ids),
            "labels": list(labels),
            "projected_annotation": {
                "type": "bbox_set",
                "bbox_set": [list(box) for box in boxes],
                "pixel_bbox_set": [list(box) for box in boxes],
                "item_ids": list(item_ids),
                "item_labels": list(labels),
            },
        }
    raise ValueError(f"unsupported radial-progress annotation type: {annotation_type}")
