"""Annotation projection helpers for pictogram scenes."""

from __future__ import annotations

from typing import Any

from .rendering import PictogramRenderResult
from .state import PictogramDataset


def annotation_payload(*, dataset: PictogramDataset, rendered: PictogramRenderResult) -> dict[str, Any]:
    """Project task-bound category row witnesses into the task's annotation schema."""

    category_ids = [str(value) for value in dataset.query.annotation_category_ids]
    category_id_to_label = {category.category_id: str(category.label) for category in dataset.categories}
    labels = [str(category_id_to_label[category_id]) for category_id in category_ids]
    boxes = [list(rendered.rendered_scene.category_bboxes_px[category_id]) for category_id in category_ids]
    annotation_type = str(dataset.query.annotation_type)

    if annotation_type == "bbox":
        if len(boxes) != 1:
            raise ValueError("bbox annotation requires exactly one category witness")
        value = list(boxes[0])
        projected = {
            "type": "bbox",
            "bbox": list(value),
            "pixel_bbox": list(value),
            "bbox_set": [list(value)],
            "category_ids": list(category_ids),
            "category_labels": list(labels),
        }
    elif annotation_type == "bbox_map":
        keyed = {str(label): list(box) for label, box in zip(labels, boxes, strict=True)}
        value = dict(keyed)
        projected = {
            "type": "bbox_map",
            "bbox_map": dict(keyed),
            "pixel_bbox_map": dict(keyed),
            "bbox_set": list(keyed.values()),
            "category_ids": list(category_ids),
            "category_labels": list(keyed.keys()),
        }
    elif annotation_type == "bbox_set":
        value = list(boxes)
        projected = {
            "type": "bbox_set",
            "bbox_set": list(boxes),
            "category_ids": list(category_ids),
            "category_labels": list(labels),
        }
    else:
        raise ValueError(f"unsupported pictogram annotation type: {annotation_type}")

    return {
        "type": annotation_type,
        "value": value,
        "projected_annotation": dict(projected),
        "category_ids": list(category_ids),
        "labels": list(labels),
        "category_id_to_label": dict(category_id_to_label),
    }
