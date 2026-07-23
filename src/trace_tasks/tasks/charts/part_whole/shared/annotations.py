"""Annotation projection helpers for part-whole chart tasks."""

from __future__ import annotations

from typing import Any

from .state import PartWholeDataset, RenderedShareChart


def point_map_annotation(
    *,
    dataset: PartWholeDataset,
    rendered_scene: RenderedShareChart,
) -> dict[str, Any]:
    """Project selected category witnesses to point-map annotation artifacts."""

    values_by_label = {str(category.label): int(category.value) for category in dataset.categories}
    annotation_values = [
        int(values_by_label[str(label)])
        for label in dataset.annotation_labels
    ]
    annotation_bboxes = [
        list(rendered_scene.annotation_bbox_by_label[str(label)])
        for label in dataset.annotation_labels
        if str(label) in rendered_scene.annotation_bbox_by_label
    ]
    annotation_points = [
        list(rendered_scene.annotation_point_by_label[str(label)])
        for label in dataset.annotation_labels
        if str(label) in rendered_scene.annotation_point_by_label
    ]
    annotation_keyed_points = {
        str(label): list(rendered_scene.annotation_point_by_label[str(label)])
        for label in dataset.annotation_labels
        if str(label) in rendered_scene.annotation_point_by_label
    }
    annotation_keys = [str(label) for label in dataset.annotation_labels]
    projected = {
        "type": "point_map",
        "point_map": dict(annotation_keyed_points),
        "pixel_point_map": dict(annotation_keyed_points),
        "point_set": list(annotation_points),
        "pixel_point_set": list(annotation_points),
        "bbox_set": list(annotation_bboxes),
        "annotation_labels": [str(label) for label in dataset.annotation_labels],
        "annotation_keys": [str(key) for key in annotation_keys],
    }
    return {
        "values": [int(value) for value in annotation_values],
        "bboxes": list(annotation_bboxes),
        "points": list(annotation_points),
        "point_map": dict(annotation_keyed_points),
        "keys": [str(key) for key in annotation_keys],
        "projected": dict(projected),
    }
