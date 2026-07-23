"""Annotation projection helpers for radar chart scenes."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .state import Point, RadarDataset, RadarRenderResult, RenderedRadarScene


def point_center_from_bbox(bbox: Sequence[float]) -> Point:
    if len(bbox) != 4:
        raise ValueError(f"expected bbox with 4 values, got {bbox}")
    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 2),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 2),
    ]


def point_from_vertex(rendered_scene: RenderedRadarScene, point_id: str) -> Point:
    box = rendered_scene.point_bboxes.get(str(point_id))
    if box is None:
        raise KeyError(f"missing radar point bbox for {point_id}")
    return point_center_from_bbox(box)


def annotation_payload(*, dataset: RadarDataset, rendered: RadarRenderResult) -> dict[str, Any]:
    """Project task-owned annotation ids into the public annotation contract."""

    annotation_type = str(dataset.query.annotation_type)
    rendered_scene = rendered.rendered_scene
    if annotation_type == "bbox_set":
        panel_labels = [str(label) for label in dataset.query.annotation_panel_labels]
        boxes = [list(rendered_scene.panel_bboxes[str(label)]) for label in panel_labels]
        return {
            "type": "bbox_set",
            "value": [list(box) for box in boxes],
            "projected_annotation": {
                "type": "bbox_set",
                "bbox_set": [list(box) for box in boxes],
                "pixel_bbox_set": [list(box) for box in boxes],
                "annotation_panel_labels": list(panel_labels),
                "annotation_point_ids": list(dataset.query.annotation_point_ids),
            },
        }
    if annotation_type == "point_set":
        point_ids = [str(point_id) for point_id in dataset.query.annotation_point_ids]
        points = [point_from_vertex(rendered_scene, str(point_id)) for point_id in point_ids]
        return {
            "type": "point_set",
            "value": [list(point) for point in points],
            "projected_annotation": {
                "type": "point_set",
                "point_set": [list(point) for point in points],
                "pixel_point_set": [list(point) for point in points],
                "annotation_point_ids": list(point_ids),
            },
        }
    if annotation_type == "segment_set":
        pairs = [
            [
                point_from_vertex(rendered_scene, str(start_id)),
                point_from_vertex(rendered_scene, str(end_id)),
            ]
            for start_id, end_id in dataset.query.annotation_point_id_pairs
        ]
        return {
            "type": "segment_set",
            "value": [[list(start), list(end)] for start, end in pairs],
            "projected_annotation": {
                "type": "segment_set",
                "segment_set": [[list(start), list(end)] for start, end in pairs],
                "pixel_segment_set": [[list(start), list(end)] for start, end in pairs],
                "annotation_point_id_pairs": [
                    [str(start_id), str(end_id)]
                    for start_id, end_id in dataset.query.annotation_point_id_pairs
                ],
            },
        }
    raise ValueError(f"unsupported radar annotation type: {annotation_type}")
