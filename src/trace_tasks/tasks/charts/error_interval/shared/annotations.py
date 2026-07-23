"""Annotation projection helpers for error-interval charts."""

from __future__ import annotations

from typing import Any, Dict

from trace_tasks.tasks.charts.error_interval.shared.state import _Dataset, _Rendered


def annotation_payload(
    *,
    dataset: _Dataset,
    rendered: _Rendered,
) -> tuple[str, Any, Dict[str, Any], list[dict[str, Any]]]:
    """Return annotation artifacts for task-selected interval marks."""

    item_by_id = {str(item.item_id): item for item in dataset.items}
    annotation_item_ids = [str(value) for value in dataset.query.annotation_item_ids]
    records = [
        {
            "item_id": str(item_id),
            "item_label": str(item_by_id[str(item_id)].label),
            "lower": int(item_by_id[str(item_id)].lower),
            "midpoint": int(item_by_id[str(item_id)].midpoint),
            "upper": int(item_by_id[str(item_id)].upper),
            "interval_width": int(item_by_id[str(item_id)].upper) - int(item_by_id[str(item_id)].lower),
            "interval_bbox_px": list(rendered.interval_bboxes_px[str(item_id)]),
            "interval_center_point_px": list(rendered.interval_center_points_px[str(item_id)]),
            "interval_segment_px": [list(point) for point in rendered.interval_segments_px[str(item_id)]],
        }
        for item_id in annotation_item_ids
    ]
    if str(dataset.query.annotation_type) == "point":
        annotation = [list(rendered.interval_center_points_px[str(item_id)]) for item_id in annotation_item_ids]
        if len(annotation) != 1:
            raise RuntimeError("error-interval scalar point annotation must contain exactly one point")
        point = list(annotation[0])
        projected = {
            "type": "point",
            "point": list(point),
            "pixel_point": list(point),
            "point_map": {str(item_id): list(rendered.interval_center_points_px[str(item_id)]) for item_id in annotation_item_ids},
            "item_ids": list(annotation_item_ids),
            "item_labels": [str(record["item_label"]) for record in records],
            "annotation_refs": [dict(record) for record in records],
        }
        return "point", list(point), dict(projected), [dict(record) for record in records]
    if str(dataset.query.annotation_type) == "point_set":
        annotation = [list(rendered.interval_center_points_px[str(item_id)]) for item_id in annotation_item_ids]
        projected = {
            "type": "point_set",
            "point_set": list(annotation),
            "pixel_point_set": list(annotation),
            "point_map": {str(item_id): list(rendered.interval_center_points_px[str(item_id)]) for item_id in annotation_item_ids},
            "item_ids": list(annotation_item_ids),
            "item_labels": [str(record["item_label"]) for record in records],
            "annotation_refs": [dict(record) for record in records],
        }
        return "point_set", list(annotation), dict(projected), [dict(record) for record in records]
    if str(dataset.query.annotation_type) == "segment":
        annotation = [[list(point) for point in rendered.interval_segments_px[str(item_id)]] for item_id in annotation_item_ids]
        if len(annotation) != 1:
            raise RuntimeError("error-interval scalar segment annotation must contain exactly one segment")
        segment = [list(point) for point in annotation[0]]
        projected = {
            "type": "segment",
            "segment": list(segment),
            "pixel_segment": list(segment),
            "segment_map": {str(item_id): [list(point) for point in rendered.interval_segments_px[str(item_id)]] for item_id in annotation_item_ids},
            "item_ids": list(annotation_item_ids),
            "item_labels": [str(record["item_label"]) for record in records],
            "annotation_refs": [dict(record) for record in records],
        }
        return "segment", list(segment), dict(projected), [dict(record) for record in records]
    if str(dataset.query.annotation_type) == "segment_set":
        annotation = [[list(point) for point in rendered.interval_segments_px[str(item_id)]] for item_id in annotation_item_ids]
        projected = {
            "type": "segment_set",
            "segment_set": list(annotation),
            "pixel_segment_set": list(annotation),
            "segment_map": {str(item_id): [list(point) for point in rendered.interval_segments_px[str(item_id)]] for item_id in annotation_item_ids},
            "item_ids": list(annotation_item_ids),
            "item_labels": [str(record["item_label"]) for record in records],
            "annotation_refs": [dict(record) for record in records],
        }
        return "segment_set", list(annotation), dict(projected), [dict(record) for record in records]
    raise ValueError(f"unsupported annotation type: {dataset.query.annotation_type}")


__all__ = ["annotation_payload"]
