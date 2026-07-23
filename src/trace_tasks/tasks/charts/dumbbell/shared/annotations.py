"""Annotation projection helpers for dumbbell charts."""

from __future__ import annotations

from typing import Any

from trace_tasks.tasks.charts.dumbbell.shared.state import DumbbellDataset, RenderedDumbbell


def _bbox_center(bbox: list[float]) -> list[float]:
    """Return the rounded center point of a pixel bbox."""

    return [
        round((float(bbox[0]) + float(bbox[2])) / 2.0, 3),
        round((float(bbox[1]) + float(bbox[3])) / 2.0, 3),
    ]


def _segment_annotation_payload(
    *,
    dataset: DumbbellDataset,
    rendered: RenderedDumbbell,
    scalar: bool,
) -> tuple[str, Any, dict[str, Any], list[dict[str, Any]]]:
    """Project task-bound rows into line segments between their two colored dots."""

    rows_by_id = {str(row.row_id): row for row in dataset.rows}
    annotation_rows = [str(row_id) for row_id in dataset.query.annotation_row_ids]
    records: list[dict[str, Any]] = []
    annotation: list[list[list[float]]] = []
    for row_id in annotation_rows:
        point_a_bbox = list(rendered.point_bboxes_px[f"{row_id}:series_a"])
        point_b_bbox = list(rendered.point_bboxes_px[f"{row_id}:series_b"])
        point_pair = [_bbox_center(point_a_bbox), _bbox_center(point_b_bbox)]
        annotation.append(point_pair)
        records.append(
            {
                "row_id": str(row_id),
                "row_label": str(rows_by_id[str(row_id)].label),
                "value_a": int(rows_by_id[str(row_id)].value_a),
                "value_b": int(rows_by_id[str(row_id)].value_b),
                "gap": int(rows_by_id[str(row_id)].gap),
                "row_pair_bbox_px": list(rendered.row_pair_bboxes_px[str(row_id)]),
                "connector_bbox_px": list(rendered.connector_bboxes_px[str(row_id)]),
                "point_a_px": list(point_pair[0]),
                "point_b_px": list(point_pair[1]),
                "point_a_bbox_px": list(point_a_bbox),
                "point_b_bbox_px": list(point_b_bbox),
            }
        )
    if scalar:
        if len(annotation) != 1:
            raise ValueError("scalar dumbbell segment annotation requires exactly one row")
        scalar_annotation = [list(point) for point in annotation[0]]
        projected = {
            "type": "segment",
            "segment": [list(point) for point in scalar_annotation],
            "pixel_segment": [list(point) for point in scalar_annotation],
            "row_ids": list(annotation_rows),
            "annotation_refs": [dict(record) for record in records],
        }
        return "segment", [list(point) for point in scalar_annotation], dict(projected), [dict(record) for record in records]
    projected = {
        "type": "segment_set",
        "segment_set": [[list(point) for point in pair] for pair in annotation],
        "pixel_segment_set": [[list(point) for point in pair] for pair in annotation],
        "row_ids": list(annotation_rows),
        "annotation_refs": [dict(record) for record in records],
    }
    return "segment_set", [[list(point) for point in pair] for pair in annotation], dict(projected), [dict(record) for record in records]


def annotation_payload(
    *,
    dataset: DumbbellDataset,
    rendered: RenderedDumbbell,
    annotation_style: str = "row_pair_segment_set",
) -> tuple[str, Any, dict[str, Any], list[dict[str, Any]]]:
    """Project task-bound row annotations into the requested public annotation payload."""

    if str(annotation_style) == "row_pair_segment":
        return _segment_annotation_payload(dataset=dataset, rendered=rendered, scalar=True)
    if str(annotation_style) == "row_pair_segment_set":
        return _segment_annotation_payload(dataset=dataset, rendered=rendered, scalar=False)
    if str(annotation_style) != "row_pair_bbox_set":
        raise ValueError(f"unsupported dumbbell annotation style: {annotation_style}")

    rows_by_id = {str(row.row_id): row for row in dataset.rows}
    annotation_rows = [str(row_id) for row_id in dataset.query.annotation_row_ids]
    annotation = [list(rendered.row_pair_bboxes_px[str(row_id)]) for row_id in annotation_rows]
    records = [
        {
            "row_id": str(row_id),
            "row_label": str(rows_by_id[str(row_id)].label),
            "value_a": int(rows_by_id[str(row_id)].value_a),
            "value_b": int(rows_by_id[str(row_id)].value_b),
            "gap": int(rows_by_id[str(row_id)].gap),
            "row_pair_bbox_px": list(rendered.row_pair_bboxes_px[str(row_id)]),
            "connector_bbox_px": list(rendered.connector_bboxes_px[str(row_id)]),
            "point_a_bbox_px": list(rendered.point_bboxes_px[f"{row_id}:series_a"]),
            "point_b_bbox_px": list(rendered.point_bboxes_px[f"{row_id}:series_b"]),
        }
        for row_id in annotation_rows
    ]
    projected = {
        "type": "bbox_set",
        "bbox_set": list(annotation),
        "pixel_bbox_set": list(annotation),
        "row_ids": list(annotation_rows),
        "annotation_refs": [dict(record) for record in records],
    }
    return "bbox_set", list(annotation), dict(projected), [dict(record) for record in records]


__all__ = ["annotation_payload"]
