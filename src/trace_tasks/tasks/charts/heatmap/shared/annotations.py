"""Annotation projection helpers for heatmap chart scenes."""

from __future__ import annotations

from typing import Any, Mapping

from trace_tasks.tasks.charts.shared.grid.geometry import bboxes_for_ids


def annotation_payload(
    *,
    annotation_type: str,
    annotation_cell_ids: list[str],
    rendered: Any,
) -> tuple[str, Any, dict[str, Any]]:
    """Project task-bound symbolic cell ids into pixel bbox annotations."""

    cell_ids = [str(cell_id) for cell_id in annotation_cell_ids]
    annotation_bboxes = bboxes_for_ids(rendered.cell_bbox_map, cell_ids, missing="error")
    if str(annotation_type) == "bbox":
        if not annotation_bboxes:
            raise ValueError("scalar heatmap bbox annotation requires one projected cell bbox")
        bbox = list(annotation_bboxes[0])
        projected_annotation = {
            "type": "bbox",
            "bbox": list(bbox),
            "pixel_bbox": list(bbox),
            "cell_id": str(cell_ids[0]) if cell_ids else "",
            "cell_ids": list(cell_ids),
        }
        return "bbox", list(bbox), dict(projected_annotation)
    if str(annotation_type) != "bbox_set":
        raise ValueError(f"unsupported heatmap annotation type: {annotation_type}")

    projected_annotation = {
        "type": "bbox_set",
        "bbox_set": list(annotation_bboxes),
        "pixel_bbox_set": list(annotation_bboxes),
        "bbox_map": {str(cell_id): list(bbox) for cell_id, bbox in zip(cell_ids, annotation_bboxes)},
        "cell_ids": list(cell_ids),
    }
    return "bbox_set", list(annotation_bboxes), dict(projected_annotation)


def annotation_refs(
    *,
    annotation_cell_ids: list[str],
    annotation_value: Any,
    annotation_type: str,
) -> list[dict[str, Any]]:
    """Return symbolic-to-pixel annotation references for trace payloads."""

    if str(annotation_type) == "bbox":
        return [
            {"cell_id": str(annotation_cell_ids[0]), "bbox_xyxy": list(annotation_value)}
        ] if annotation_cell_ids else []

    return [
        {"cell_id": str(cell_id), "bbox_xyxy": list(bbox)}
        for cell_id, bbox in zip(annotation_cell_ids, annotation_value)
    ]


def annotation_cell_ids_from_dataset(dataset: Mapping[str, Any]) -> list[str]:
    """Read the symbolic annotation cells chosen by the public task."""

    return [str(cell_id) for cell_id in dataset["annotation_cell_ids"]]


__all__ = ["annotation_cell_ids_from_dataset", "annotation_payload", "annotation_refs"]
