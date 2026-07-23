"""Annotation and trace helpers for record-table pages."""

from __future__ import annotations

from typing import Any, Dict, List

from .rendering import RecordTableCase, RenderedRecordTableBundle


def counted_row_bboxes(case: RecordTableCase, rendered: RenderedRecordTableBundle) -> List[List[float]]:
    """Return full row boxes for every counted table row."""

    table = rendered.rendered_table
    return [list(table.row_bboxes_by_id[str(row_id)]) for row_id in case.annotation_row_ids]


def section_records(case: RecordTableCase, rendered: RenderedRecordTableBundle) -> List[Dict[str, Any]]:
    """Return visible section membership and boxes for trace metadata."""

    table = rendered.rendered_table
    records: List[Dict[str, Any]] = []
    for section_index, section_name in enumerate(case.section_names):
        records.append(
            {
                "section_name": str(section_name),
                "section_index": int(section_index),
                "row_ids": [
                    str(row.row_id)
                    for row in case.rows
                    if int(row.section_index) == int(section_index)
                ],
                "bbox_px": list(table.section_bboxes_by_name[str(section_name)]),
            }
        )
    return records


def matching_row_records(case: RecordTableCase, rendered: RenderedRecordTableBundle) -> List[Dict[str, Any]]:
    """Return rendered row records that satisfy the task predicate."""

    selected = {str(value) for value in case.annotation_row_ids}
    return [
        dict(record)
        for record in rendered.rendered_table.row_records
        if str(record["row_id"]) in selected
    ]


def row_entities(rendered: RenderedRecordTableBundle) -> List[Dict[str, Any]]:
    """Return scene-IR entities for every visible table row."""

    entities: List[Dict[str, Any]] = []
    for record in rendered.rendered_table.row_records:
        entities.append(
            {
                "entity_id": str(record["row_id"]),
                "entity_type": "gui_table_row",
                "attrs": {
                    "row_label": str(record["row_label"]),
                    "section_name": str(record["section_name"]),
                    "section_index": int(record["section_index"]),
                    "order_in_section": int(record["order_in_section"]),
                    "global_order_index": int(record["global_order_index"]),
                    "item_name": str(record["item_name"]),
                    "type_label": str(record["type_label"]),
                    "status_label": str(record["status_label"]),
                    "size_mb": int(record["size_mb"]),
                    "selected": bool(record["selected"]),
                    "action_label": str(record["action_label"]),
                    "action_enabled": bool(record["action_enabled"]),
                    "bbox_px": list(record["bbox_px"]),
                },
            }
        )
    return entities
