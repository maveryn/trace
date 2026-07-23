"""Annotation and trace payload helpers for control-board pages."""

from __future__ import annotations

from typing import Any, Dict, List

from .state import ControlBoardCase, RenderedControlBoard


def counted_control_bboxes(case: ControlBoardCase, rendered: RenderedControlBoard) -> List[List[float]]:
    """Return full control boxes for every counted witness."""

    return [
        list(rendered.control_bboxes_by_id[str(control_id)])
        for control_id in case.annotation_control_ids
    ]


def target_group_bbox(case: ControlBoardCase, rendered: RenderedControlBoard) -> List[float]:
    """Return the full group panel box for the selected target group."""

    return list(rendered.group_bboxes_by_name[str(case.target_group_name)])


def group_records(case: ControlBoardCase, rendered: RenderedControlBoard) -> List[Dict[str, Any]]:
    """Return group membership and pixel boxes for trace metadata."""

    records: List[Dict[str, Any]] = []
    for group_index, group_name in enumerate(case.group_names):
        members = [
            str(control.control_id)
            for control in case.controls
            if int(control.group_index) == int(group_index)
        ]
        records.append(
            {
                "group_name": str(group_name),
                "group_index": int(group_index),
                "control_ids": list(members),
                "bbox_px": list(rendered.group_bboxes_by_name[str(group_name)]),
            }
        )
    return records


def matching_control_records(case: ControlBoardCase, rendered: RenderedControlBoard) -> List[Dict[str, Any]]:
    """Return rendered control records that satisfy the task predicate."""

    selected = {str(value) for value in case.annotation_control_ids}
    return [
        dict(record)
        for record in rendered.control_records
        if str(record["control_id"]) in selected
    ]


def control_entities(rendered: RenderedControlBoard) -> List[Dict[str, Any]]:
    """Return scene-IR entities for every visible control tile."""

    entities: List[Dict[str, Any]] = []
    for record in rendered.control_records:
        entities.append(
            {
                "entity_id": str(record["control_id"]),
                "entity_type": "gui_control",
                "attrs": {
                    "candidate_label": str(record["candidate_label"]),
                    "group_name": str(record["group_name"]),
                    "group_index": int(record["group_index"]),
                    "order_in_group": int(record["order_in_group"]),
                    "global_order_index": int(record["global_order_index"]),
                    "command_key": str(record["command_key"]),
                    "display_text": str(record["display_text"]),
                    "icon_kind": str(record["icon_kind"]),
                    "enabled": bool(record["enabled"]),
                    "selected": bool(record["selected"]),
                    "is_reference": bool(record["is_reference"]),
                    "bbox_px": list(record["bbox_px"]),
                },
            }
        )
    return entities
