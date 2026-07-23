"""Annotation helpers for form-section page tasks."""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence


OPERAND_ROLE_KEYS: tuple[str, ...] = ("first_operand", "second_operand", "third_operand")


def operand_role_to_field_id(operand_field_ids: Sequence[str]) -> Dict[str, str]:
    """Return role-keyed operand field ids in left-to-right expression order."""

    if len(operand_field_ids) > len(OPERAND_ROLE_KEYS):
        raise ValueError("form-section arithmetic supports at most three operand annotation roles")
    return {
        str(OPERAND_ROLE_KEYS[index]): str(field_id)
        for index, field_id in enumerate(operand_field_ids)
    }


def project_operand_field_bbox_map(
    field_box_bbox_map: Mapping[str, Sequence[float]],
    role_to_field_id: Mapping[str, str],
) -> Dict[str, List[float]]:
    """Project role-keyed operand field ids onto rendered full-field pixel boxes."""

    annotation: Dict[str, List[float]] = {}
    missing: list[str] = []
    for role, field_id in role_to_field_id.items():
        bbox = field_box_bbox_map.get(str(field_id))
        if bbox is None:
            missing.append(str(field_id))
            continue
        annotation[str(role)] = [round(float(value), 3) for value in bbox]
    if missing:
        raise RuntimeError(f"missing operand field ids: {missing}")
    return annotation


def operand_records(case: Mapping[str, object], annotation_bboxes: Mapping[str, Sequence[float]]) -> List[Dict[str, object]]:
    """Return operand records with semantic role, field id, value, and bbox."""

    records: List[Dict[str, object]] = []
    operand_field_ids = [str(value) for value in case["operand_field_ids"]]  # type: ignore[index]
    operand_field_labels = [str(value) for value in case["operand_field_labels"]]  # type: ignore[index]
    operand_field_values = [str(value) for value in case["operand_field_values"]]  # type: ignore[index]
    for index, role in enumerate(list(annotation_bboxes.keys())):
        records.append(
            {
                "role": str(role),
                "field_id": str(operand_field_ids[index]),
                "field_label": str(operand_field_labels[index]),
                "field_value": str(operand_field_values[index]),
                "bbox_px": [float(value) for value in annotation_bboxes[str(role)]],
            }
        )
    return records


__all__ = [
    "OPERAND_ROLE_KEYS",
    "operand_records",
    "operand_role_to_field_id",
    "project_operand_field_bbox_map",
]
