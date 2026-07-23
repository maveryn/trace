"""Annotation projection helpers for toggle-grid answer witnesses."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

from trace_tasks.tasks.shared.annotation_artifacts import bbox_annotation_artifacts


def option_panel_bbox_annotation(
    option_panel_bbox_map: Mapping[str, Sequence[float]],
    label: str,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Project one selected result option panel to scalar bbox annotation."""

    item_id = f"option_{str(label)}"
    if item_id not in option_panel_bbox_map:
        raise RuntimeError(f"missing toggle result option bbox: {item_id!r}")
    artifacts = bbox_annotation_artifacts(option_panel_bbox_map[item_id])
    witness = {"type": "bbox", "item_id": item_id, "label": str(label)}
    return artifacts.annotation_gt, artifacts.projected_annotation, witness


def switch_cell_bbox_annotation(
    cell_bbox_map: Mapping[str, Sequence[float]],
    *,
    row: int,
    col: int,
    label: str,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Project one selected switch cell to scalar bbox annotation."""

    item_id = f"cell_{int(row)}_{int(col)}"
    if item_id not in cell_bbox_map:
        raise RuntimeError(f"missing toggle switch cell bbox: {item_id!r}")
    artifacts = bbox_annotation_artifacts(cell_bbox_map[item_id])
    witness = {
        "type": "bbox",
        "item_id": item_id,
        "label": str(label),
        "row": int(row),
        "col": int(col),
    }
    return artifacts.annotation_gt, artifacts.projected_annotation, witness
