"""Annotation projection helpers for the named-grid icons scene."""

from __future__ import annotations

from typing import Dict, List, Tuple

from .state import NamedGridScenePayload


def line_region_bbox(scene: NamedGridScenePayload, *, axis: str, line_index: int) -> Tuple[int, int, int, int]:
    if str(axis) == "row":
        boxes = tuple(scene.cell_bboxes_xyxy[int(line_index)])
        return (
            min(int(box[0]) for box in boxes),
            min(int(box[1]) for box in boxes),
            max(int(box[2]) for box in boxes),
            max(int(box[3]) for box in boxes),
        )
    boxes = tuple(row[int(line_index)] for row in scene.cell_bboxes_xyxy)
    return (
        min(int(box[0]) for box in boxes),
        min(int(box[1]) for box in boxes),
        max(int(box[2]) for box in boxes),
        max(int(box[3]) for box in boxes),
    )


def all_line_region_bboxes(scene: NamedGridScenePayload, *, axis: str) -> Dict[str, List[int]]:
    if str(axis) == "row":
        return {
            f"row_{int(index) + 1}": [int(value) for value in line_region_bbox(scene, axis="row", line_index=int(index))]
            for index in range(len(scene.cell_bboxes_xyxy))
        }
    column_count = len(scene.cell_bboxes_xyxy[0]) if scene.cell_bboxes_xyxy else 0
    return {
        f"column_{int(index) + 1}": [int(value) for value in line_region_bbox(scene, axis="column", line_index=int(index))]
        for index in range(int(column_count))
    }
