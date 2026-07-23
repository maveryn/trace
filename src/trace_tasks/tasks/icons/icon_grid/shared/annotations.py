"""Annotation helpers for icon-grid scenes."""

from __future__ import annotations

from typing import Any, List, Mapping, Sequence, Tuple

from .state import IconGridScenePayload


def representative_cell_bboxes_by_field(
    scene_payload: IconGridScenePayload,
    *,
    field_name: str,
) -> Tuple[List[List[int]], List[int], List[str]]:
    """Return the first top-row/left-column cell for each field value."""

    def sort_key(item: tuple[int, Mapping[str, Any]]) -> tuple[int, int, int]:
        index, entity = item
        return (
            int(entity.get("row_index", 0)),
            int(entity.get("col_index", 0)),
            int(index),
        )

    selected: dict[str, tuple[int, Mapping[str, Any]]] = {}
    for index, entity in sorted(enumerate(scene_payload.scene_instances), key=sort_key):
        key = str(entity.get(str(field_name)))
        if key not in selected:
            selected[key] = (int(index), entity)

    ordered_items = sorted(selected.items(), key=lambda item: sort_key(item[1]))
    bboxes: List[List[int]] = []
    indices: List[int] = []
    keys: List[str] = []
    for key, (index, entity) in ordered_items:
        bbox = entity.get("cell_bbox_xyxy")
        if not isinstance(bbox, Sequence) or len(bbox) != 4:
            raise RuntimeError(f"invalid representative cell bbox for {field_name}={key}: {bbox}")
        keys.append(str(key))
        indices.append(int(index))
        bboxes.append([int(round(float(value))) for value in bbox])
    return bboxes, indices, keys


__all__ = ["representative_cell_bboxes_by_field"]
