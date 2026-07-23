"""Neutral annotation projection helpers for grid-like chart scenes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.core.types import TypedValue

from .geometry import bbox_union, round_bbox


def bbox_projection(box: Sequence[float]) -> dict[str, Any]:
    """Return projected annotation payload for one bbox."""

    value = round_bbox(box)
    return {"type": "bbox", "bbox": value}


def bbox_set_projection(boxes: Sequence[Sequence[float]]) -> dict[str, Any]:
    """Return projected annotation payload for an unordered bbox set."""

    values = [round_bbox(box) for box in boxes]
    return {"type": "bbox_set", "bbox_set": values}


def bbox_map_projection(box_map: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Return projected annotation payload for keyed scalar bboxes."""

    value = {str(key): round_bbox(box) for key, box in box_map.items()}
    return {"type": "bbox_map", "bbox_map": value, "pixel_bbox_map": dict(value)}


def bbox_set_map_projection(box_map: Mapping[str, Sequence[Sequence[float]]]) -> dict[str, Any]:
    """Return projected annotation payload for keyed bbox sets."""

    value = {
        str(key): [round_bbox(box) for box in boxes]
        for key, boxes in box_map.items()
    }
    return {"type": "bbox_set_map", "bbox_set_map": value}


def annotation_value_from_projection(projected: Mapping[str, Any]) -> TypedValue:
    """Build the verifier annotation value from one projected annotation."""

    kind = str(projected["type"])
    if kind == "bbox":
        return TypedValue(type="bbox", value=list(projected["bbox"]))
    if kind == "bbox_set":
        return TypedValue(type="bbox_set", value=list(projected["bbox_set"]))
    if kind == "bbox_map":
        return TypedValue(type="bbox_map", value=dict(projected["bbox_map"]))
    if kind == "bbox_set_map":
        return TypedValue(type="bbox_set_map", value=dict(projected["bbox_set_map"]))
    raise ValueError(f"unsupported grid annotation type: {kind}")


def bbox_refs(
    *,
    ids: Sequence[str],
    boxes: Sequence[Sequence[float]],
    role: str = "cell",
    id_key: str = "id",
    bbox_key: str = "bbox",
) -> list[dict[str, Any]]:
    """Return symbolic-to-pixel bbox references in the requested key shape."""

    return [
        {
            "role": str(role),
            str(id_key): str(item_id),
            str(bbox_key): round_bbox(box),
        }
        for item_id, box in zip(ids, boxes)
    ]


def union_box_map(cell_id_map: Mapping[str, Sequence[str]], cell_bbox_map: Mapping[str, Sequence[float]]) -> dict[str, list[float]]:
    """Return a keyed map of bbox unions from keyed cell-id groups."""

    return {
        str(key): bbox_union([cell_bbox_map[str(cell_id)] for cell_id in cell_ids])
        for key, cell_ids in cell_id_map.items()
    }


__all__ = [
    "annotation_value_from_projection",
    "bbox_map_projection",
    "bbox_projection",
    "bbox_refs",
    "bbox_set_map_projection",
    "bbox_set_projection",
    "union_box_map",
]
