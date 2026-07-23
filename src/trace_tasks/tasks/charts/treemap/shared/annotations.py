"""Annotation projection helpers for treemap charts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from trace_tasks.core.types import TypedValue

from .state import RenderedTreemap


def leaf_cell_boxes(rendered: RenderedTreemap, leaf_ids: Sequence[str]) -> list[list[float]]:
    lookup = {
        str(leaf_id): [round(float(value), 3) for value in bbox]
        for leaf_id, bbox in rendered.annotation_bbox_by_leaf_id.items()
    }
    return [list(lookup[str(leaf_id)]) for leaf_id in leaf_ids if str(leaf_id) in lookup]


def bbox_set_projection(boxes: Sequence[Sequence[float]]) -> dict[str, Any]:
    values = [[round(float(value), 3) for value in box] for box in boxes]
    return {"type": "bbox_set", "bbox_set": values}


def annotation_value_from_projection(projected: Mapping[str, Any]) -> TypedValue:
    kind = str(projected["type"])
    if kind == "bbox_set":
        return TypedValue(type="bbox_set", value=list(projected["bbox_set"]))
    raise ValueError(f"unsupported treemap annotation type: {kind}")


__all__ = ["annotation_value_from_projection", "bbox_set_projection", "leaf_cell_boxes"]
