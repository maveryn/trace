"""Annotation projection helpers for size-encoded chart tasks."""

from __future__ import annotations

from typing import List, Mapping, Sequence

from .state import RenderedSizeEncodingScene


def item_bbox(rendered: RenderedSizeEncodingScene, item_id: str) -> list[float]:
    bbox = rendered.item_bboxes.get(str(item_id))
    if bbox is None:
        raise RuntimeError(f"missing item bbox: {item_id}")
    return [float(value) for value in bbox]


def item_bbox_set(rendered: RenderedSizeEncodingScene, item_ids: tuple[str, ...]) -> list[list[float]]:
    boxes: List[List[float]] = []
    for item_id in item_ids:
        boxes.append(item_bbox(rendered, str(item_id)))
    if not boxes:
        raise RuntimeError("empty size-encoding annotation item set")
    return boxes


def reference_answer_bbox_map(
    rendered: RenderedSizeEncodingScene,
    *,
    reference_item_id: str,
    answer_item_id: str,
) -> dict[str, list[float]]:
    return {
        "reference_item": item_bbox(rendered, str(reference_item_id)),
        "answer_item": item_bbox(rendered, str(answer_item_id)),
    }


def item_bbox_set_map(
    rendered: RenderedSizeEncodingScene,
    groups: Mapping[str, Sequence[str]],
) -> tuple[str, dict[str, list[list[float]]], dict[str, object]]:
    value = {str(key): item_bbox_set(rendered, tuple(str(item_id) for item_id in item_ids)) for key, item_ids in groups.items()}
    return "bbox_set_map", value, {"type": "bbox_set_map", "bbox_set_map": value, "pixel_bbox_set_map": value}
