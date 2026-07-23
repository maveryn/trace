"""Annotation projection helpers for cards scene tasks."""

from __future__ import annotations

from typing import Mapping, Sequence


def card_bboxes_for_ids(render_map: Mapping[str, object], card_ids: Sequence[str]) -> list[list[float]]:
    """Return card bounding boxes for stable rendered card ids."""

    boxes = render_map.get("card_bboxes_px", {})
    if not isinstance(boxes, Mapping):
        raise ValueError("cards render map is missing card_bboxes_px")
    return [list(boxes[str(card_id)]) for card_id in card_ids]


def keyed_card_bbox_set_map(render_map: Mapping[str, object], keyed_card_ids: Sequence[tuple[str, Sequence[str]]]) -> dict[str, list[list[float]]]:
    """Return card bbox-set-map annotations for rank-group witnesses."""

    return {
        str(key): card_bboxes_for_ids(render_map, card_ids)
        for key, card_ids in keyed_card_ids
    }


__all__ = ["card_bboxes_for_ids", "keyed_card_bbox_set_map"]
