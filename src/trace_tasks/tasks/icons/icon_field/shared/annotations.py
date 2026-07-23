"""Annotation helpers for icon-field scenes."""

from __future__ import annotations

from typing import List, Sequence

from ...shared.icon_scene import sort_bboxes_reading_order

from .state import IconFieldScenePayload


def bboxes_for_icon_ids(
    scene_payload: IconFieldScenePayload,
    icon_ids: Sequence[str],
) -> List[List[int]]:
    """Return reading-order bboxes for all instances of the given icon ids."""

    selected_ids = {str(icon_id) for icon_id in icon_ids}
    return sort_bboxes_reading_order(
        tuple(
            entity["bbox_xyxy"]
            for entity in scene_payload.scene_instances
            if str(entity.get("icon_id")) in selected_ids
        )
    )


def indices_for_icon_ids(
    scene_payload: IconFieldScenePayload,
    icon_ids: Sequence[str],
) -> List[int]:
    """Return scene instance indices for all instances of the given icon ids."""

    selected_ids = {str(icon_id) for icon_id in icon_ids}
    return [
        int(index)
        for index, icon_id in enumerate(scene_payload.scene_icon_ids)
        if str(icon_id) in selected_ids
    ]


__all__ = ["bboxes_for_icon_ids", "indices_for_icon_ids"]
