"""Annotation and trace projection helpers for park/playground scenes."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from .rendering import (
    park_decor_bbox_map,
    park_person_bbox_map,
    park_scene_entities,
    serialize_park_scene,
    sort_park_bboxes,
)


def sort_park_bbox_centers(bbox_map: Mapping[str, Sequence[float]], ids: Iterable[str]) -> list[list[float]]:
    """Return bbox center points sorted with the same order as park bbox witnesses."""

    boxes = [(str(item_id), [float(v) for v in bbox_map[str(item_id)]]) for item_id in ids]
    boxes.sort(key=lambda item: (float(item[1][1]), float(item[1][0]), str(item[0])))
    return [
        [
            round((float(box[0]) + float(box[2])) / 2.0, 3),
            round((float(box[1]) + float(box[3])) / 2.0, 3),
        ]
        for _item_id, box in boxes
    ]


__all__ = [
    "park_decor_bbox_map",
    "park_person_bbox_map",
    "park_scene_entities",
    "serialize_park_scene",
    "sort_park_bbox_centers",
    "sort_park_bboxes",
]
