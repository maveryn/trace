"""Annotation projection helpers for the 2048 games scene."""

from __future__ import annotations

from trace_tasks.tasks.shared.bbox_projection import bbox_center

from .state import Move2048Result, coord_to_cell_id


def source_merge_cell_id_pairs(result: Move2048Result) -> list[list[str]]:
    """Return source cell-id pairs for visible merge events."""

    return [
        [coord_to_cell_id(tuple(left)), coord_to_cell_id(tuple(right))]
        for left, right in result.merge_pairs
    ]


def source_merge_point_pairs(
    result: Move2048Result,
    rendered_scene,
) -> list[list[list[float]]]:
    """Project visible merge source-cell pairs to segments."""

    entity_bboxes = rendered_scene.render_map["entity_bboxes_px"]
    pairs: list[list[list[float]]] = []
    for left_id, right_id in source_merge_cell_id_pairs(result):
        pairs.append(
            [
                list(bbox_center(entity_bboxes[str(left_id)])),
                list(bbox_center(entity_bboxes[str(right_id)])),
            ]
        )
    return pairs


def entity_bboxes_for_ids(rendered_scene, entity_ids: tuple[str, ...]) -> list[list[float]]:
    """Return rendered entity bboxes for one ordered id tuple."""

    entity_bboxes = rendered_scene.render_map["entity_bboxes_px"]
    return [list(entity_bboxes[str(entity_id)]) for entity_id in entity_ids]


__all__ = [
    "entity_bboxes_for_ids",
    "source_merge_cell_id_pairs",
    "source_merge_point_pairs",
]
