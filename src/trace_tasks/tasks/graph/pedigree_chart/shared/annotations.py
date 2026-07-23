"""Annotation projection helpers for pedigree-chart scenes."""

from __future__ import annotations

from typing import Any, Dict, List, Sequence, Tuple

from ....shared.bbox_projection import round_bbox
from .state import RenderedPedigreeScene

def projected_pedigree_person_point_annotation(
    rendered_scene: RenderedPedigreeScene,
    person_ids: Sequence[str],
) -> Dict[str, Any]:
    """Project selected person ids to point-set annotation."""

    rendered_by_id = {str(person.person_id): person for person in rendered_scene.people}
    points: List[List[int]] = []
    bbox_map: Dict[str, List[int]] = {}
    for person_id in [str(item) for item in person_ids]:
        person = rendered_by_id[str(person_id)]
        point = [int(person.center_xy[0]), int(person.center_xy[1])]
        points.append(point)
        bbox_map[str(person_id)] = [int(value) for value in round_bbox(person.symbol_bbox_xyxy)]
    return {
        "point_set": [list(point) for point in points],
        "pixel_point_set": [list(point) for point in points],
        "person_symbol_bbox_map": dict(bbox_map),
    }


def projected_keyed_pedigree_person_annotation(
    rendered_scene: RenderedPedigreeScene,
    role_person_ids: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    """Project role-bound person ids to keyed bbox annotation."""

    rendered_by_id = {str(person.person_id): person for person in rendered_scene.people}
    bbox_map: Dict[str, List[int]] = {}
    point_map: Dict[str, List[int]] = {}
    role_person_map: Dict[str, str] = {}
    for role, person_id in role_person_ids:
        rendered = rendered_by_id[str(person_id)]
        role_key = str(role)
        bbox_map[role_key] = [int(value) for value in round_bbox(rendered.symbol_bbox_xyxy)]
        point_map[role_key] = [int(rendered.center_xy[0]), int(rendered.center_xy[1])]
        role_person_map[role_key] = str(person_id)
    return {
        "bbox_map": dict(bbox_map),
        "pixel_bbox_map": dict(bbox_map),
        "point_map": dict(point_map),
        "pixel_point_map": dict(point_map),
        "role_person_id_map": dict(role_person_map),
    }


def projected_pedigree_person_bbox_set_annotation(
    rendered_scene: RenderedPedigreeScene,
    role_person_ids: Sequence[Tuple[str, str]],
) -> Dict[str, Any]:
    """Project role-bound person ids to an unordered de-duplicated bbox-set annotation."""

    keyed_projection = projected_keyed_pedigree_person_annotation(rendered_scene, role_person_ids)
    bbox_by_person_id: Dict[str, List[int]] = {}
    person_ids: List[str] = []
    for role, person_id in role_person_ids:
        person_key = str(person_id)
        if person_key in bbox_by_person_id:
            continue
        role_key = str(role)
        bbox_by_person_id[person_key] = list(keyed_projection["bbox_map"][role_key])
        person_ids.append(person_key)
    bbox_set = [list(bbox_by_person_id[person_id]) for person_id in person_ids]
    return {
        "type": "bbox_set",
        "bbox_set": [list(bbox) for bbox in bbox_set],
        "pixel_bbox_set": [list(bbox) for bbox in bbox_set],
        "person_symbol_bbox_map": {person_id: list(bbox_by_person_id[person_id]) for person_id in person_ids},
        "role_person_id_map": dict(keyed_projection["role_person_id_map"]),
    }


def projected_pedigree_generation_annotation(
    rendered_scene: RenderedPedigreeScene,
    *,
    generation_label: str,
    person_ids: Sequence[str],
) -> Dict[str, Any]:
    """Project an answer generation row and its affected witnesses to bbox annotation."""

    rendered_by_id = {str(person.person_id): person for person in rendered_scene.people}
    bbox_set: List[List[int]] = []
    label_bbox = rendered_scene.generation_label_bboxes.get(str(generation_label))
    generation_label_bbox = [int(value) for value in round_bbox(label_bbox)] if label_bbox is not None else []
    if generation_label_bbox:
        bbox_set.append(list(generation_label_bbox))
    person_symbol_bbox_map: Dict[str, List[int]] = {}
    for person_id in [str(item) for item in person_ids]:
        rendered = rendered_by_id[str(person_id)]
        bbox = [int(value) for value in round_bbox(rendered.symbol_bbox_xyxy)]
        bbox_set.append(list(bbox))
        person_symbol_bbox_map[str(person_id)] = list(bbox)
    return {
        "bbox_set": [list(bbox) for bbox in bbox_set],
        "pixel_bbox_set": [list(bbox) for bbox in bbox_set],
        "generation_label_bbox": list(generation_label_bbox),
        "person_symbol_bbox_map": dict(person_symbol_bbox_map),
    }

__all__ = [
    "projected_keyed_pedigree_person_annotation",
    "projected_pedigree_person_bbox_set_annotation",
    "projected_pedigree_generation_annotation",
    "projected_pedigree_person_point_annotation",
]
