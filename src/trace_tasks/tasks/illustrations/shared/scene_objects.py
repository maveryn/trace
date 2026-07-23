"""Helpers for consuming normalized illustration objects from any scene."""

from __future__ import annotations

from typing import Any, Iterable, Mapping

from .object_registry import make_object_record
from .object_schema import json_safe


_ENTITY_TYPE_OBJECT_MAP: dict[str, str] = {
    "construction_equipment": "construction_equipment",
    "construction_material": "construction_material",
    "construction_worker": "worker",
    "construction_zone": "zone",
    "environment_building": "building",
    "environment_feature": "environment_feature",
    "illustration_object": "",
    "indoor_container": "container",
    "indoor_furniture": "furniture",
    "indoor_surface": "surface",
    "library_book": "book",
    "library_decor": "decor",
    "library_section": "library_section",
    "park_decor": "decor",
    "park_person": "person",
}


def _entity_id(entity: Mapping[str, Any]) -> str:
    return str(entity.get("entity_id", entity.get("id", "")))


def _entity_type(entity: Mapping[str, Any]) -> str:
    return str(entity.get("entity_type", entity.get("type", "")))


def _entity_bbox(entity: Mapping[str, Any]) -> list[float] | None:
    bbox = entity.get("bbox", entity.get("bbox_xyxy"))
    if bbox is None:
        return None
    values = [float(value) for value in bbox]
    if len(values) != 4:
        return None
    return values


def _semantic_attributes(entity: Mapping[str, Any]) -> dict[str, Any]:
    skip = {
        "bbox",
        "bbox_xyxy",
        "entity_id",
        "entity_type",
        "id",
        "object_record",
        "parts",
        "role",
        "type",
        "visual_attributes",
    }
    result = {str(key): value for key, value in entity.items() if str(key) not in skip}
    return json_safe(result)


def normalized_object_record(entity: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return a normalized object record for one entity when possible."""

    existing = entity.get("object_record")
    if isinstance(existing, Mapping):
        return json_safe(existing)

    entity_type = _entity_type(entity)
    object_type = str(entity.get("object_type") or _ENTITY_TYPE_OBJECT_MAP.get(entity_type, ""))
    if not object_type:
        return None
    entity_id = _entity_id(entity)
    if not entity_id:
        return None

    record = make_object_record(
        object_id=entity_id,
        object_type=object_type,
        bbox_xyxy=_entity_bbox(entity),
        semantic_attributes=_semantic_attributes(entity),
        visual_attributes=dict(entity.get("visual_attributes", {})) if isinstance(entity.get("visual_attributes"), Mapping) else {},
        role=str(entity.get("role", "distractor")),
        source_entity_type=entity_type,
        parts=tuple(entity.get("parts", ())) if isinstance(entity.get("parts", ()), (list, tuple)) else (),
    )
    return record.as_dict()


def extract_scene_object_records(entities: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Return normalized object records from mixed scene entity payloads."""

    records: list[dict[str, Any]] = []
    for entity in entities:
        record = normalized_object_record(entity)
        if record is not None:
            records.append(record)
    return records


__all__ = ["extract_scene_object_records", "normalized_object_record"]
