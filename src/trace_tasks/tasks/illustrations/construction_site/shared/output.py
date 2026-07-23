"""Trace and review serialization helpers for construction-site illustrations."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

from ...shared.object_rendering import make_vector_scene_object_record

from .annotations import (
    construction_equipment_bbox_map,
    construction_material_bbox_map,
    construction_worker_bbox_map,
    construction_zone_bbox_map,
)
from .state import RenderedConstructionSiteScene, safe_json_value


def construction_scene_entities(scene: RenderedConstructionSiteScene) -> List[Dict[str, Any]]:
    """Return trace-ready construction entities with visible witness metadata."""

    entities: List[Dict[str, Any]] = []
    zone_label_font = scene.layout.get("zone_label_font", {}) if isinstance(scene.layout, Mapping) else {}
    for zone in scene.zones:
        zone_visual_attributes: Dict[str, Any] = {
            "fill_rgb": [int(v) for v in zone.fill_rgb],
            "outline_rgb": [int(v) for v in zone.outline_rgb],
        }
        if isinstance(zone_label_font, Mapping):
            zone_visual_attributes["label_font"] = safe_json_value(zone_label_font)
        object_record = make_vector_scene_object_record(
            object_id=str(zone.zone_id),
            object_type="zone",
            bbox_xyxy=zone.bbox_xyxy,
            semantic_attributes={"zone_id": str(zone.zone_id), "label": str(zone.label)},
            visual_attributes=zone_visual_attributes,
            source_entity_type="construction_zone",
            render_scale=int(scene.render_scale),
            style_id=str(scene.style_id),
        )
        entities.append(
            {
                "id": str(zone.zone_id),
                "type": "construction_zone",
                "label": str(zone.label),
                "bbox_xyxy": [round(float(v), 3) for v in zone.bbox_xyxy],
                "label_font": safe_json_value(zone_label_font) if isinstance(zone_label_font, Mapping) else {},
                "object_record": object_record,
            }
        )
    for worker in scene.workers:
        object_record = (
            dict(worker.object_record)
            if worker.object_record is not None
            else make_vector_scene_object_record(
                object_id=str(worker.worker_id),
                object_type="worker",
                bbox_xyxy=worker.bbox_xyxy,
                semantic_attributes={
                    "hard_hat_color": str(worker.hard_hat_color),
                    "vest_color": str(worker.vest_color),
                    "tool_type": str(worker.tool_type) if worker.tool_type else None,
                    **dict(worker.attributes),
                },
                visual_attributes={"style_id": str(worker.style_id), "gender_id": str(worker.gender_id)},
                role=str(worker.role),
                source_entity_type="construction_worker",
                render_scale=int(scene.render_scale),
                style_id=str(scene.style_id),
            )
        )
        entities.append(
            {
                "id": str(worker.worker_id),
                "type": "construction_worker",
                "hard_hat_color": str(worker.hard_hat_color),
                "vest_color": str(worker.vest_color),
                "tool_type": str(worker.tool_type) if worker.tool_type else None,
                "bbox_xyxy": [round(float(v), 3) for v in worker.bbox_xyxy],
                "role": str(worker.role),
                "gender_id": str(worker.gender_id),
                "attributes": safe_json_value(worker.attributes),
                "object_record": object_record,
            }
        )
    for material in scene.materials:
        object_record = (
            dict(material.object_record)
            if material.object_record is not None
            else make_vector_scene_object_record(
                object_id=str(material.material_id),
                object_type="construction_material",
                bbox_xyxy=material.bbox_xyxy,
                semantic_attributes={
                    "material_type": str(material.material_type),
                    "material_label": str(material.material_label),
                    **dict(material.attributes),
                },
                visual_attributes={"style_id": str(material.style_id)},
                role=str(material.role),
                source_entity_type="construction_material",
                render_scale=int(scene.render_scale),
                style_id=str(scene.style_id),
            )
        )
        entities.append(
            {
                "id": str(material.material_id),
                "type": "construction_material",
                "material_type": str(material.material_type),
                "label": str(material.material_label),
                "bbox_xyxy": [round(float(v), 3) for v in material.bbox_xyxy],
                "role": str(material.role),
                "attributes": safe_json_value(material.attributes),
                "object_record": object_record,
            }
        )
    for equipment in scene.equipment:
        object_record = (
            dict(equipment.object_record)
            if equipment.object_record is not None
            else make_vector_scene_object_record(
                object_id=str(equipment.equipment_id),
                object_type="construction_equipment",
                bbox_xyxy=equipment.bbox_xyxy,
                semantic_attributes={
                    "equipment_type": str(equipment.equipment_type),
                    "equipment_label": str(equipment.equipment_label),
                    "zone_id": str(equipment.zone_id),
                    **dict(equipment.attributes),
                },
                visual_attributes={"style_id": str(equipment.style_id)},
                role=str(equipment.role),
                source_entity_type="construction_equipment",
                render_scale=int(scene.render_scale),
                style_id=str(scene.style_id),
            )
        )
        entities.append(
            {
                "id": str(equipment.equipment_id),
                "type": "construction_equipment",
                "equipment_type": str(equipment.equipment_type),
                "label": str(equipment.equipment_label),
                "zone_id": str(equipment.zone_id),
                "bbox_xyxy": [round(float(v), 3) for v in equipment.bbox_xyxy],
                "role": str(equipment.role),
                "attributes": safe_json_value(equipment.attributes),
                "object_record": object_record,
            }
        )
    for item in scene.decor:
        object_record = make_vector_scene_object_record(
            object_id=str(item.decor_id),
            object_type="decor",
            bbox_xyxy=item.bbox_xyxy,
            semantic_attributes={"decor_type": str(item.decor_type), **dict(item.attributes)},
            source_entity_type="construction_decor",
            render_scale=int(scene.render_scale),
            style_id=str(scene.style_id),
        )
        entities.append(
            {
                "id": str(item.decor_id),
                "type": "construction_decor",
                "decor_type": str(item.decor_type),
                "bbox_xyxy": [round(float(v), 3) for v in item.bbox_xyxy],
                "attributes": safe_json_value(item.attributes),
                "object_record": object_record,
            }
        )
    return entities


def serialize_construction_scene(scene: RenderedConstructionSiteScene) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
    """Serialize construction scene for trace metadata."""

    payload = {
        "setting_id": str(scene.setting_id),
        "style_id": str(scene.style_id),
        "layout": safe_json_value(scene.layout),
        "zones": [
            {
                "zone_id": str(zone.zone_id),
                "label": str(zone.label),
                "bbox_xyxy": [round(float(v), 3) for v in zone.bbox_xyxy],
            }
            for zone in scene.zones
        ],
        "workers": [
            {
                "worker_id": str(worker.worker_id),
                "hard_hat_color": str(worker.hard_hat_color),
                "vest_color": str(worker.vest_color),
                "tool_type": str(worker.tool_type) if worker.tool_type else None,
                "bbox_xyxy": [round(float(v), 3) for v in worker.bbox_xyxy],
                "gender_id": str(worker.gender_id),
                "role": str(worker.role),
                "attributes": safe_json_value(worker.attributes),
            }
            for worker in scene.workers
        ],
        "materials": [
            {
                "material_id": str(material.material_id),
                "material_type": str(material.material_type),
                "material_label": str(material.material_label),
                "bbox_xyxy": [round(float(v), 3) for v in material.bbox_xyxy],
                "role": str(material.role),
                "attributes": safe_json_value(material.attributes),
            }
            for material in scene.materials
        ],
        "equipment": [
            {
                "equipment_id": str(equipment.equipment_id),
                "equipment_type": str(equipment.equipment_type),
                "equipment_label": str(equipment.equipment_label),
                "zone_id": str(equipment.zone_id),
                "bbox_xyxy": [round(float(v), 3) for v in equipment.bbox_xyxy],
                "role": str(equipment.role),
                "attributes": safe_json_value(equipment.attributes),
            }
            for equipment in scene.equipment
        ],
        "decor": [
            {
                "decor_id": str(item.decor_id),
                "decor_type": str(item.decor_type),
                "bbox_xyxy": [round(float(v), 3) for v in item.bbox_xyxy],
                "attributes": safe_json_value(item.attributes),
            }
            for item in scene.decor
        ],
    }
    bbox_map: Dict[str, List[float]] = {}
    bbox_map.update(construction_worker_bbox_map(scene))
    bbox_map.update(construction_material_bbox_map(scene))
    bbox_map.update(construction_equipment_bbox_map(scene))
    bbox_map.update(construction_zone_bbox_map(scene))
    return [payload], bbox_map


def construction_count_trace_sections(
    *,
    domain: str,
    scene_id: str,
    scene: RenderedConstructionSiteScene,
    relations: Mapping[str, Any],
    render_map: Mapping[str, Any],
    execution_trace: Mapping[str, Any],
    witness_symbolic: Mapping[str, Any],
    projected_annotation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Assemble non-identity trace sections for construction count tasks."""

    return {
        "scene_ir": {
            "domain": str(domain),
            "scene_id": str(scene_id),
            "entities": construction_scene_entities(scene),
            "relations": safe_json_value(dict(relations)),
        },
        "render_spec": {
            "canvas_size": [int(scene.canvas_width), int(scene.canvas_height)],
            "coord_space": "pixel",
            "scene_id": str(scene_id),
            "style": {
                "setting_id": str(scene.setting_id),
                "style_id": str(scene.style_id),
                "render_scale": int(scene.render_scale),
                "layout": safe_json_value(dict(scene.layout)),
            },
        },
        "render_map": safe_json_value(dict(render_map)),
        "execution_trace": safe_json_value(dict(execution_trace)),
        "witness_symbolic": safe_json_value(dict(witness_symbolic)),
        "projected_annotation": safe_json_value(dict(projected_annotation)),
    }


__all__ = [
    "construction_scene_entities",
    "serialize_construction_scene",
]
