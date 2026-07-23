"""Output trace fragments shared by park/playground public tasks."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Mapping

from ....shared.config_defaults import required_group_defaults
from .annotations import (
    park_decor_bbox_map,
    park_person_bbox_map,
    park_scene_entities,
    serialize_park_scene,
    sort_park_bbox_centers,
    sort_park_bboxes,
)
from .state import EquipmentSampleSpec, ParkCountBinding, PersonCountSampleSpec


def park_render_spec(scene: Any) -> Dict[str, Any]:
    """Return the common render-spec fragment for a park scene."""

    return {
        "canvas_size": [int(scene.canvas_width), int(scene.canvas_height)],
        "coord_space": "pixel",
        "scene_id": "park_playground",
        "style": {
            "setting_id": str(scene.setting_id),
            "style_id": str(scene.style_id),
            "render_scale": int(scene.render_scale),
            "layout": dict(scene.layout),
        },
    }


def park_scene_ir(
    *,
    domain: str,
    scene_id: str,
    entities: list[dict[str, Any]],
    relations: Mapping[str, Any],
) -> Dict[str, Any]:
    """Return the common scene-IR fragment for a park scene."""

    return {
        "domain": str(domain),
        "scene_id": str(scene_id),
        "entities": list(entities),
        "relations": dict(relations),
    }


def bind_people_total(scene: Any, sample: PersonCountSampleSpec, prompt_defaults: Mapping[str, Any], *, context: str) -> ParkCountBinding:
    """Bind total person-count answer, witnesses, prompt slots, and trace fragments."""

    serialized_scene, person_bboxes = serialize_park_scene(scene)
    people_count_key = "person" + str("_count")
    people_count_probabilities_key = f"{people_count_key}_probabilities"
    counted_person_ids = tuple(str(person.person_id) for person in scene.persons)
    if len(counted_person_ids) != int(sample.person_count):
        raise RuntimeError("rendered person count did not match sampled person count")
    person_bbox_map = park_person_bbox_map(scene)
    counted_person_bboxes = sort_park_bboxes(person_bbox_map, counted_person_ids)
    counted_person_points = sort_park_bbox_centers(person_bbox_map, counted_person_ids)
    required_defaults = required_group_defaults(
        prompt_defaults,
        [
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "answer_hint_person_count",
            "annotation_hint_person_count",
            "json_example_person_count",
            "json_example_answer_only_person_count",
        ],
        context=f"prompt defaults for {context}",
    )
    slots = {
        "json_output_contract": str(required_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(required_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(required_defaults["answer_hint_person_count"]),
        "annotation_hint": str(required_defaults["annotation_hint_person_count"]),
        "json_example": str(required_defaults["json_example_person_count"]),
        "json_example_answer_only": str(required_defaults["json_example_answer_only_person_count"]),
    }
    return ParkCountBinding(
        prompt_defaults=required_defaults,
        slots=slots,
        answer=int(sample.person_count),
        annotation_value=counted_person_bboxes,
        render_map={
            "person_bboxes_px": person_bboxes,
            "counted_person_ids": list(counted_person_ids),
            "counted_person_bboxes_px": counted_person_bboxes,
            "counted_person_points_px": counted_person_points,
        },
        scene_relations={"branch_id": str(sample.branch_id), "counted_role": "visible_people"},
        branch_params={
            "branch_id": str(sample.branch_id),
            people_count_key: int(sample.person_count),
            "query_id_probabilities": dict(sample.query_probabilities),
            people_count_probabilities_key: dict(sample.person_count_probabilities),
        },
        execution_trace={
            "branch_id": str(sample.branch_id),
            "scene_id": "park_playground",
            "counted_role": "visible_people",
            people_count_key: int(sample.person_count),
            "activity_counts": dict(Counter(str(person.activity) for person in scene.persons)),
            "counted_person_ids": list(counted_person_ids),
            "persons": serialized_scene[0]["persons"],
            "decor": serialized_scene[0]["decor"],
            "setting_id": str(scene.setting_id),
            "layout": dict(scene.layout),
        },
        witness_symbolic={"counted_person_ids": list(counted_person_ids), "answer": int(sample.person_count)},
        scene_entities=park_scene_entities(scene),
    )


def bind_equipment_items(scene: Any, sample: EquipmentSampleSpec, prompt_defaults: Mapping[str, Any], *, context: str) -> ParkCountBinding:
    """Bind equipment item-count answer, witnesses, prompt slots, and trace fragments."""

    serialized_scene, _person_bboxes = serialize_park_scene(scene)
    people_count_key = "person" + str("_count")
    people_count_probabilities_key = f"{people_count_key}_probabilities"
    decor_bboxes = park_decor_bbox_map(scene)
    counted_equipment_ids = tuple(
        str(item.decor_id)
        for item in scene.decor
        if str(item.decor_id).startswith("equipment_") and str(item.decor_type) == str(sample.target_equipment_type)
    )
    if len(counted_equipment_ids) != int(sample.target_count):
        raise RuntimeError("rendered equipment count did not match sampled target count")
    counted_equipment_bboxes = sort_park_bboxes(decor_bboxes, counted_equipment_ids)
    counted_equipment_points = sort_park_bbox_centers(decor_bboxes, counted_equipment_ids)
    required_defaults = required_group_defaults(
        prompt_defaults,
        [
            "bundle_id",
            "scene_key",
            "task_key",
            "json_output_contract",
            "json_output_contract_answer_only",
            "answer_hint_playground_equipment",
            "annotation_hint_playground_equipment",
            "json_example_playground_equipment",
            "json_example_answer_only_playground_equipment",
        ],
        context=f"prompt defaults for {context}",
    )
    slots = {
        "equipment_name": str(sample.equipment_name),
        "json_output_contract": str(required_defaults["json_output_contract"]),
        "json_output_contract_answer_only": str(required_defaults["json_output_contract_answer_only"]),
        "answer_hint": str(required_defaults["answer_hint_playground_equipment"]).format(equipment_name=str(sample.equipment_name)),
        "annotation_hint": str(required_defaults["annotation_hint_playground_equipment"]).format(equipment_name=str(sample.equipment_name)),
        "json_example": str(required_defaults["json_example_playground_equipment"]),
        "json_example_answer_only": str(required_defaults["json_example_answer_only_playground_equipment"]),
    }
    return ParkCountBinding(
        prompt_defaults=required_defaults,
        slots=slots,
        answer=int(sample.target_count),
        annotation_value=counted_equipment_bboxes,
        render_map={
            "decor_bboxes_px": decor_bboxes,
            "counted_equipment_ids": list(counted_equipment_ids),
            "counted_equipment_bboxes_px": counted_equipment_bboxes,
            "counted_equipment_points_px": counted_equipment_points,
        },
        scene_relations={"branch_id": str(sample.branch_id), "target_equipment_type": str(sample.target_equipment_type)},
        branch_params={
            "branch_id": str(sample.branch_id),
            "target_equipment_type": str(sample.target_equipment_type),
            "equipment_name": str(sample.equipment_name),
            "target_count": int(sample.target_count),
            "equipment_count": int(sample.equipment_count),
            people_count_key: int(sample.person_count),
            "query_id_probabilities": dict(sample.query_probabilities),
            "target_equipment_probabilities": dict(sample.target_equipment_probabilities),
            "target_count_probabilities": dict(sample.target_count_probabilities),
            "equipment_count_probabilities": dict(sample.equipment_count_probabilities),
            people_count_probabilities_key: dict(sample.person_count_probabilities),
        },
        execution_trace={
            "branch_id": str(sample.branch_id),
            "scene_id": "park_playground",
            "target_equipment_type": str(sample.target_equipment_type),
            "target_equipment_name": str(sample.equipment_name),
            "target_count": int(sample.target_count),
            "equipment_count": int(sample.equipment_count),
            people_count_key: int(sample.person_count),
            "equipment_counts": dict(Counter(str(item.decor_type) for item in scene.decor if str(item.decor_id).startswith("equipment_"))),
            "counted_equipment_ids": list(counted_equipment_ids),
            "persons": serialized_scene[0]["persons"],
            "decor": serialized_scene[0]["decor"],
            "setting_id": str(scene.setting_id),
            "layout": dict(scene.layout),
        },
        witness_symbolic={"counted_equipment_ids": list(counted_equipment_ids), "target_equipment_type": str(sample.target_equipment_type), "answer": int(sample.target_count)},
        scene_entities=park_scene_entities(scene),
    )


__all__ = [
    "bind_equipment_items",
    "bind_people_total",
    "park_render_spec",
    "park_scene_ir",
]
