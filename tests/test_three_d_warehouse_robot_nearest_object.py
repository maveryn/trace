"""Tests for the 3D warehouse nearest-reference task."""

from __future__ import annotations

import pytest

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import list_default_task_ids
from trace_tasks.tasks.three_d.shared.object_resources import WAREHOUSE_NEAREST_OBJECT_CANDIDATE_TYPES
from trace_tasks.tasks.three_d.warehouse.nearest_candidate_to_reference_label import (
    MIN_NEAREST_OBJECT_MARGIN,
    MIN_NEAREST_REFERENCE_OBJECT_MARGIN,
    SUPPORTED_AISLE_HEADINGS,
    SUPPORTED_QUERY_IDS,
    TASK_ID,
)
from trace_tasks.tasks.three_d.warehouse.shared.state import SCENE_ID
from tests.three_d_option_panel_helpers import assert_option_panel_matches_candidates


@pytest.mark.parametrize(
    ("scene_variant", "aisle_heading"),
    [
        ("storage_aisle", "east"),
        ("loading_zone", "north"),
        ("packing_floor", "west"),
    ],
)
def test_warehouse_object_nearest_reference_answer_annotation_and_geometry(
    scene_variant: str,
    aisle_heading: str,
) -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20260524,
        params={
            "query_id": "closest_object_to_reference",
            "scene_variant": scene_variant,
            "aisle_heading": aisle_heading,
            "candidate_count": 4,
            "context_object_count": 10,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=700,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    entities = output.trace_payload["scene_ir"]["entities"]
    candidates = list(trace["candidate_object_specs"])
    reference = dict(trace["reference_object"])
    answer_label = str(trace["answer_label"])
    answer_spec = next(spec for spec in candidates if str(spec["point_label"]) == answer_label)
    expected_bbox = render_map["object_bboxes_px"][str(answer_spec["object_id"])]
    nearest_labels = [str(label) for label, flag in trace["nearest_object_by_label"].items() if bool(flag)]

    assert output.scene_id == SCENE_ID
    assert output.query_id == "closest_object_to_reference"
    assert output.answer_gt.type == "option_letter"
    assert output.answer_gt.value == answer_label
    assert output.annotation_gt.type == "bbox"
    assert output.annotation_gt.value == expected_bbox
    assert_option_panel_matches_candidates(
        output,
        candidates,
        answer_label=answer_label,
        answer_object_id=str(answer_spec["object_id"]),
        expected_image_size=(1180, 1088),
    )
    assert trace["target_object_ids"] == [str(answer_spec["object_id"])]
    assert nearest_labels == [answer_label]
    assert trace["nearest_object_candidate_labels"] == [answer_label]
    assert trace["distance_order_near_to_far"][0] == answer_label
    assert float(trace["nearest_reference_object_margin"]) >= MIN_NEAREST_REFERENCE_OBJECT_MARGIN
    assert bool(answer_spec["is_nearest_object_to_reference_object"]) is True
    assert str(reference["object_type"]) == "red_sphere"
    assert str(reference["object_role"]) == "warehouse_reference_object"
    assert str(trace["reference_object_name"]) == "red sphere"
    assert str(trace["aisle_heading"]) == str(aisle_heading)
    assert sorted(str(spec["point_label"]) for spec in candidates) == list("ABCD")
    assert len(candidates) == 4
    assert len(trace["candidate_specs"]) == 4
    assert len(trace["candidate_robot_specs"]) == 0
    assert len(trace["context_object_specs"]) == 10
    assert len(trace["reference_object_specs"]) == 1
    assert all(str(spec["object_type"]) in set(WAREHOUSE_NEAREST_OBJECT_CANDIDATE_TYPES) for spec in candidates)
    assert all(str(spec["object_role"]) == "warehouse_object_candidate" for spec in candidates)
    candidate_entities = [
        entity
        for entity in entities
        if str(entity["entity_type"]) == "three_d_warehouse_candidate_object"
    ]
    reference_entities = [
        entity
        for entity in entities
        if str(entity["entity_type"]) == "three_d_warehouse_reference_object"
    ]
    assert len(candidate_entities) == 4
    assert len(reference_entities) == 1
    assert "red sphere" in output.prompt
    assert "candidate robots" not in output.prompt
    assert "gripper" not in output.prompt
    assert "{answer_hint}" not in output.prompt


@pytest.mark.parametrize(
    ("scene_variant", "aisle_heading"),
    [
        ("storage_aisle", "south"),
        ("loading_zone", "east"),
        ("packing_floor", "north"),
    ],
)
def test_warehouse_object_nearest_robot_answer_annotation_and_geometry(
    scene_variant: str,
    aisle_heading: str,
) -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20260525,
        params={
            "query_id": "closest_object_to_robot",
            "scene_variant": scene_variant,
            "aisle_heading": aisle_heading,
            "candidate_count": 4,
            "context_object_count": 10,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=700,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    entities = output.trace_payload["scene_ir"]["entities"]
    candidates = list(trace["candidate_object_specs"])
    references = list(trace["reference_robot_specs"])
    answer_label = str(trace["answer_label"])
    answer_spec = next(spec for spec in candidates if str(spec["point_label"]) == answer_label)
    expected_bbox = render_map["object_bboxes_px"][str(answer_spec["object_id"])]
    nearest_labels = [str(label) for label, flag in trace["nearest_object_by_label"].items() if bool(flag)]

    assert output.scene_id == SCENE_ID
    assert output.query_id == "closest_object_to_robot"
    assert output.answer_gt.type == "option_letter"
    assert output.answer_gt.value == answer_label
    assert output.annotation_gt.type == "bbox"
    assert output.annotation_gt.value == expected_bbox
    assert_option_panel_matches_candidates(
        output,
        candidates,
        answer_label=answer_label,
        answer_object_id=str(answer_spec["object_id"]),
        expected_image_size=(1180, 1088),
    )
    assert trace["target_object_ids"] == [str(answer_spec["object_id"])]
    assert nearest_labels == [answer_label]
    assert trace["nearest_object_candidate_labels"] == [answer_label]
    assert trace["distance_order_near_to_far"][0] == answer_label
    assert float(trace["nearest_object_margin"]) >= MIN_NEAREST_OBJECT_MARGIN
    assert bool(answer_spec["is_nearest_object_to_reference_robot"]) is True
    assert len(references) == 1
    assert str(references[0]["object_type"]) == "warehouse_robot"
    assert str(references[0]["object_role"]) == "warehouse_reference_robot"
    assert len(candidates) == 4
    assert sorted(str(spec["point_label"]) for spec in candidates) == list("ABCD")
    assert all(str(spec["object_type"]) in set(WAREHOUSE_NEAREST_OBJECT_CANDIDATE_TYPES) for spec in candidates)
    assert all(str(spec["object_role"]) == "warehouse_object_candidate" for spec in candidates)
    assert len(trace["candidate_robot_specs"]) == 0
    assert len(trace["context_object_specs"]) == 10
    assert len([spec for spec in trace["context_object_specs"] if str(spec["object_type"]) == "shelf_rack"]) == 4
    candidate_entities = [
        entity
        for entity in entities
        if str(entity["entity_type"]) == "three_d_warehouse_candidate_object"
    ]
    reference_entities = [
        entity
        for entity in entities
        if str(entity["entity_type"]) == "three_d_warehouse_reference_robot"
    ]
    assert len(candidate_entities) == 4
    answer_entity = next(entity for entity in candidate_entities if str(entity["entity_id"]) == str(answer_spec["object_id"]))
    answer_record = answer_entity["attrs"]["object_record"]
    assert answer_record["object_id"] == str(answer_spec["object_id"])
    assert answer_record["object_type"] == str(answer_spec["object_type"])
    assert answer_record["visual_attributes"]["renderer_id"] == "warehouse_object"
    assert answer_record["visual_attributes"]["renderer_style"] == "projected_3d"
    assert len(reference_entities) == 1
    assert "robot" in output.prompt
    assert "candidate warehouse objects" in output.prompt or "option" in output.prompt
    assert "gripper" not in output.prompt
    assert "red sphere" not in output.prompt
    assert "{answer_hint}" not in output.prompt


def test_warehouse_robot_nearest_object_registered() -> None:
    taxonomy = resolve_task_taxonomy(TASK_ID)

    assert TASK_ID in list_default_task_ids()
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == SCENE_ID
    assert taxonomy.source_scene_id == ""
    assert SUPPORTED_QUERY_IDS == ("closest_object_to_reference", "closest_object_to_robot")
    assert SUPPORTED_AISLE_HEADINGS == ("east", "north", "west", "south")
