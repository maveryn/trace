"""Tests for the 3D street same-road-arm reference task."""

from __future__ import annotations

import pytest

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import TASK_REGISTRY, ensure_scene_tasks_registered
from trace_tasks.tasks.three_d.street.same_road_arm_reference_label import (
    ROAD_ARMS,
    SCENE_ID,
    STREET_OBJECT_TYPES,
    SUPPORTED_QUERY_IDS,
    TASK_ID,
)
from tests.three_d_option_panel_helpers import assert_option_panel_matches_candidates


@pytest.mark.parametrize(
    ("scene_variant", "intersection_layout"),
    [
        ("downtown_intersection", "four_way"),
        ("neighborhood_intersection", "t_missing_north"),
        ("transit_intersection", "t_missing_west"),
    ],
)
def test_street_same_road_arm_reference_answer_annotation_and_geometry(
    scene_variant: str,
    intersection_layout: str,
) -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20260522,
        params={
            "query_id": "single",
            "scene_variant": scene_variant,
            "intersection_layout": intersection_layout,
            "candidate_count": 4,
            "context_object_count": 10,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=260,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    entities = output.trace_payload["scene_ir"]["entities"]
    candidates = list(trace["candidate_object_specs"])
    reference = dict(trace["reference_object"])
    answer_label = str(trace["answer_label"])
    answer_spec = next(
        spec for spec in candidates if str(spec["point_label"]) == answer_label
    )
    expected_bbox = render_map["object_bboxes_px"][str(answer_spec["object_id"])]
    same_arm_labels = [
        str(label)
        for label, flag in trace["same_road_arm_as_reference_by_label"].items()
        if bool(flag)
    ]

    assert output.query_id == "single"
    assert output.scene_id == SCENE_ID
    assert output.query_id == "single"
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
    assert same_arm_labels == [answer_label]
    assert trace["same_road_arm_candidate_labels"] == [answer_label]
    assert str(answer_spec["road_arm"]) == str(reference["road_arm"])
    assert str(reference["road_arm"]) in set(trace["present_road_arms"])
    assert sorted(str(spec["point_label"]) for spec in candidates) == list("ABCD")
    assert len(candidates) == 4
    assert len(trace["context_object_specs"]) == 10
    assert len(trace["reference_object_specs"]) == 1
    assert trace["reference_object_specs"][0]["object_role"] == "street_reference"
    assert not reference["object_id"].startswith("street_object_")
    assert str(reference["object_id"]) in render_map["object_bboxes_px"]
    reference_entity = next(
        entity
        for entity in entities
        if str(entity["entity_id"]) == str(reference["object_id"])
    )
    assert reference_entity["attrs"]["reference_marker"] == "red_bbox"
    assert reference_entity["attrs"]["reference_marker_bbox_px"] is not None
    assert str(reference["object_type"]) not in {
        str(spec["object_type"]) for spec in candidates
    }
    assert str(reference["object_type"]) in set(STREET_OBJECT_TYPES)
    assert set(trace["candidate_road_arm_by_label"].values()).issubset(set(ROAD_ARMS))
    assert str(trace["missing_road_arm"]) not in set(
        trace["candidate_road_arm_by_label"].values()
    )
    assert "red-boxed object" in output.prompt
    assert "{reference_object_name}" not in output.prompt
    assert "{answer_hint}" not in output.prompt


def test_street_same_road_arm_reference_registered() -> None:
    taxonomy = resolve_task_taxonomy(TASK_ID)

    ensure_scene_tasks_registered("three_d", "street")
    assert TASK_ID in TASK_REGISTRY
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == SCENE_ID
    assert taxonomy.source_scene_id == ""
    assert SUPPORTED_QUERY_IDS == ("single",)
