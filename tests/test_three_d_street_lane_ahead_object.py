"""Tests for the 3D street lane-ahead object task."""

from __future__ import annotations

import pytest

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import TASK_REGISTRY, ensure_scene_tasks_registered
from trace_tasks.tasks.three_d.street.lane_ahead_object_label import (
    MIN_FORWARD_DISTANCE,
    REFERENCE_OBJECT_TYPE,
    SCENE_ID,
    SUPPORTED_QUERY_IDS,
    SUPPORTED_TRAVEL_MODES,
    TASK_ID,
)
from tests.three_d_option_panel_helpers import assert_option_panel_matches_candidates


@pytest.mark.parametrize(
    ("scene_variant", "intersection_layout", "travel_mode"),
    [
        ("downtown_intersection", "four_way", "toward_intersection"),
        ("neighborhood_intersection", "t_missing_north", "away_from_intersection"),
        ("transit_intersection", "t_missing_west", "toward_intersection"),
    ],
)
def test_street_lane_ahead_answer_annotation_and_geometry(
    scene_variant: str,
    intersection_layout: str,
    travel_mode: str,
) -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20260522,
        params={
            "query_id": "single",
            "scene_variant": scene_variant,
            "intersection_layout": intersection_layout,
            "travel_mode": travel_mode,
            "candidate_count": 4,
            "context_object_count": 8,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=320,
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
    ahead_labels = [
        str(label)
        for label, flag in trace["ahead_along_lane_by_label"].items()
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
    assert ahead_labels == [answer_label]
    assert trace["ahead_along_lane_candidate_labels"] == [answer_label]
    assert str(answer_spec["road_arm"]) == str(reference["road_arm"])
    assert str(answer_spec["lane_id"]) == str(reference["lane_id"])
    assert float(answer_spec["forward_distance_from_reference"]) >= MIN_FORWARD_DISTANCE
    assert str(reference["object_type"]) == REFERENCE_OBJECT_TYPE
    assert str(reference["road_arm"]) in set(trace["present_road_arms"])
    assert str(trace["travel_mode"]) == str(travel_mode)
    assert len(reference["travel_direction_vector_xy"]) == 2
    assert sorted(str(spec["point_label"]) for spec in candidates) == list("ABCD")
    assert len(candidates) == 4
    assert len(trace["context_object_specs"]) == 8
    assert len(trace["reference_object_specs"]) == 1
    assert trace["reference_object_specs"][0]["object_role"] == "street_reference"
    assert str(reference["object_id"]) in render_map["object_bboxes_px"]
    reference_entity = next(
        entity
        for entity in entities
        if str(entity["entity_id"]) == str(reference["object_id"])
    )
    assert reference_entity["attrs"]["reference_marker"] == "red_bbox"
    assert reference_entity["attrs"]["reference_marker_bbox_px"] is not None
    assert reference_entity["attrs"]["reference_direction_marker_bbox_px"] is not None
    assert str(REFERENCE_OBJECT_TYPE) not in {
        str(spec["object_type"]) for spec in candidates
    }
    assert "red-boxed object" in output.prompt
    assert "arrow" in output.prompt
    assert "{answer_hint}" not in output.prompt


def test_street_lane_ahead_registered() -> None:
    taxonomy = resolve_task_taxonomy(TASK_ID)

    ensure_scene_tasks_registered("three_d", "street")
    assert TASK_ID in TASK_REGISTRY
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == SCENE_ID
    assert taxonomy.source_scene_id == ""
    assert SUPPORTED_QUERY_IDS == ("single",)
    assert SUPPORTED_TRAVEL_MODES == ("toward_intersection", "away_from_intersection")
