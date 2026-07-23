"""Tests for the 3D room wall-object camera-distance task."""

from __future__ import annotations

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import is_default_dataset_task
from trace_tasks.tasks.three_d.room.wall_object_camera_distance_label import (
    CAMERA_DISTANCE_MIN_MARGIN,
    CAMERA_DISTANCE_MIN_ROOM_DEPTH_MARGIN,
    LETTERED_WALL_OBJECT_MIN_VISIBLE_PX,
    SCENE_ID,
    SUPPORTED_QUERY_IDS,
    TASK_ID,
)
from trace_tasks.tasks.three_d.room.shared.state import SIDE_WALL_OBJECT_HPOS_MAX
from tests.three_d_option_panel_helpers import assert_option_panel_matches_candidates


def _assert_side_wall_objects_are_not_deep(trace) -> None:
    for spec in trace["wall_object_specs"]:
        if str(spec["wall"]) in {"left", "right"}:
            assert float(spec["world_xyz"][1]) <= float(SIDE_WALL_OBJECT_HPOS_MAX)


def test_room_wall_object_camera_distance_answer_and_annotation() -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20261003,
        params={
            "query_id": "single",
            "scene_variant": "studio_room",
            "candidate_count": 6,
            "context_wall_count": 0,
            "floor_context_count": 6,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=160,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    entities = output.trace_payload["scene_ir"]["entities"]
    candidates = list(trace["candidate_object_specs"])
    nearest = min(candidates, key=lambda spec: (float(spec["camera_distance"]), str(spec["point_label"])))
    expected_bbox = render_map["object_bboxes_px"][str(nearest["object_id"])]
    assert output.scene_id == SCENE_ID
    assert output.query_id == "single"
    assert output.answer_gt.type == "option_letter"
    assert output.answer_gt.value == str(nearest["point_label"])
    assert output.annotation_gt.type == "bbox"
    assert output.annotation_gt.value == expected_bbox
    assert_option_panel_matches_candidates(
        output,
        candidates,
        answer_label=str(nearest["point_label"]),
        answer_object_id=str(nearest["object_id"]),
        expected_image_size=(1180, 1068),
    )
    assert trace["answer_label"] == str(nearest["point_label"])
    assert trace["answer_object_id"] == str(nearest["object_id"])
    assert trace["target_object_ids"] == [str(nearest["object_id"])]
    assert trace["camera_distance_order_near_to_far"][0] == str(nearest["point_label"])
    assert trace["room_depth_order_front_to_back"][0] == str(nearest["point_label"])
    nearest_entity = next(entity for entity in entities if str(entity["entity_id"]) == str(nearest["object_id"]))
    nearest_record = nearest_entity["attrs"]["object_record"]
    assert nearest_record["object_id"] == str(nearest["object_id"])
    assert nearest_record["object_type"] == str(nearest["object_type"])
    assert nearest_record["visual_attributes"]["renderer_id"] == "room_wall_object"
    assert nearest_record["visual_attributes"]["renderer_style"] == "projected_3d"
    assert len(candidates) == 6
    assert int(trace["context_wall_count"]) == 0
    assert int(trace["wall_object_count"]) == 6
    assert int(trace["floor_object_count"]) == 6
    assert sorted(str(spec["point_label"]) for spec in candidates) == list("ABCDEF")
    assert {str(spec["wall"]) for spec in candidates} == {"back", "left", "right"}
    assert all(bool(spec["is_wall_mounted"]) for spec in candidates)
    assert all(bool(spec["is_answer_candidate"]) for spec in candidates)
    _assert_side_wall_objects_are_not_deep(trace)
    assert abs(float(trace["camera"]["yaw_degrees"])) <= 8.0
    for bbox in trace["candidate_visible_bboxes_by_label"].values():
        width = float(bbox[2]) - float(bbox[0])
        height = float(bbox[3]) - float(bbox[1])
        assert width >= LETTERED_WALL_OBJECT_MIN_VISIBLE_PX
        assert height >= LETTERED_WALL_OBJECT_MIN_VISIBLE_PX
    assert float(trace["camera_distance_margin"]) >= CAMERA_DISTANCE_MIN_MARGIN
    assert float(trace["room_depth_margin"]) >= CAMERA_DISTANCE_MIN_ROOM_DEPTH_MARGIN
    assert any(entity["entity_id"] == "room_shell" for entity in entities)


def test_room_wall_object_camera_distance_registered() -> None:
    taxonomy = resolve_task_taxonomy(TASK_ID)

    assert is_default_dataset_task(TASK_ID)
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == SCENE_ID
    assert taxonomy.source_scene_id == ""
    assert SUPPORTED_QUERY_IDS == ("single",)
