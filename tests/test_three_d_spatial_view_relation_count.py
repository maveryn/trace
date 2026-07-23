"""Tests for the synthetic 3D view-relation count task."""

from __future__ import annotations

import pytest

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import list_default_task_ids
from trace_tasks.tasks.three_d.shared.object_scene_view_relation_count import (
    CAMERA_DEPTH_RELATION_COUNT_TASK_ID,
    IMAGE_PLANE_LATERAL_RELATION_COUNT_TASK_ID,
    MIN_REFERENCE_DEPTH_MARGIN,
    MIN_REFERENCE_X_BBOX_GAP_PX,
)
from tests.three_d_canvas_helpers import assert_three_d_canvas_contract

TASK_ID_BY_QUERY_ID = {
    "left_of_reference_in_view_count": IMAGE_PLANE_LATERAL_RELATION_COUNT_TASK_ID,
    "right_of_reference_in_view_count": IMAGE_PLANE_LATERAL_RELATION_COUNT_TASK_ID,
    "closer_to_camera_than_reference_count": CAMERA_DEPTH_RELATION_COUNT_TASK_ID,
    "farther_from_camera_than_reference_count": CAMERA_DEPTH_RELATION_COUNT_TASK_ID,
}


def _bbox_min_side(bbox) -> float:
    return min(float(bbox[2]) - float(bbox[0]), float(bbox[3]) - float(bbox[1]))


def test_view_relation_count_answer_and_annotation() -> None:
    task = create_task(IMAGE_PLANE_LATERAL_RELATION_COUNT_TASK_ID)
    output = task.generate(
        20260529,
        params={
            "query_id": "left_of_reference_in_view_count",
            "scene_variant": "floor_grid_room",
            "object_count": 9,
            "target_count": 6,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=220,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    target_object_ids = [str(object_id) for object_id in trace["target_object_ids"]]
    reference_object_id = str(trace["reference_object_id"])
    reference_bbox = render_map["object_bboxes_px"][reference_object_id]
    target_set = set(target_object_ids)
    reference_highlight_entity_id = f"red_reference_box_{reference_object_id}"

    assert output.scene_id == "object_scene"
    assert output.query_id == "left_of_reference_in_view_count"
    assert output.answer_gt.type == "integer"
    assert output.answer_gt.value == 6
    assert output.annotation_gt.type == "bbox_set"
    assert reference_object_id not in target_set
    assert int(trace["reference_prompt_name_count"]) == 1
    assert "red-boxed reference object" in output.prompt
    assert "red-boxed" in output.prompt
    assert output.trace_payload["render_map"]["reference_highlight_entity_id"] == reference_highlight_entity_id
    assert any(entity["entity_id"] == reference_highlight_entity_id for entity in output.trace_payload["scene_ir"]["entities"])
    assert len(output.annotation_gt.value) == int(output.answer_gt.value)
    expected_annotation_bboxes = [render_map["target_object_bboxes_px"][object_id] for object_id in target_object_ids]
    expected_raw_bboxes = [render_map["object_bboxes_px"][object_id] for object_id in target_object_ids]
    assert output.annotation_gt.value == expected_annotation_bboxes
    assert [render_map["target_object_raw_bboxes_px"][object_id] for object_id in target_object_ids] == expected_raw_bboxes
    assert all(_bbox_min_side(bbox) >= 24.0 for bbox in output.annotation_gt.value)
    assert output.trace_payload["projected_annotation"]["bbox_set"] == output.annotation_gt.value

    for spec in trace["object_specs"]:
        object_id = str(spec["object_id"])
        if object_id == reference_object_id:
            continue
        bbox = render_map["object_bboxes_px"][object_id]
        left_gap = float(reference_bbox[0]) - float(bbox[2])
        right_gap = float(bbox[0]) - float(reference_bbox[2])
        assert left_gap >= MIN_REFERENCE_X_BBOX_GAP_PX or right_gap >= MIN_REFERENCE_X_BBOX_GAP_PX
        assert (object_id in target_set) == (left_gap >= MIN_REFERENCE_X_BBOX_GAP_PX)
        assert bool(trace["view_relation_status_by_object_id"][object_id]) == (object_id in target_set)
    assert_three_d_canvas_contract(output)


def test_view_relation_count_query_variants_generate() -> None:
    query_ids = (
        "left_of_reference_in_view_count",
        "right_of_reference_in_view_count",
        "closer_to_camera_than_reference_count",
        "farther_from_camera_than_reference_count",
    )
    for offset, query_id in enumerate(query_ids):
        task = create_task(TASK_ID_BY_QUERY_ID[query_id])
        output = task.generate(
            20260530 + int(offset),
            params={
                "query_id": query_id,
                "scene_variant": "studio_platform",
                "object_count": 9 if query_id in {"left_of_reference_in_view_count", "right_of_reference_in_view_count"} else 10,
                "target_count": 3,
                "post_image_noise_apply_prob": 0.0,
            },
            max_attempts=220,
        )
        trace = output.trace_payload["execution_trace"]
        render_map = output.trace_payload["render_map"]
        reference_id = str(trace["reference_object_id"])
        reference_spec = next(spec for spec in trace["object_specs"] if str(spec["object_id"]) == reference_id)
        reference_bbox = render_map["object_bboxes_px"][reference_id]
        reference_distance = float(reference_spec["camera_distance"])
        target_ids = {str(object_id) for object_id in trace["target_object_ids"]}

        assert output.query_id == query_id
        assert output.answer_gt.type == "integer"
        assert "red-boxed" in output.prompt
        assert render_map["reference_highlight_entity_id"] == f"red_reference_box_{reference_id}"
        assert any(entity["entity_id"] == f"red_reference_box_{reference_id}" for entity in output.trace_payload["scene_ir"]["entities"])
        assert len(output.annotation_gt.value) == int(output.answer_gt.value)
        for spec in trace["object_specs"]:
            object_id = str(spec["object_id"])
            if object_id == reference_id:
                continue
            if query_id in {"left_of_reference_in_view_count", "right_of_reference_in_view_count"}:
                bbox = render_map["object_bboxes_px"][object_id]
                left_gap = float(reference_bbox[0]) - float(bbox[2])
                right_gap = float(bbox[0]) - float(reference_bbox[2])
                assert left_gap >= MIN_REFERENCE_X_BBOX_GAP_PX or right_gap >= MIN_REFERENCE_X_BBOX_GAP_PX
                expected = (
                    left_gap >= MIN_REFERENCE_X_BBOX_GAP_PX
                    if query_id == "left_of_reference_in_view_count"
                    else right_gap >= MIN_REFERENCE_X_BBOX_GAP_PX
                )
            else:
                distance_delta = float(spec["camera_distance"]) - reference_distance
                assert abs(distance_delta) >= MIN_REFERENCE_DEPTH_MARGIN
                expected = (
                    distance_delta < 0.0
                    if query_id == "closer_to_camera_than_reference_count"
                    else distance_delta > 0.0
                )
            assert (object_id in target_ids) == expected
            assert bool(trace["view_relation_status_by_object_id"][object_id]) == expected


@pytest.mark.parametrize(
    "task_id",
    [IMAGE_PLANE_LATERAL_RELATION_COUNT_TASK_ID, CAMERA_DEPTH_RELATION_COUNT_TASK_ID],
)
def test_view_relation_count_task_registered_in_three_d_taxonomy(task_id: str) -> None:
    taxonomy = resolve_task_taxonomy(task_id)

    assert task_id in list_default_task_ids()
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == "object_scene"
    assert not taxonomy.source_scene_id
