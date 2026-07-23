"""Tests for the 3D room wall-object wall-side relation task."""

from __future__ import annotations

import pytest

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import is_default_dataset_task
from trace_tasks.tasks.three_d.room.wall_object_camera_distance_label import (
    LETTERED_WALL_OBJECT_MIN_VISIBLE_PX,
)
from trace_tasks.tasks.three_d.room.wall_object_side_relation_label import (
    REFERENCE_OBJECT_TYPE,
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


@pytest.mark.parametrize(
    ("reference_wall", "query_id", "relation_key"),
    [
        ("back", "left_of_reference_on_wall", "left_of_reference_on_wall_by_label"),
        ("back", "right_of_reference_on_wall", "right_of_reference_on_wall_by_label"),
        ("left", "left_of_reference_on_wall", "left_of_reference_on_wall_by_label"),
        ("right", "left_of_reference_on_wall", "left_of_reference_on_wall_by_label"),
        ("right", "right_of_reference_on_wall", "right_of_reference_on_wall_by_label"),
    ],
)
def test_room_wall_object_side_relation_answer_annotation_and_unique_reference(
    reference_wall: str,
    query_id: str,
    relation_key: str,
) -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20260522,
        params={
            "query_id": query_id,
            "scene_variant": "studio_room",
            "candidate_count": 6,
            "context_wall_count": 0,
            "floor_context_count": 6,
            "reference_wall": reference_wall,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=260,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    candidates = list(trace["candidate_object_specs"])
    reference = dict(trace["reference_object"])
    answer_label = str(trace["answer_label"])
    answer_spec = next(
        spec for spec in candidates if str(spec["point_label"]) == answer_label
    )
    expected_bbox = render_map["object_bboxes_px"][str(answer_spec["object_id"])]
    selected_relation_labels = [
        label
        for label, is_selected in trace[relation_key].items()
        if bool(is_selected)
    ]
    generic_relation_labels = [
        label
        for label, is_selected in trace["selected_side_relation_by_label"].items()
        if bool(is_selected)
    ]
    reference_left_coord = float(reference["wall_left_coordinate"])
    candidate_left_coords = {
        str(label): float(value)
        for label, value in trace[
            "candidate_wall_left_coordinates_by_label"
        ].items()
    }
    assert output.scene_id == SCENE_ID
    assert output.query_id == query_id
    assert output.answer_gt.type == "option_letter"
    assert output.answer_gt.value == answer_label
    assert output.annotation_gt.type == "bbox"
    assert output.annotation_gt.value == expected_bbox
    assert_option_panel_matches_candidates(
        output,
        candidates,
        answer_label=answer_label,
        answer_object_id=str(answer_spec["object_id"]),
        expected_image_size=(1180, 1068),
    )
    assert trace["target_object_ids"] == [str(answer_spec["object_id"])]
    assert selected_relation_labels == [answer_label]
    assert generic_relation_labels == [answer_label]
    assert str(reference["wall"]) == reference_wall
    assert str(reference["object_type"]) == REFERENCE_OBJECT_TYPE
    assert str(reference["prompt_name"]) == "TV"
    assert int(reference["prompt_name_count"]) == 1
    assert str(reference["object_id"]) not in trace["target_object_ids"]
    assert str(reference["object_type"]) not in {
        str(spec["object_type"]) for spec in candidates
    }
    assert str(answer_spec["wall"]) == reference_wall
    assert all(bool(spec["is_wall_mounted"]) for spec in candidates)
    assert all(bool(spec["is_answer_candidate"]) for spec in candidates)
    assert int(trace["context_wall_count"]) == 0
    assert int(trace["wall_object_count"]) == 7
    assert int(trace["floor_object_count"]) == 6
    assert sorted(str(spec["point_label"]) for spec in candidates) == list("ABCDEF")
    assert len(candidates) == 6
    _assert_side_wall_objects_are_not_deep(trace)
    assert trace["candidate_walls_by_label"][answer_label] == reference_wall
    same_wall_candidate_labels = [
        str(label)
        for label, wall in trace["candidate_walls_by_label"].items()
        if str(wall) == reference_wall
    ]
    other_wall_candidate_labels = [
        str(label)
        for label, wall in trace["candidate_walls_by_label"].items()
        if str(wall) != reference_wall
    ]
    assert len(same_wall_candidate_labels) == 2
    assert len(other_wall_candidate_labels) == 4
    assert len({str(spec["wall"]) for spec in candidates}) >= 2
    if query_id == "left_of_reference_on_wall":
        assert candidate_left_coords[answer_label] > reference_left_coord
        for label, coord in candidate_left_coords.items():
            if label != answer_label and label in same_wall_candidate_labels:
                assert coord <= reference_left_coord
    else:
        assert candidate_left_coords[answer_label] < reference_left_coord
        for label, coord in candidate_left_coords.items():
            if label != answer_label and label in same_wall_candidate_labels:
                assert coord >= reference_left_coord
    for bbox in trace["candidate_visible_bboxes_by_label"].values():
        width = float(bbox[2]) - float(bbox[0])
        height = float(bbox[3]) - float(bbox[1])
        assert width >= LETTERED_WALL_OBJECT_MIN_VISIBLE_PX
        assert height >= LETTERED_WALL_OBJECT_MIN_VISIBLE_PX
    assert (
        render_map["reference_object_bbox_px"]
        == render_map["object_bboxes_px"][str(reference["object_id"])]
    )


def test_room_wall_object_side_relation_registered() -> None:
    taxonomy = resolve_task_taxonomy(TASK_ID)

    assert is_default_dataset_task(TASK_ID)
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == SCENE_ID
    assert taxonomy.source_scene_id == ""
    assert SUPPORTED_QUERY_IDS == (
        "left_of_reference_on_wall",
        "right_of_reference_on_wall",
    )
