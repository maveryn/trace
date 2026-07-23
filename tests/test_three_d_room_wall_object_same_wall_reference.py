"""Tests for the 3D room wall-object same-wall reference task."""

from __future__ import annotations

import pytest

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import is_default_dataset_task
from trace_tasks.tasks.three_d.room.wall_object_camera_distance_label import (
    LETTERED_WALL_OBJECT_MIN_VISIBLE_PX,
)
from trace_tasks.tasks.three_d.room.wall_object_same_wall_reference_label import (
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


@pytest.mark.parametrize("reference_wall", ["back", "left", "right"])
def test_room_wall_object_same_wall_reference_answer_annotation_and_unique_reference(
    reference_wall: str,
) -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20260522,
        params={
            "query_id": "single",
            "scene_variant": "studio_room",
            "candidate_count": 6,
            "context_wall_count": 0,
            "floor_context_count": 6,
            "reference_wall": reference_wall,
            "reference_object_type": "mirror",
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=180,
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
    same_wall_labels = [
        label
        for label, is_same_wall in trace["same_wall_as_reference_by_label"].items()
        if bool(is_same_wall)
    ]
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
        expected_image_size=(1180, 1068),
    )
    assert trace["target_object_ids"] == [str(answer_spec["object_id"])]
    assert same_wall_labels == [answer_label]
    assert str(reference["wall"]) == reference_wall
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
    assert {str(spec["wall"]) for spec in candidates} == {"back", "left", "right"}
    _assert_side_wall_objects_are_not_deep(trace)
    assert sum(1 for spec in candidates if str(spec["wall"]) == reference_wall) == 1
    assert trace["candidate_walls_by_label"][answer_label] == reference_wall
    for bbox in trace["candidate_visible_bboxes_by_label"].values():
        width = float(bbox[2]) - float(bbox[0])
        height = float(bbox[3]) - float(bbox[1])
        assert width >= LETTERED_WALL_OBJECT_MIN_VISIBLE_PX
        assert height >= LETTERED_WALL_OBJECT_MIN_VISIBLE_PX
    assert (
        render_map["reference_object_bbox_px"]
        == render_map["object_bboxes_px"][str(reference["object_id"])]
    )


def test_room_wall_object_same_wall_reference_registered() -> None:
    taxonomy = resolve_task_taxonomy(TASK_ID)

    assert is_default_dataset_task(TASK_ID)
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == SCENE_ID
    assert taxonomy.source_scene_id == ""
    assert SUPPORTED_QUERY_IDS == ("single",)
