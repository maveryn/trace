"""Tests for 3D object-scene marked-point ordering tasks."""

from __future__ import annotations

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import is_default_dataset_task
from trace_tasks.tasks.three_d.object_scene.point_camera_distance_order_label import (
    MIN_CAMERA_DISTANCE_MARGIN,
    MIN_SCREEN_DEPTH_STEP_PX,
    TASK_ID as CAMERA_DISTANCE_ORDER_TASK_ID,
)
from tests.three_d_canvas_helpers import assert_three_d_canvas_contract


def _descriptor_for_answer(output) -> str:
    choices = output.trace_payload["render_map"]["option_choices"]
    answer = str(output.answer_gt.value)
    return str(next(choice["descriptor"] for choice in choices if str(choice["label"]) == answer))


def _assert_order_task_common(output, task_id: str) -> None:
    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]

    assert output.scene_id == "object_scene"
    assert output.query_id == "single"
    assert_three_d_canvas_contract(output)
    assert output.answer_gt.type == "option_letter"
    assert output.annotation_gt.type == "point_map"
    assert set(output.annotation_gt.value) == {"P", "Q", "R"}
    assert output.annotation_gt.value == render_map["annotation_point_map_px"]
    assert output.trace_payload["projected_annotation"]["type"] == "point_map"
    assert output.trace_payload["projected_annotation"]["point_map"] == output.annotation_gt.value
    assert output.trace_payload["projected_annotation"]["pixel_point_map"] == output.annotation_gt.value
    assert set(trace["point_labels"]) == {"P", "Q", "R"}
    assert {str(point["point_label"]) for point in trace["marked_points"]} == {"P", "Q", "R"}
    assert output.trace_payload["query_spec"]["params"]["answer_support"] == ["A", "B", "C", "D", "E", "F"]
    assert len(render_map["option_choices"]) == 6
    assert {str(choice["label"]) for choice in render_map["option_choices"]} == {"A", "B", "C", "D", "E", "F"}
    assert len({str(choice["descriptor"]) for choice in render_map["option_choices"]}) == 6
    assert _descriptor_for_answer(output) == str(trace["answer_descriptor"])
    assert output.trace_payload["witness_symbolic"]["answer_order"] == trace["answer_order"]
    assert is_default_dataset_task(task_id)
    taxonomy = resolve_task_taxonomy(task_id)
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == "object_scene"
    assert not taxonomy.source_scene_id


def test_point_camera_distance_order_answer_and_annotation() -> None:
    task = create_task(CAMERA_DISTANCE_ORDER_TASK_ID)
    output = task.generate(
        20260628,
        params={
            "scene_variant": "floor_grid_room",
            "point_count": 3,
            "context_object_count": 4,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=280,
    )

    _assert_order_task_common(output, CAMERA_DISTANCE_ORDER_TASK_ID)
    trace = output.trace_payload["execution_trace"]
    marked_points = list(trace["marked_points"])
    expected_order = [
        str(point["point_label"])
        for point in sorted(marked_points, key=lambda point: (float(point["camera_distance"]), str(point["point_label"])))
    ]
    screen_order = [
        str(point["point_label"])
        for point in sorted(marked_points, key=lambda point: (float(point["screen_xy"][1]), str(point["point_label"])), reverse=True)
    ]

    assert trace["answer_order"] == expected_order
    assert trace["solver_trace"]["camera_distance_order_near_to_far"] == expected_order
    assert trace["solver_trace"]["screen_y_order_front_to_back"] == screen_order
    assert expected_order == screen_order
    assert float(trace["solver_trace"]["unique_camera_distance_margin"]) >= MIN_CAMERA_DISTANCE_MARGIN
    assert float(trace["solver_trace"]["unique_screen_depth_margin_px"]) >= MIN_SCREEN_DEPTH_STEP_PX
    assert "nearest" in output.prompt.lower() or "closest" in output.prompt.lower()
