"""Tests for synthetic 3D marked-point geometry tasks."""

from __future__ import annotations

import pytest

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import is_default_dataset_task
from trace_tasks.tasks.three_d.object_scene.line_side_label import (
    MIN_SIDE_DISTANCE_PX,
    SUPPORTED_QUERY_IDS as LINE_SIDE_QUERY_IDS,
    TASK_ID as LINE_SIDE_TASK_ID,
)
from trace_tasks.tasks.three_d.object_scene.reference_triangle_inside_label import (
    MIN_INSIDE_MARGIN_PX,
    MIN_OUTSIDE_MARGIN_PX,
    TASK_ID as TRIANGLE_TASK_ID,
)
from tests.three_d_canvas_helpers import assert_three_d_canvas_contract


def _assert_marked_point_contract(output) -> None:
    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    answer_label = str(trace["answer_label"])

    assert output.scene_id == "object_scene"
    assert_three_d_canvas_contract(output)
    assert output.answer_gt.type == "option_letter"
    assert output.answer_gt.value == answer_label
    assert output.annotation_gt.type == "point"
    assert output.annotation_gt.value == render_map["selected_point_px"]
    assert output.annotation_gt.value == render_map["marked_point_centers_px"][answer_label]
    assert output.trace_payload["projected_annotation"]["type"] == "point"
    assert output.trace_payload["projected_annotation"]["point"] == output.annotation_gt.value
    assert len(trace["marked_points"]) == 6
    assert set(point["point_label"] for point in trace["marked_points"]) == {"A", "B", "C", "D", "E", "F"}
    assert output.trace_payload["query_spec"]["params"]["answer_support"] == ["A", "B", "C", "D", "E", "F"]
    for label in {"A", "B", "C", "D", "E", "F"}:
        glyph_bbox = render_map["marked_point_glyph_bboxes_px"][label]
        label_bbox = render_map["marked_point_label_bboxes_px"][label]
        center = render_map["marked_point_centers_px"][label]
        assert glyph_bbox[0] <= center[0] <= glyph_bbox[2]
        assert glyph_bbox[1] <= center[1] <= glyph_bbox[3]
        assert label_bbox != glyph_bbox


@pytest.mark.parametrize("query_id", LINE_SIDE_QUERY_IDS)
def test_line_side_label_answer_and_annotation(query_id: str) -> None:
    task = create_task(LINE_SIDE_TASK_ID)
    output = task.generate(
        20260628,
        params={
            "query_id": str(query_id),
            "scene_variant": "floor_grid_room",
            "point_count": 6,
            "object_count": 6,
            "context_object_count": 1,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=512,
    )

    _assert_marked_point_contract(output)
    trace = output.trace_payload["execution_trace"]
    solver = trace["solver_trace"]
    answer_label = str(trace["answer_label"])
    requested_side = str(solver["requested_side"])
    requested_sign = 1 if requested_side == "left" else -1
    signed_distances = solver["signed_side_distances_px_by_label"]

    assert output.query_id == str(query_id)
    assert trace["reference_object_a_name"] in output.prompt
    assert trace["reference_object_b_name"] in output.prompt
    assert requested_side in output.prompt.lower()
    assert requested_sign * float(signed_distances[answer_label]) >= MIN_SIDE_DISTANCE_PX
    for label, distance in signed_distances.items():
        if str(label) == answer_label:
            continue
        assert requested_sign * float(distance) <= -MIN_SIDE_DISTANCE_PX
    assert output.trace_payload["witness_symbolic"]["ids_by_role"]["selected_point"] == str(trace["answer_point_id"])
    assert output.trace_payload["witness_symbolic"]["ids_by_role"]["reference_object_a"] == str(trace["reference_object_a_id"])
    assert output.trace_payload["witness_symbolic"]["ids_by_role"]["reference_object_b"] == str(trace["reference_object_b_id"])


def test_reference_triangle_inside_label_answer_and_annotation() -> None:
    task = create_task(TRIANGLE_TASK_ID)
    output = task.generate(
        20260628,
        params={
            "scene_variant": "floor_grid_room",
            "point_count": 6,
            "object_count": 6,
            "context_object_count": 1,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=640,
    )

    _assert_marked_point_contract(output)
    trace = output.trace_payload["execution_trace"]
    solver = trace["solver_trace"]
    answer_label = str(trace["answer_label"])
    margins = solver["triangle_margins_px_by_label"]

    assert output.query_id == "single"
    assert trace["reference_object_a_name"] in output.prompt
    assert trace["reference_object_b_name"] in output.prompt
    assert trace["reference_object_c_name"] in output.prompt
    assert float(margins[answer_label]) >= MIN_INSIDE_MARGIN_PX
    for label, margin in margins.items():
        if str(label) == answer_label:
            continue
        assert float(margin) <= -MIN_OUTSIDE_MARGIN_PX
    assert output.trace_payload["witness_symbolic"]["ids_by_role"]["selected_point"] == str(trace["answer_point_id"])
    assert output.trace_payload["witness_symbolic"]["ids_by_role"]["reference_object_a"] == str(trace["reference_object_a_id"])
    assert output.trace_payload["witness_symbolic"]["ids_by_role"]["reference_object_b"] == str(trace["reference_object_b_id"])
    assert output.trace_payload["witness_symbolic"]["ids_by_role"]["reference_object_c"] == str(trace["reference_object_c_id"])


def test_marked_point_geometry_tasks_registered_in_three_d_taxonomy() -> None:
    for task_id in (LINE_SIDE_TASK_ID, TRIANGLE_TASK_ID):
        taxonomy = resolve_task_taxonomy(task_id)
        assert is_default_dataset_task(task_id)
        assert taxonomy.domain == "three_d"
        assert taxonomy.scene_id == "object_scene"
        assert not taxonomy.source_scene_id
