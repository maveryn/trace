"""Tests for the synthetic 3D marked-point depth task."""

from __future__ import annotations

import pytest

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import list_default_task_ids
from trace_tasks.tasks.three_d.object_scene.marked_point_depth_extremum_label import TASK_ID
from tests.three_d_canvas_helpers import assert_three_d_canvas_contract


@pytest.mark.parametrize("query_id", ["closest_marked_point", "farthest_marked_point"])
def test_marked_point_depth_answer_and_annotation(query_id: str) -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20260531,
        params={
            "query_id": query_id,
            "scene_variant": "floor_grid_room",
            "point_count": 6,
            "context_object_count": 6,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=220,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    marked_points = list(trace["marked_points"])
    sorted_points = sorted(marked_points, key=lambda point: (float(point["camera_distance"]), str(point["point_label"])))
    expected = sorted_points[0] if query_id == "closest_marked_point" else sorted_points[-1]
    expected_label = str(expected["point_label"])

    assert output.scene_id == "object_scene"
    assert output.query_id == query_id
    assert_three_d_canvas_contract(output)
    assert output.answer_gt.type == "option_letter"
    assert output.answer_gt.value == expected_label
    assert output.annotation_gt.type == "point"
    assert output.annotation_gt.value == render_map["selected_point_px"]
    assert output.annotation_gt.value == render_map["marked_point_centers_px"][expected_label]
    assert output.trace_payload["projected_annotation"]["type"] == "point"
    assert output.trace_payload["projected_annotation"]["point"] == output.annotation_gt.value
    assert output.trace_payload["projected_annotation"]["pixel_point"] == output.annotation_gt.value
    assert trace["point_specs"] == []
    assert len(marked_points) == 6
    assert len(trace["context_object_specs"]) == 6
    assert set(point["point_label"] for point in marked_points) == {"A", "B", "C", "D", "E", "F"}
    assert output.trace_payload["query_spec"]["params"]["answer_support"] == ["A", "B", "C", "D", "E", "F"]
    assert {str(point["surface_kind"]) for point in marked_points} == {"floor"}
    screen_depth_order = sorted(marked_points, key=lambda point: (float(point["screen_xy"][1]), str(point["point_label"])), reverse=True)
    expected_by_screen_depth = screen_depth_order[0] if query_id == "closest_marked_point" else screen_depth_order[-1]
    assert str(expected_by_screen_depth["point_label"]) == expected_label
    assert float(trace["solver_trace"]["answer_screen_depth_margin_px"]) >= 32.0
    assert output.trace_payload["scene_ir"]["relations"]["answer_point_id"] == str(expected["point_id"])
    assert output.trace_payload["witness_symbolic"]["ids_by_role"]["selected_point"] == str(expected["point_id"])
    assert render_map["marked_point_label_bboxes_px"][expected_label] != render_map["marked_point_glyph_bboxes_px"][expected_label]
    assert render_map["marked_point_circle_bboxes_px"][expected_label] == render_map["marked_point_glyph_bboxes_px"][expected_label]
    for label in {"A", "B", "C", "D", "E", "F"}:
        glyph_bbox = render_map["marked_point_glyph_bboxes_px"][label]
        label_bbox = render_map["marked_point_label_bboxes_px"][label]
        center = render_map["marked_point_centers_px"][label]
        assert glyph_bbox[0] <= center[0] <= glyph_bbox[2]
        assert glyph_bbox[1] <= center[1] <= glyph_bbox[3]
        assert label_bbox != glyph_bbox
    assert "marked" in output.prompt.lower()


def test_marked_point_depth_task_registered_in_three_d_taxonomy() -> None:
    taxonomy = resolve_task_taxonomy(TASK_ID)

    assert TASK_ID in list_default_task_ids()
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == "object_scene"
    assert not taxonomy.source_scene_id
