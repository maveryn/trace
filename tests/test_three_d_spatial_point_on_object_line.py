"""Tests for the synthetic 3D point-on-object-line task."""

from __future__ import annotations

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import list_default_task_ids
from trace_tasks.tasks.three_d.object_scene.point_on_object_line_label import TASK_ID
from tests.three_d_canvas_helpers import assert_three_d_canvas_contract


def test_point_on_object_line_answer_and_annotation() -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20260628,
        params={
            "scene_variant": "floor_grid_room",
            "point_count": 6,
            "object_count": 6,
            "context_object_count": 1,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=320,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    solver = trace["solver_trace"]
    marked_points = list(trace["marked_points"])
    answer_label = str(trace["answer_label"])

    assert output.scene_id == "object_scene"
    assert output.query_id == "single"
    assert_three_d_canvas_contract(output)
    assert output.answer_gt.type == "option_letter"
    assert output.answer_gt.value == answer_label
    assert output.annotation_gt.type == "point"
    assert output.annotation_gt.value == render_map["selected_point_px"]
    assert output.annotation_gt.value == render_map["marked_point_centers_px"][answer_label]
    assert output.trace_payload["projected_annotation"]["type"] == "point"
    assert output.trace_payload["projected_annotation"]["point"] == output.annotation_gt.value
    assert len(marked_points) == 6
    assert set(point["point_label"] for point in marked_points) == {"A", "B", "C", "D", "E", "F"}
    assert output.trace_payload["query_spec"]["params"]["answer_support"] == ["A", "B", "C", "D", "E", "F"]
    assert trace["reference_object_a_name"] in output.prompt
    assert trace["reference_object_b_name"] in output.prompt
    assert trace["reference_object_a_id"] != trace["reference_object_b_id"]
    assert float(solver["answer_line_distance_px"]) <= 18.0
    assert float(solver["line_distance_margin_px"]) >= 42.0
    assert output.trace_payload["witness_symbolic"]["ids_by_role"]["selected_point"] == str(trace["answer_point_id"])
    assert output.trace_payload["witness_symbolic"]["ids_by_role"]["reference_object_a"] == str(trace["reference_object_a_id"])
    assert output.trace_payload["witness_symbolic"]["ids_by_role"]["reference_object_b"] == str(trace["reference_object_b_id"])
    for label in {"A", "B", "C", "D", "E", "F"}:
        glyph_bbox = render_map["marked_point_glyph_bboxes_px"][label]
        label_bbox = render_map["marked_point_label_bboxes_px"][label]
        center = render_map["marked_point_centers_px"][label]
        assert glyph_bbox[0] <= center[0] <= glyph_bbox[2]
        assert glyph_bbox[1] <= center[1] <= glyph_bbox[3]
        assert label_bbox != glyph_bbox


def test_point_on_object_line_task_registered_in_three_d_taxonomy() -> None:
    taxonomy = resolve_task_taxonomy(TASK_ID)

    assert TASK_ID in list_default_task_ids()
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == "object_scene"
    assert not taxonomy.source_scene_id
