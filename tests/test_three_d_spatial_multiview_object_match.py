"""Tests for the synthetic 3D multi-view object-match task."""

from __future__ import annotations

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import list_default_task_ids
from trace_tasks.tasks.three_d.object_scene.multiview_object_match_label import (
    CANDIDATE_VIEW_KEY,
    REFERENCE_VIEW_KEY,
    TASK_ID,
)
from tests.three_d_canvas_helpers import assert_three_d_canvas_contract


def test_multiview_object_match_answer_and_annotation() -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20260529,
        params={
            "scene_variant": "floor_grid_room",
            "point_count": 4,
            "context_object_count": 2,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=160,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    render_spec = output.trace_payload["render_spec"]
    solver_trace = dict(trace["solver_trace"])
    answer_label = str(output.answer_gt.value)
    target_object_id = str(trace["target_object_id"])
    candidate_labels = dict(solver_trace["candidate_labels_by_object_id"])

    assert output.scene_id == "object_scene"
    assert output.query_id == "single"
    assert output.answer_gt.type == "option_letter"
    assert answer_label == str(candidate_labels[target_object_id])
    assert output.annotation_gt.type == "bbox"
    assert output.annotation_gt.value == render_map["second_view_match_bbox_px"]
    assert output.trace_payload["projected_annotation"]["type"] == "bbox"
    assert output.trace_payload["projected_annotation"]["bbox"] == output.annotation_gt.value
    assert output.trace_payload["projected_annotation"]["pixel_bbox"] == output.annotation_gt.value
    assert len(trace["canonical_point_specs"]) == 4
    assert len(trace["canonical_context_object_specs"]) == 2
    assert {spec["shape_type"] for spec in trace["canonical_point_specs"]} == {trace["solver_trace"]["candidate_shape_type"]}
    assert len({tuple(spec["fill_rgb"]) for spec in trace["canonical_point_specs"]}) == 1
    assert {spec["anchor_part"] for spec in trace["canonical_context_object_specs"]} == {"platform", "corner_marker"}
    assert set(trace["views"]) == {REFERENCE_VIEW_KEY, CANDIDATE_VIEW_KEY}
    assert trace["views"][REFERENCE_VIEW_KEY]["camera"]["yaw_degrees"] != trace["views"][CANDIDATE_VIEW_KEY]["camera"]["yaw_degrees"]
    assert float(solver_trace["view_yaw_separation_degrees"]) >= 72.0
    assert solver_trace["same_object_unique_answer"] is True
    assert solver_trace["candidate_appearance_control"] == "same_type_same_color"
    assert solver_trace["anchor_structure"] == "low_rectangular_platform_with_corner_block"
    reference_panel = render_spec["panel_layout"][REFERENCE_VIEW_KEY]
    candidate_panel = render_spec["panel_layout"][CANDIDATE_VIEW_KEY]
    source_width = int(render_spec["scene_canvas_width"])
    source_height = int(render_spec["scene_canvas_height"])
    assert int(reference_panel["source_width"]) == source_width
    assert int(reference_panel["source_height"]) == source_height
    assert int(candidate_panel["width"]) == int(reference_panel["width"])
    assert int(candidate_panel["height"]) == int(reference_panel["height"])
    assert abs((float(reference_panel["width"]) / float(reference_panel["height"])) - (float(source_width) / float(source_height))) < 0.01
    assert any(
        entity["entity_type"] == "red_reference_box"
        for entity in output.trace_payload["scene_ir"]["entities"]
        if str(entity["entity_id"]).startswith(f"{REFERENCE_VIEW_KEY}:")
    )
    assert_three_d_canvas_contract(output)


def test_multiview_object_match_task_registered_in_three_d_taxonomy() -> None:
    taxonomy = resolve_task_taxonomy(TASK_ID)

    assert TASK_ID in list_default_task_ids()
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == "object_scene"
    assert not taxonomy.source_scene_id
