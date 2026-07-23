"""Tests for the 3D warehouse robot forward-path task."""

from __future__ import annotations

import pytest

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import list_default_task_ids
from trace_tasks.tasks.three_d.warehouse.robot_forward_path_label import (
    MIN_FIRST_OBJECT_MARGIN,
    SUPPORTED_QUERY_IDS,
    TASK_ID,
)
from trace_tasks.tasks.three_d.warehouse.shared.state import (
    SCENE_ID,
    SUPPORTED_ROBOT_DESIGNS,
    SUPPORTED_ROBOT_HEADINGS,
    SUPPORTED_SHELF_RACK_STYLES,
)
from tests.three_d_option_panel_helpers import assert_option_panel_matches_candidates


@pytest.mark.parametrize(
    ("scene_variant", "robot_heading"),
    [
        ("storage_aisle", "east"),
        ("loading_zone", "north"),
        ("packing_floor", "west"),
    ],
)
def test_warehouse_robot_forward_path_answer_annotation_and_geometry(
    scene_variant: str,
    robot_heading: str,
) -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20260524,
        params={
            "query_id": "single",
            "scene_variant": scene_variant,
            "robot_heading": robot_heading,
            "candidate_count": 4,
            "context_object_count": 11,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=420,
    )

    trace = output.trace_payload["execution_trace"]
    render_map = output.trace_payload["render_map"]
    entities = output.trace_payload["scene_ir"]["entities"]
    candidates = list(trace["candidate_object_specs"])
    reference = dict(trace["reference_object"])
    answer_label = str(trace["answer_label"])
    answer_spec = next(spec for spec in candidates if str(spec["point_label"]) == answer_label)
    expected_bbox = render_map["object_bboxes_px"][str(answer_spec["object_id"])]
    first_labels = [
        str(label)
        for label, flag in trace["first_reached_by_label"].items()
        if bool(flag)
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
        expected_image_size=(1180, 1088),
    )
    assert trace["target_object_ids"] == [str(answer_spec["object_id"])]
    assert first_labels == [answer_label]
    assert trace["first_reached_candidate_labels"] == [answer_label]
    assert answer_label in trace["forward_path_candidate_labels"]
    assert float(trace["first_reached_margin"]) >= MIN_FIRST_OBJECT_MARGIN
    assert bool(answer_spec["is_in_forward_path_corridor"]) is True
    assert bool(answer_spec["is_first_reached_object"]) is True
    assert str(reference["object_type"]) == "warehouse_robot"
    assert str(reference["object_role"]) == "warehouse_reference_robot"
    assert str(reference["robot_design"]) in SUPPORTED_ROBOT_DESIGNS
    answer_entity = next(entity for entity in entities if str(entity["entity_id"]) == str(answer_spec["object_id"]))
    answer_record = answer_entity["attrs"]["object_record"]
    assert answer_record["object_id"] == str(answer_spec["object_id"])
    assert answer_record["object_type"] == str(answer_spec["object_type"])
    assert answer_record["visual_attributes"]["renderer_id"] == "warehouse_object"
    assert answer_record["visual_attributes"]["renderer_style"] == "projected_3d"
    assert len(reference["robot_base_rgb"]) == 3
    assert len(reference["robot_accent_rgb"]) == 3
    assert str(trace["robot_heading"]) == str(robot_heading)
    assert str(trace["robot_design"]) == str(reference["robot_design"])
    assert len(trace["travel_direction_vector_xy"]) == 2
    assert sorted(str(spec["point_label"]) for spec in candidates) == list("ABCD")
    assert len(candidates) == 4
    assert len(trace["context_object_specs"]) == 11
    assert len(trace["reference_object_specs"]) == 1
    shelf_specs = [spec for spec in trace["context_object_specs"] if str(spec["object_type"]) == "shelf_rack"]
    assert shelf_specs
    assert all(str(spec["shelf_style"]) in SUPPORTED_SHELF_RACK_STYLES for spec in shelf_specs)
    assert all(int(spec["shelf_levels"]) >= 2 for spec in shelf_specs)
    assert all(len(spec["shelf_frame_rgb"]) == 3 for spec in shelf_specs)
    assert all(0.50 <= float(spec["shelf_height_scale"]) <= 1.30 for spec in shelf_specs)
    assert str(reference["object_id"]) in render_map["object_bboxes_px"]
    assert len(set(str(spec["object_type"]) for spec in candidates)) >= 4
    marker_entity = next(
        entity
        for entity in entities
        if str(entity["entity_type"]) == "three_d_warehouse_robot_reference_marker"
    )
    assert marker_entity["attrs"]["reference_marker"] == "red_bbox"
    assert marker_entity["attrs"]["reference_marker_bbox_px"] is not None
    assert marker_entity["attrs"]["reference_direction_marker_bbox_px"] is not None
    assert "red-boxed robot" in output.prompt
    assert "arrow" in output.prompt
    assert "{answer_hint}" not in output.prompt


def test_warehouse_robot_forward_path_registered() -> None:
    taxonomy = resolve_task_taxonomy(TASK_ID)

    assert TASK_ID in list_default_task_ids()
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == SCENE_ID
    assert taxonomy.source_scene_id == ""
    assert SUPPORTED_QUERY_IDS == ("first_object_ahead",)
    assert SUPPORTED_ROBOT_HEADINGS == ("east", "north", "west", "south")
    assert SUPPORTED_ROBOT_DESIGNS == ("low_cart", "sensor_tower", "stacker_bot")
    assert SUPPORTED_SHELF_RACK_STYLES == ("open_frame", "loaded_bins", "mixed_crates", "tall_sparse", "heavy_low")
