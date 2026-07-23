"""Tests for the synthetic 3D occlusion-order task."""

from __future__ import annotations

import trace_tasks.tasks  # noqa: F401 - registers tasks.
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks import create_task
from trace_tasks.tasks.registry import list_default_task_ids
from trace_tasks.tasks.three_d.object_scene.occlusion_order_label import OCCLUSION_CANDIDATE_SHAPE_TYPES, TASK_ID
from tests.three_d_option_panel_helpers import assert_option_panel_matches_candidates


def test_occlusion_order_answer_and_annotation() -> None:
    task = create_task(TASK_ID)
    output = task.generate(
        20260521,
        params={
            "scene_variant": "floor_grid_room",
            "point_count": 6,
            "context_object_count": 1,
            "post_image_noise_apply_prob": 0.0,
        },
        max_attempts=160,
    )

    trace = output.trace_payload["execution_trace"]
    point_specs = list(trace["point_specs"])
    context_specs = list(trace["context_object_specs"])
    occlusion_status = dict(trace["occlusion_status_by_label"])
    expected_labels = [str(label) for label, is_front in occlusion_status.items() if bool(is_front)]
    reference_id = str(trace["reference_object_id"])
    reference_spec = next(spec for spec in context_specs if str(spec["object_id"]) == reference_id)
    assert output.scene_id == "object_scene"
    assert output.query_id == "single"
    assert output.answer_gt.type == "option_letter"
    assert output.answer_gt.value == expected_labels[0]
    assert len(expected_labels) == 1
    assert len(point_specs) == 6
    assert len(context_specs) == 1
    assert all(spec["is_answer_candidate"] for spec in point_specs)
    assert not reference_spec["is_answer_candidate"]
    assert reference_id not in {str(spec["object_id"]) for spec in point_specs}
    assert reference_spec["nameable_for_prompt"]
    assert str(reference_spec["prompt_name"]) == str(trace["reference_object_name"])
    assert str(trace["reference_object_name"]) == "rectangular platform"
    assert str(reference_spec["shape_type"]) == "platform"
    assert str(reference_spec["occlusion_reference_role"]) == "solid_platform"
    answer_spec = next(spec for spec in point_specs if str(spec["point_label"]) == expected_labels[0])
    expected_bbox = output.trace_payload["render_map"]["object_bboxes_px"][str(answer_spec["object_id"])]
    expected_annotation_bbox = output.trace_payload["render_map"]["annotation_bboxes_px"][0]
    assert output.annotation_gt.type == "bbox"
    assert output.annotation_gt.value == expected_annotation_bbox
    assert output.trace_payload["render_map"]["point_bboxes_px"][expected_labels[0]] == expected_bbox
    assert_option_panel_matches_candidates(
        output,
        point_specs,
        answer_label=expected_labels[0],
        answer_object_id=str(answer_spec["object_id"]),
        expected_image_size=(1180, 1068),
    )

    overlap_by_label = dict(trace["candidate_reference_overlap_area_by_label"])
    depth_margin_by_label = dict(trace["candidate_depth_margin_to_reference_by_label"])
    assert float(overlap_by_label[expected_labels[0]]) >= 650.0
    assert float(depth_margin_by_label[expected_labels[0]]) >= 0.18
    assert float(trace["solver_trace"]["answer_front_gap_to_reference"]) >= 0.045
    assert all(
        float(overlap) <= 300.0
        for label, overlap in overlap_by_label.items()
        if str(label) != str(expected_labels[0])
    )
    assert trace["solver_trace"]["occluding_reference_labels"] == expected_labels
    assert trace["solver_trace"]["unique_occlusion_answer"] is True


def test_occlusion_order_uses_occlusion_safe_candidate_pool() -> None:
    assert "open_book" in OCCLUSION_CANDIDATE_SHAPE_TYPES
    assert "half_cylinder" in OCCLUSION_CANDIDATE_SHAPE_TYPES
    assert "apple" not in OCCLUSION_CANDIDATE_SHAPE_TYPES
    assert "carrot" not in OCCLUSION_CANDIDATE_SHAPE_TYPES
    assert "crown" not in OCCLUSION_CANDIDATE_SHAPE_TYPES
    assert "hat" not in OCCLUSION_CANDIDATE_SHAPE_TYPES
    assert "helmet" not in OCCLUSION_CANDIDATE_SHAPE_TYPES
    assert "star_prism" not in OCCLUSION_CANDIDATE_SHAPE_TYPES
    assert "clock" not in OCCLUSION_CANDIDATE_SHAPE_TYPES
    assert "ruler" not in OCCLUSION_CANDIDATE_SHAPE_TYPES

    task = create_task(TASK_ID)
    for index in range(8):
        output = task.generate(
            20260627 + index,
            params={
                "scene_variant": "floor_grid_room",
                "point_count": 6,
                "context_object_count": 1,
                "post_image_noise_apply_prob": 0.0,
            },
            max_attempts=240,
        )
        trace = output.trace_payload["execution_trace"]
        assert {
            str(spec["shape_type"])
            for spec in trace["point_specs"]
        }.issubset(set(OCCLUSION_CANDIDATE_SHAPE_TYPES))


def test_occlusion_order_task_registered_in_three_d_taxonomy() -> None:
    taxonomy = resolve_task_taxonomy(TASK_ID)

    assert TASK_ID in list_default_task_ids()
    assert taxonomy.domain == "three_d"
    assert taxonomy.scene_id == "object_scene"
    assert not taxonomy.source_scene_id
