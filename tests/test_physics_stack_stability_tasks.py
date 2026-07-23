"""Contract tests for the migrated stack-stability physics scene."""

from __future__ import annotations

import trace_tasks.tasks  # noqa: F401
from trace_tasks.core.scene_config import get_scene_defaults
from trace_tasks.tasks.physics.stack_stability.stability_status_label import (
    PhysicsStackStabilityStatusLabelTask,
)
from trace_tasks.tasks.shared.config_defaults import split_generation_rendering_prompt_defaults


def _assert_bbox_in_bounds(out) -> None:
    width, height = out.image.size
    bbox = out.annotation_gt.value
    assert out.annotation_gt.type == "bbox"
    assert 0 <= bbox[0] < bbox[2] <= width
    assert 0 <= bbox[1] < bbox[3] <= height


def _assert_bbox_contains(outer, inner) -> None:
    assert float(outer[0]) <= float(inner[0])
    assert float(outer[1]) <= float(inner[1])
    assert float(outer[2]) >= float(inner[2])
    assert float(outer[3]) >= float(inner[3])


def _assert_selected_geometry_matches_status(out, *, expected_status: str) -> None:
    render_map = out.trace_payload["render_map"]
    label = str(out.answer_gt.value)
    com_x = float(render_map["candidate_com_points_px"][label][0])
    support_bbox = render_map["candidate_support_bboxes_px"][label]
    support_left = float(support_bbox[0])
    support_right = float(support_bbox[2])

    assert render_map["candidate_statuses"][label] == expected_status
    if expected_status == "stable":
        assert support_left < com_x < support_right
    else:
        assert com_x < support_left or com_x > support_right


def test_stack_stability_stable_label_contract() -> None:
    out = PhysicsStackStabilityStatusLabelTask().generate(
        94111,
        params={"query_id": "stable_stack_label", "correct_option_letter": "D"},
        max_attempts=10,
    )
    statuses = out.trace_payload["render_map"]["candidate_statuses"]

    assert out.scene_id == "stack_stability"
    assert out.query_id == "stable_stack_label"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "D"
    assert list(statuses.values()).count("stable") == 1
    assert statuses["D"] == "stable"
    _assert_bbox_in_bounds(out)
    _assert_selected_geometry_matches_status(out, expected_status="stable")
    assert out.trace_payload["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert out.prompt_variants["answer_only"]
    assert out.prompt_variants["answer_and_annotation"]


def test_stack_stability_tipping_label_contract() -> None:
    out = PhysicsStackStabilityStatusLabelTask().generate(
        94121,
        params={"query_id": "tipping_stack_label", "correct_option_letter": "B"},
        max_attempts=10,
    )
    statuses = out.trace_payload["render_map"]["candidate_statuses"]

    assert out.scene_id == "stack_stability"
    assert out.query_id == "tipping_stack_label"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "B"
    assert list(statuses.values()).count("tipping") == 1
    assert statuses["B"] == "tipping"
    _assert_bbox_in_bounds(out)
    _assert_selected_geometry_matches_status(out, expected_status="tipping")


def test_stack_stability_annotation_wraps_selected_stack_witnesses() -> None:
    out = PhysicsStackStabilityStatusLabelTask().generate(
        94131,
        params={},
        max_attempts=10,
    )
    render_map = out.trace_payload["render_map"]
    selected_label = str(out.answer_gt.value)
    annotation_bbox = out.annotation_gt.value

    assert set(render_map["candidate_statuses"]) == {"A", "B", "C", "D", "E", "F"}
    assert out.answer_gt.value in render_map["candidate_statuses"]
    _assert_bbox_contains(annotation_bbox, render_map["candidate_stack_bboxes_px"][selected_label])
    for witness_bbox in render_map["selected_witness_bboxes_px"].values():
        _assert_bbox_contains(annotation_bbox, witness_bbox)
    assert render_map["annotation_bbox_px"] == annotation_bbox


def test_stack_stability_scene_config_uses_v1_scene_defaults() -> None:
    cfg = get_scene_defaults("physics", "stack_stability")
    generation, rendering, prompt = split_generation_rendering_prompt_defaults(
        cfg,
        task_id="task_physics__stack_stability__stability_status_label",
    )

    assert "query_id_weights" not in generation
    assert "balanced_query_id_sampling" not in generation
    assert set(generation["correct_option_letter_weights"]) == {
        "A",
        "B",
        "C",
        "D",
        "E",
        "F",
    }
    assert bool(generation["balanced_correct_option_letter_sampling"]) is True
    assert int(rendering["canvas_width"]) == 1180
    assert int(rendering["canvas_height"]) == 780
    assert int(rendering["brick_width_px"]) == 80
    assert int(rendering["projection_width_px"]) == 4
    assert str(prompt["bundle_id"]) == "physics_stack_stability_v1"
    assert str(prompt["task_key"]) == "stack_stability_status_label_query"
