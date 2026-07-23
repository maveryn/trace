from __future__ import annotations

import math

import pytest

from trace_tasks.tasks.physics.vernier_caliper.length_readout_value import (
    PhysicsVernierCaliperLengthReadoutValueTask,
)


def _assert_bbox_in_bounds(bbox: list[float], *, width: int, height: int) -> None:
    assert len(bbox) == 4
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


def _assert_point_in_bounds(point: list[float], *, width: int, height: int) -> None:
    assert len(point) == 2
    x, y = [float(value) for value in point]
    assert 0 <= x <= width
    assert 0 <= y <= height


def test_vernier_caliper_length_readout_contract() -> None:
    out = PhysicsVernierCaliperLengthReadoutValueTask().generate(
        80304,
        params={"main_mm": 23, "aligned_vernier_tick": 4, "correct_option_letter": "D"},
        max_attempts=20,
    )
    execution = out.trace_payload["execution_trace"]
    query_params = out.trace_payload["query_spec"]["params"]
    render_map = out.trace_payload["render_map"]

    assert out.scene_id == "vernier_caliper"
    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "D"
    assert execution["query_id"] == "single"
    assert execution["internal_query_id"] == "main_scale_vernier_mm"
    assert query_params["query_id"] == "single"
    assert query_params["internal_query_id"] == "main_scale_vernier_mm"
    assert query_params["answer_support"] == ["A", "B", "C", "D", "E", "F"]
    assert execution["main_mm"] == 23
    assert execution["aligned_vernier_tick"] == 4
    assert math.isclose(float(execution["target_readout_mm"]), 23.4, abs_tol=1e-9)
    assert execution["target_answer"] == "D"
    assert execution["correct_option_letter"] == "D"
    assert math.isclose(float(execution["option_values_mm"]["D"]), 23.4, abs_tol=1e-9)
    assert math.isclose(float(render_map["answer_mm"]), 23.4, abs_tol=1e-9)
    assert render_map["correct_option_letter"] == "D"
    assert math.isclose(float(render_map["option_values_mm"]["D"]), 23.4, abs_tol=1e-9)
    assert set(render_map["option_bboxes_px"]) == {"A", "B", "C", "D", "E", "F"}
    assert render_map["correct_option_bbox_px"] == render_map["option_bboxes_px"]["D"]
    assert render_map["nearest_aligned_main_tick"] == 27
    assert render_map["main_scale_max_mm"] == 62

    assert out.annotation_gt.type == "bbox"
    assert out.annotation_gt.value == render_map["correct_option_bbox_px"]
    assert render_map["annotation_source"] == "selected_option_bbox_px"
    assert set(render_map["readout_witness_point_map_px"]) == {
        "vernier_zero_tick",
        "aligned_vernier_tick",
    }
    assert set(render_map["context_bbox_map_px"]) == {
        "main_scale_region",
        "vernier_zero",
        "vernier_scale_region",
        "aligned_vernier_tick",
    }
    width, height = out.image.size
    for point in render_map["readout_witness_point_map_px"].values():
        _assert_point_in_bounds(point, width=width, height=height)
    _assert_bbox_in_bounds(out.annotation_gt.value, width=width, height=height)
    for bbox in render_map["option_bboxes_px"].values():
        _assert_bbox_in_bounds(bbox, width=width, height=height)
    for bbox in render_map["context_bbox_map_px"].values():
        _assert_bbox_in_bounds(bbox, width=width, height=height)
    assert out.trace_payload["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert out.trace_payload["query_spec"]["prompt_variant"]["prompt_schema_version"] == "v1"
    assert out.prompt_variants["answer_only"]
    assert out.prompt_variants["answer_and_annotation"]


def test_vernier_caliper_single_query_validation() -> None:
    task = PhysicsVernierCaliperLengthReadoutValueTask()
    out = task.generate(80305, params={"query_id": "single"}, max_attempts=20)
    assert out.query_id == "single"

    with pytest.raises(ValueError, match="query_id"):
        task.generate(
            80305,
            params={"query_id": "main_scale_vernier_mm"},
            max_attempts=20,
        )


def test_vernier_caliper_target_answer_overrides_are_consistent() -> None:
    out = PhysicsVernierCaliperLengthReadoutValueTask().generate(
        80306,
        params={"target_answer": 32.7, "correct_option_letter": "B"},
        max_attempts=20,
    )
    execution = out.trace_payload["execution_trace"]
    assert execution["main_mm"] == 32
    assert execution["aligned_vernier_tick"] == 7
    assert out.answer_gt.value == "B"
    assert math.isclose(float(execution["option_values_mm"]["B"]), 32.7, abs_tol=1e-9)
