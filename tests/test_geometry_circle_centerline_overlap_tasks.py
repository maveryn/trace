"""Regression tests for circle centerline overlap geometry tasks."""

from __future__ import annotations

import json

import pytest

from trace_tasks.tasks import TASK_REGISTRY, create_task
from trace_tasks.tasks.geometry.circle_centerline_overlap.segment_length_value import (
    BOUNDARY_PAIRS,
    BOUNDARY_TARGET_ROLES,
    GeometryCircleCenterlineOverlapSegmentLengthValueTask,
    TASK_ID,
    QUERY_ID_BOUNDARY_SEGMENT,
    QUERY_ID_CENTER_DISTANCE,
    SCENE_ID,
    _segment_length,
)
from trace_tasks.tasks.geometry.circle_centerline_overlap.shared.state import CircleOverlapCase
from trace_tasks.tasks.geometry.circle_centerline_overlap.shared.construction import (
    center_distance_length,
    generated_overlap_cases,
    select_boundary_pair,
    select_boundary_target_role,
    select_circle_count,
    select_center_distance_overlap_case,
    select_overlap_case,
)


def _generate(seed: int, **params):
    task = create_task(TASK_ID)
    return task.generate(seed, params=dict(params), max_attempts=80)


def test_circle_centerline_overlap_task_registered() -> None:
    assert TASK_ID in TASK_REGISTRY
    assert TASK_REGISTRY[TASK_ID] is GeometryCircleCenterlineOverlapSegmentLengthValueTask


@pytest.mark.parametrize("query_id", [QUERY_ID_CENTER_DISTANCE, QUERY_ID_BOUNDARY_SEGMENT])
def test_circle_centerline_overlap_queries_emit_segment_annotation(query_id: str) -> None:
    out = _generate(20260701, query_id=query_id)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == query_id
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == execution["answer"]
    assert out.annotation_gt.type == "segment"
    assert trace["projected_annotation"]["segment"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_segment"] == out.annotation_gt.value
    assert trace["render_spec"]["prompt"]["prompt_variant"]["prompt_bundle_id"] == "geometry_circle_centerline_overlap_v1"
    technical_style = trace["render_spec"]["style"]["technical_diagram"]
    background_style = trace["render_spec"]["style"]["background"]["style_spec"]["background_style"]
    assert trace["render_spec"]["style"]["font_bold"] is False
    assert trace["render_spec"]["style"]["label_stroke_width"] == 0
    assert technical_style["selection"]["require_grid"] is False
    assert technical_style["grid_style"]["kind"] == "none"
    assert background_style["kind"] != "graph_paper"
    assert "task_variant" not in json.dumps(trace)
    _assert_segment_inside_image(out.annotation_gt.value, out.image.size)


def test_circle_centerline_overlap_center_distance_formula() -> None:
    out = _generate(
        20260702,
        query_id=QUERY_ID_CENTER_DISTANCE,
        overlap_case=(5, 15, 5, 2, 2),
        circle_count=3,
    )
    execution = out.trace_payload["execution_trace"]

    assert out.answer_gt.value == 36
    assert execution["distance_ab"] == execution["radius_a"] + execution["radius_b"] - execution["overlap_ab"]
    assert execution["distance_bc"] == execution["radius_b"] + execution["radius_c"] - execution["overlap_bc"]
    assert execution["answer"] == execution["distance_ac"] == execution["distance_ab"] + execution["distance_bc"]
    assert out.annotation_gt.value == [
        out.trace_payload["render_map"]["centers"]["A"],
        out.trace_payload["render_map"]["centers"]["C"],
    ]
    assert execution["circle_count"] == 3
    assert "three overlapping circles" in out.prompt
    assert "radius labels" in out.prompt


def test_circle_centerline_overlap_two_circle_center_distance_formula() -> None:
    out = _generate(
        20260702,
        query_id=QUERY_ID_CENTER_DISTANCE,
        overlap_case=(5, 15, 2),
        circle_count=2,
    )
    execution = out.trace_payload["execution_trace"]

    assert out.answer_gt.value == 18
    assert execution["circle_count"] == 2
    assert execution["distance_ab"] == execution["radius_a"] + execution["radius_b"] - execution["overlap_ab"]
    assert execution["answer"] == execution["distance_ab"]
    assert out.annotation_gt.value == [
        out.trace_payload["render_map"]["centers"]["A"],
        out.trace_payload["render_map"]["centers"]["B"],
    ]
    assert "two overlapping circles" in out.prompt
    assert "C" not in out.trace_payload["render_map"]["centers"]


@pytest.mark.parametrize("boundary_pair", BOUNDARY_PAIRS)
@pytest.mark.parametrize("boundary_target_role", BOUNDARY_TARGET_ROLES)
def test_circle_centerline_overlap_boundary_segment_formula(boundary_pair: str, boundary_target_role: str) -> None:
    out = _generate(
        20260703,
        query_id=QUERY_ID_BOUNDARY_SEGMENT,
        overlap_case=(6, 14, 8, 3, 4),
        boundary_pair=boundary_pair,
        boundary_target_role=boundary_target_role,
    )
    execution = out.trace_payload["execution_trace"]
    expected = _segment_length(CircleOverlapCase(6, 14, 8, 3, 4), boundary_pair, boundary_target_role)

    assert out.answer_gt.value == expected
    assert execution["answer"] == expected
    assert execution["boundary_pair"] == boundary_pair
    assert execution["boundary_target_role"] == boundary_target_role
    expected_segments = {
        ("AB", "left_center_to_right_boundary"): ("A", "P"),
        ("AB", "left_boundary_to_right_center"): ("Q", "B"),
        ("BC", "left_center_to_right_boundary"): ("B", "R"),
        ("BC", "left_boundary_to_right_center"): ("S", "C"),
    }
    start_key, end_key = expected_segments[(boundary_pair, boundary_target_role)]
    render_map = out.trace_payload["render_map"]
    points = {**render_map["centers"], **render_map["boundary_points"]}
    assert out.annotation_gt.value == [points[start_key], points[end_key]]


def test_circle_centerline_overlap_two_circle_boundary_segment_formula() -> None:
    out = _generate(
        20260704,
        query_id=QUERY_ID_BOUNDARY_SEGMENT,
        overlap_case=(6, 14, 3),
        circle_count=2,
        boundary_pair="AB",
        boundary_target_role="left_boundary_to_right_center",
    )
    execution = out.trace_payload["execution_trace"]
    expected = _segment_length(CircleOverlapCase(6, 14, 0, 3, 0), "AB", "left_boundary_to_right_center")

    assert out.answer_gt.value == expected
    assert execution["circle_count"] == 2
    assert execution["boundary_pair"] == "AB"
    assert out.annotation_gt.value == [
        out.trace_payload["render_map"]["boundary_points"]["Q"],
        out.trace_payload["render_map"]["centers"]["B"],
    ]
    assert set(out.trace_payload["render_map"]["boundary_points"]) == {"P", "Q"}


def test_circle_centerline_overlap_prompt_describes_radius_labels() -> None:
    radius = _generate(
        20260705,
        query_id=QUERY_ID_BOUNDARY_SEGMENT,
        label_mode="radius",
    )

    assert "radius labels" in radius.prompt
    assert "radius or diameter" not in radius.prompt


def test_circle_centerline_overlap_case_bank_constraints_and_support() -> None:
    generated_cases = generated_overlap_cases()
    center_answers = {center_distance_length(case) for case in generated_cases}
    boundary_answers = {
        _segment_length(case, pair, role)
        for case in generated_cases
        for pair in BOUNDARY_PAIRS
        for role in BOUNDARY_TARGET_ROLES
        if not (case.circle_count == 2 and pair == "BC")
    }
    assert len(center_answers) >= 35
    assert len(boundary_answers) >= 18
    assert min(center_answers) > 0
    assert min(boundary_answers) >= 3
    for case in generated_cases:
        assert abs(case.radius_a - case.radius_b) + 1 < case.distance_ab < case.radius_a + case.radius_b
        if case.circle_count == 3:
            assert abs(case.radius_b - case.radius_c) + 1 < case.distance_bc < case.radius_b + case.radius_c
            assert case.distance_ac > case.radius_a + case.radius_c + 1


def test_circle_centerline_overlap_default_sampling_uses_broad_answer_support() -> None:
    answers_by_query = {QUERY_ID_CENTER_DISTANCE: [], QUERY_ID_BOUNDARY_SEGMENT: []}
    sampled_circle_counts: list[int] = []
    for index in range(100):
        circle_count, _circle_probs = select_circle_count(
            instance_seed=1234500 + index,
            params={},
            namespace=f"{TASK_ID}.{QUERY_ID_CENTER_DISTANCE}.circle_count",
        )
        sampled_circle_counts.append(circle_count)
        case, _case_probs = select_center_distance_overlap_case(
            circle_count=circle_count,
            instance_seed=1234500 + index,
            params={},
            namespace=f"{TASK_ID}.{QUERY_ID_CENTER_DISTANCE}.overlap_case",
        )
        answers_by_query[QUERY_ID_CENTER_DISTANCE].append(center_distance_length(case))

        boundary_circle_count, _circle_probs = select_circle_count(
            instance_seed=1234500 + index,
            params={},
            namespace=f"{TASK_ID}.{QUERY_ID_BOUNDARY_SEGMENT}.circle_count",
        )
        sampled_circle_counts.append(boundary_circle_count)
        boundary_case, _case_probs = select_overlap_case(
            circle_count=boundary_circle_count,
            instance_seed=1234500 + index,
            params={},
            namespace=f"{TASK_ID}.{QUERY_ID_BOUNDARY_SEGMENT}.overlap_case",
        )
        boundary_pair, _pair_probs = select_boundary_pair(
            circle_count=boundary_circle_count,
            params={},
            instance_seed=1234500 + index,
            namespace=f"{TASK_ID}.{QUERY_ID_BOUNDARY_SEGMENT}.boundary_pair",
        )
        boundary_target_role, _role_probs = select_boundary_target_role(
            params={},
            instance_seed=1234500 + index,
            namespace=f"{TASK_ID}.{QUERY_ID_BOUNDARY_SEGMENT}.boundary_target_role",
        )
        answers_by_query[QUERY_ID_BOUNDARY_SEGMENT].append(
            _segment_length(boundary_case, boundary_pair, boundary_target_role)
        )

    assert len(set(answers_by_query[QUERY_ID_CENTER_DISTANCE])) >= 25
    assert len(set(answers_by_query[QUERY_ID_BOUNDARY_SEGMENT])) >= 15
    assert 2 in sampled_circle_counts
    assert 3 in sampled_circle_counts


def test_circle_centerline_overlap_generation_is_deterministic() -> None:
    params = {
        "query_id": QUERY_ID_BOUNDARY_SEGMENT,
        "overlap_case": (8, 16, 7, 4, 3),
        "circle_count": 3,
        "label_mode": "radius",
        "boundary_pair": "BC",
        "boundary_target_role": "left_boundary_to_right_center",
    }
    first = _generate(314159, **params)
    second = _generate(314159, **params)

    assert first.prompt == second.prompt
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
    assert first.image.tobytes() == second.image.tobytes()


def test_circle_centerline_overlap_rejects_invalid_params() -> None:
    task = create_task(TASK_ID)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": "bad_query"}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"label_mode": "diameter"}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"circle_count": 4}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(
            1,
            params={"query_id": QUERY_ID_BOUNDARY_SEGMENT, "circle_count": 2, "boundary_pair": "BC"},
            max_attempts=1,
        )
    with pytest.raises(ValueError):
        task.generate(
            1,
            params={"query_id": QUERY_ID_BOUNDARY_SEGMENT, "boundary_target_role": "whole_diameter"},
            max_attempts=1,
        )
    with pytest.raises(ValueError):
        task.generate(1, params={"overlap_case": (4, 13, 9, 3, 2)}, max_attempts=1)


def _assert_segment_inside_image(annotation: list[list[float]], image_size: tuple[int, int]) -> None:
    width, height = image_size
    assert len(annotation) == 2
    for point in annotation:
        assert isinstance(point, list)
        assert len(point) == 2
        x, y = [float(value) for value in point]
        assert 0.0 <= x <= float(width)
        assert 0.0 <= y <= float(height)
