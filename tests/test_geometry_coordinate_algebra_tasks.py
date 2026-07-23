"""Contract tests for coordinate algebra geometry tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks import create_task
from trace_tasks.tasks.geometry.coordinate_plane.missing_endpoint_label import (
    MISSING_ENDPOINT_QUERY_IDS,
    MISSING_ENDPOINT_TASK_ID,
    REFLECTED_POINT_TASK_ID,
    REFLECTED_POINT_QUERY_IDS,
    ROTATED_POINT_TASK_ID,
    ROTATED_POINT_QUERY_IDS,
    SCENE_ID,
    SECTION_POINT_QUERY_IDS,
    SECTION_POINT_TASK_ID,
    TRANSLATED_POINT_TASK_ID,
    TRANSLATED_POINT_QUERY_IDS,
)


@pytest.mark.parametrize("query_id", MISSING_ENDPOINT_QUERY_IDS)
def test_missing_endpoint_task_has_unique_candidate_answer(query_id: str) -> None:
    task = create_task(MISSING_ENDPOINT_TASK_ID)
    out = task.generate(77901, params={"query_id": query_id, "winner_label": "C"}, max_attempts=50)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    candidates = execution["candidate_points_by_label"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == query_id
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "point"
    assert out.annotation_gt.value == candidates["C"]["point_px"]
    assert trace["projected_annotation"]["point"] == out.annotation_gt.value
    assert execution["target_point_graph"] == candidates["C"]["point_graph"]
    assert len(candidates) == 6
    assert sum(1 for payload in candidates.values() if payload["point_graph"] == execution["target_point_graph"]) == 1

    known = execution["known_points_by_label"]
    midpoint = known["M"]["point_graph"]
    endpoint_label = "P" if query_id == "missing_endpoint_from_midpoint" else "Q"
    endpoint = known[endpoint_label]["point_graph"]
    expected = [(2 * midpoint[0]) - endpoint[0], (2 * midpoint[1]) - endpoint[1]]
    assert execution["target_point_graph"] == expected


@pytest.mark.parametrize("query_id", SECTION_POINT_QUERY_IDS)
def test_section_point_task_has_unique_candidate_answer(query_id: str) -> None:
    task = create_task(SECTION_POINT_TASK_ID)
    out = task.generate(77906, params={"query_id": query_id, "winner_label": "D", "algebra_candidate_count": 6}, max_attempts=50)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    candidates = execution["candidate_points_by_label"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == query_id
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "D"
    assert out.annotation_gt.type == "point"
    assert out.annotation_gt.value == candidates["D"]["point_px"]
    assert trace["projected_annotation"]["point"] == out.annotation_gt.value
    assert execution["target_point_graph"] == candidates["D"]["point_graph"]
    assert len(candidates) == 4
    assert sum(1 for payload in candidates.values() if payload["point_graph"] == execution["target_point_graph"]) == 1

    known = execution["known_points_by_label"]
    point_p = known["P"]["point_graph"]
    point_q = known["Q"]["point_graph"]
    segment_dx = point_q[0] - point_p[0]
    segment_dy = point_q[1] - point_p[1]
    segment_length_squared = (segment_dx * segment_dx) + (segment_dy * segment_dy)
    for payload in candidates.values():
        point = payload["point_graph"]
        rel_x = point[0] - point_p[0]
        rel_y = point[1] - point_p[1]
        assert (rel_x * segment_dy) == (rel_y * segment_dx)
        projection = (rel_x * segment_dx) + (rel_y * segment_dy)
        assert 0 < projection < segment_length_squared
    step = 1 if query_id == "one_third_from_p_to_q" else 2
    expected = [
        point_p[0] + (step * (point_q[0] - point_p[0]) // 3),
        point_p[1] + (step * (point_q[1] - point_p[1]) // 3),
    ]
    assert execution["target_point_graph"] == expected


@pytest.mark.parametrize(
    ("task_id", "query_id"),
    (
        *((TRANSLATED_POINT_TASK_ID, query_id) for query_id in TRANSLATED_POINT_QUERY_IDS),
        *((REFLECTED_POINT_TASK_ID, query_id) for query_id in REFLECTED_POINT_QUERY_IDS),
        *((ROTATED_POINT_TASK_ID, query_id) for query_id in ROTATED_POINT_QUERY_IDS),
    ),
)
def test_transformed_point_task_has_unique_candidate_answer(task_id: str, query_id: str) -> None:
    task = create_task(task_id)
    out = task.generate(
        77911,
        params={"query_id": query_id, "winner_label": "E", "algebra_candidate_count": 6},
        max_attempts=50,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    candidates = execution["candidate_points_by_label"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == query_id
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "E"
    assert out.annotation_gt.type == "point"
    assert out.annotation_gt.value == candidates["E"]["point_px"]
    assert trace["projected_annotation"]["point"] == out.annotation_gt.value
    assert execution["target_point_graph"] == candidates["E"]["point_graph"]
    assert 4 <= len(candidates) <= 6
    assert "E" in candidates
    assert sum(1 for payload in candidates.values() if payload["point_graph"] == execution["target_point_graph"]) == 1

    source = execution["known_points_by_label"]["P"]["point_graph"]
    target = execution["target_point_graph"]
    if query_id == "translate_point":
        assert execution["transform_text"].startswith("translate P by")
        assert source != target
    elif query_id == "translate_by_reference_vector":
        point_r = execution["known_points_by_label"]["R"]["point_graph"]
        point_s = execution["known_points_by_label"]["S"]["point_graph"]
        expected = [source[0] + point_s[0] - point_r[0], source[1] + point_s[1] - point_r[1]]
        assert target == expected
        assert trace["render_spec"]["marker_style"]["guide_arrow_head_length_px"] >= 20
    elif query_id == "reflect_over_vertical_line":
        line_value = int(execution["transform_line"]["value"])
        assert execution["transform_line"]["axis"] == "x"
        assert target == [(2 * line_value) - source[0], source[1]]
        assert (
            trace["render_spec"]["marker_style"]["transform_axis_width_px"]
            > trace["render_spec"]["background_style"]["style_spec"]["axis_line_width"]
        )
    elif query_id == "reflect_over_horizontal_line":
        line_value = int(execution["transform_line"]["value"])
        assert execution["transform_line"]["axis"] == "y"
        assert target == [source[0], (2 * line_value) - source[1]]
        assert (
            trace["render_spec"]["marker_style"]["transform_axis_width_px"]
            > trace["render_spec"]["background_style"]["style_spec"]["axis_line_width"]
        )
    elif query_id == "single":
        center = execution["known_points_by_label"]["O"]["point_graph"]
        dx = source[0] - center[0]
        dy = source[1] - center[1]
        if "clockwise" in execution["formula"] and "counterclockwise" not in execution["formula"]:
            expected = [center[0] + dy, center[1] - dx]
        else:
            expected = [center[0] - dy, center[1] + dx]
        assert target == expected
        assert trace["render_spec"]["marker_style"]["guide_segment_color"] == [202, 45, 55]
        assert (
            trace["render_spec"]["marker_style"]["guide_segment_width_px"]
            > trace["render_spec"]["background_style"]["style_spec"]["axis_line_width"]
        )
    else:
        raise AssertionError(f"unhandled transform query {query_id}")


@pytest.mark.parametrize(
    ("task_id", "query_id"),
    (
        (MISSING_ENDPOINT_TASK_ID, "missing_endpoint_from_midpoint"),
        (SECTION_POINT_TASK_ID, "two_thirds_from_p_to_q"),
        (REFLECTED_POINT_TASK_ID, "reflect_over_vertical_line"),
        (ROTATED_POINT_TASK_ID, "single"),
        (TRANSLATED_POINT_TASK_ID, "translate_point"),
    ),
)
def test_coordinate_algebra_tasks_are_deterministic(task_id: str, query_id: str) -> None:
    task = create_task(task_id)
    params = {"query_id": query_id, "winner_label": "B"}
    out_a = task.generate(77921, params=dict(params), max_attempts=50)
    out_b = task.generate(77921, params=dict(params), max_attempts=50)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()
