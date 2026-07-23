"""Contract tests for coordinate-composite geometry tasks."""

from __future__ import annotations

import json

import pytest

from trace_tasks.tasks import TASK_REGISTRY, create_task
from trace_tasks.tasks.geometry.coordinate_composite.boundary_point_match_label import (
    QUERY_IDS as BOUNDARY_QUERY_IDS,
    TASK_ID as BOUNDARY_TASK_ID,
    GeometryCoordinateCompositeBoundaryPointMatchLabelTask,
)
from trace_tasks.tasks.geometry.coordinate_composite.intersection_point_count import (
    QUERY_IDS,
    SCENE_ID,
    TASK_ID,
    GeometryCoordinateCompositeIntersectionPointCountTask,
)
from trace_tasks.tasks.geometry.coordinate_composite.region_membership_label import (
    QUERY_IDS as REGION_QUERY_IDS,
    TASK_ID as REGION_TASK_ID,
    GeometryCoordinateCompositeRegionMembershipLabelTask,
)

TRANSFORMS = ("identity", "reflect_x", "reflect_y", "rotate90", "rotate180")


def _generate(seed: int, **params):
    task = create_task(TASK_ID)
    return task.generate(seed, params=dict(params), max_attempts=20)


def _generate_region(seed: int, **params):
    task = create_task(REGION_TASK_ID)
    return task.generate(seed, params=dict(params), max_attempts=20)


def _generate_boundary(seed: int, **params):
    task = create_task(BOUNDARY_TASK_ID)
    return task.generate(seed, params=dict(params), max_attempts=20)


def _assert_segment_not_on_axis(p0, p1) -> None:
    x0, y0 = [float(value) for value in p0]
    x1, y1 = [float(value) for value in p1]
    assert not (abs(y0) < 1e-6 and abs(y1) < 1e-6)
    assert not (abs(x0) < 1e-6 and abs(x1) < 1e-6)


def test_coordinate_composite_intersection_count_registered() -> None:
    assert TASK_ID in TASK_REGISTRY
    assert TASK_REGISTRY[TASK_ID] is GeometryCoordinateCompositeIntersectionPointCountTask


def test_coordinate_composite_region_membership_registered() -> None:
    assert REGION_TASK_ID in TASK_REGISTRY
    assert TASK_REGISTRY[REGION_TASK_ID] is GeometryCoordinateCompositeRegionMembershipLabelTask


def test_coordinate_composite_boundary_point_match_registered() -> None:
    assert BOUNDARY_TASK_ID in TASK_REGISTRY
    assert TASK_REGISTRY[BOUNDARY_TASK_ID] is GeometryCoordinateCompositeBoundaryPointMatchLabelTask


@pytest.mark.parametrize("query_id", QUERY_IDS)
def test_coordinate_composite_intersection_count_contract(query_id: str) -> None:
    out = _generate(20260620, query_id=query_id)
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == query_id
    assert trace["scene_id"] == SCENE_ID
    assert trace["query_id"] == query_id
    assert trace["query_spec"]["query_id"] == query_id
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == len(execution["intersection_points_graph"])
    assert out.annotation_gt.type == "point_set"
    assert len(out.annotation_gt.value) == out.answer_gt.value
    assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_point_set"] == out.annotation_gt.value
    assert trace["scene_ir"]["scene_id"] == SCENE_ID
    assert trace["render_spec"]["graph_paper_grid"]["spacing_px"] > 0
    assert trace["prompt"]["prompt_variant"]["prompt_bundle_id"] == "geometry_coordinate_composite_v0"
    assert "task_variant" not in json.dumps(trace)

    width, height = out.image.size
    for point in out.annotation_gt.value:
        assert len(point) == 2
        x_value, y_value = [float(value) for value in point]
        assert 0.0 <= x_value <= float(width)
        assert 0.0 <= y_value <= float(height)


def test_coordinate_composite_zero_intersection_uses_empty_annotation() -> None:
    out = _generate(
        20260621,
        query_id="line_circle_intersection_count",
        target_count=0,
    )
    assert out.answer_gt.value == 0
    assert out.annotation_gt.value == []
    assert out.trace_payload["projected_annotation"]["point_set"] == []


@pytest.mark.parametrize("query_id", QUERY_IDS)
@pytest.mark.parametrize("target_count", range(5))
@pytest.mark.parametrize("transform", TRANSFORMS)
def test_coordinate_composite_line_segments_do_not_overlap_axes(query_id: str, target_count: int, transform: str) -> None:
    out = _generate(
        20260623,
        query_id=query_id,
        target_count=target_count,
        transform=transform,
    )
    for obj in out.trace_payload["scene_ir"]["objects"]:
        if obj["kind"] == "line_segment":
            _assert_segment_not_on_axis(obj["p0"], obj["p1"])
        if obj["kind"] == "polygon":
            vertices = list(obj["vertices"])
            for index, p0 in enumerate(vertices):
                _assert_segment_not_on_axis(p0, vertices[(index + 1) % len(vertices)])


def test_coordinate_composite_is_deterministic() -> None:
    params = {"query_id": "circle_polygon_intersection_count", "target_count": 2, "transform": "rotate90"}
    first = _generate(20260622, **params)
    second = _generate(20260622, **params)

    assert first.prompt == second.prompt
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
    assert first.image.tobytes() == second.image.tobytes()


def test_coordinate_composite_rejects_invalid_params() -> None:
    task = create_task(TASK_ID)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": "bad_query"}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": "circle_circle_intersection_count", "target_count": 5}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": "line_circle_intersection_count", "transform": "skew"}, max_attempts=1)


@pytest.mark.parametrize("query_id", REGION_QUERY_IDS)
def test_coordinate_composite_region_membership_contract(query_id: str) -> None:
    out = _generate_region(20260701, query_id=query_id)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    point_by_label = {
        str(item["label"]): list(item["point"])
        for item in trace["render_map"]["candidate_points_px"]
    }

    assert out.scene_id == SCENE_ID
    assert out.query_id == query_id
    assert trace["scene_id"] == SCENE_ID
    assert trace["query_id"] == query_id
    assert trace["query_spec"]["query_id"] == query_id
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value in {"A", "B", "C", "D"}
    assert out.answer_gt.value == execution["answer_label"]
    assert out.annotation_gt.type == "point"
    assert out.annotation_gt.value == point_by_label[str(out.answer_gt.value)]
    assert trace["projected_annotation"]["type"] == "point"
    assert trace["projected_annotation"]["point"] == out.annotation_gt.value
    assert trace["prompt"]["prompt_variant"]["prompt_bundle_id"] == "geometry_coordinate_composite_v0"
    assert "task_variant" not in json.dumps(trace)

    width, height = out.image.size
    x_value, y_value = [float(value) for value in out.annotation_gt.value]
    assert 0.0 <= x_value <= float(width)
    assert 0.0 <= y_value <= float(height)
    assert len(trace["render_map"]["candidate_points_px"]) == 4
    assert len(trace["render_map"]["candidate_marker_bboxes"]) == 4


def test_coordinate_composite_region_membership_is_deterministic() -> None:
    params = {"query_id": "inside_polygon_outside_circle", "transform": "rotate180"}
    first = _generate_region(20260702, **params)
    second = _generate_region(20260702, **params)

    assert first.prompt == second.prompt
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
    assert first.image.tobytes() == second.image.tobytes()


def test_coordinate_composite_region_membership_rejects_invalid_params() -> None:
    task = create_task(REGION_TASK_ID)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": "bad_query"}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": "inside_circle_above_line", "transform": "rotate90"}, max_attempts=1)


@pytest.mark.parametrize("query_id", BOUNDARY_QUERY_IDS)
def test_coordinate_composite_boundary_point_match_contract(query_id: str) -> None:
    out = _generate_boundary(20260703, query_id=query_id)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    point_by_label = {
        str(item["label"]): list(item["point"])
        for item in trace["render_map"]["candidate_points_px"]
    }

    assert out.scene_id == SCENE_ID
    assert out.query_id == query_id
    assert trace["scene_id"] == SCENE_ID
    assert trace["query_id"] == query_id
    assert trace["query_spec"]["query_id"] == query_id
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value in {"A", "B", "C", "D"}
    assert out.answer_gt.value == execution["answer_label"]
    assert out.annotation_gt.type == "point"
    assert out.annotation_gt.value == point_by_label[str(out.answer_gt.value)]
    assert trace["projected_annotation"]["type"] == "point"
    assert trace["projected_annotation"]["point"] == out.annotation_gt.value
    assert trace["prompt"]["prompt_variant"]["prompt_bundle_id"] == "geometry_coordinate_composite_v0"
    assert "task_variant" not in json.dumps(trace)

    width, height = out.image.size
    x_value, y_value = [float(value) for value in out.annotation_gt.value]
    assert 0.0 <= x_value <= float(width)
    assert 0.0 <= y_value <= float(height)
    assert len(trace["render_map"]["candidate_points_px"]) == 4
    assert len(trace["render_map"]["candidate_marker_bboxes"]) == 4


def test_coordinate_composite_boundary_point_match_is_deterministic() -> None:
    params = {"query_id": "circle_polygon_boundary_point", "transform": "rotate180"}
    first = _generate_boundary(20260704, **params)
    second = _generate_boundary(20260704, **params)

    assert first.prompt == second.prompt
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
    assert first.image.tobytes() == second.image.tobytes()


def test_coordinate_composite_boundary_point_match_rejects_invalid_params() -> None:
    task = create_task(BOUNDARY_TASK_ID)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": "bad_query"}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": "line_circle_boundary_point", "transform": "rotate90"}, max_attempts=1)
