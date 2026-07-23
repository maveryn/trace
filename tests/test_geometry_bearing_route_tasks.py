"""Contracts for compass-bearing route geometry tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.bearing_route.endpoint_position_label import GeometryBearingRouteEndpointPositionLabelTask
from trace_tasks.tasks.geometry.bearing_route.final_bearing_value import GeometryBearingRouteFinalBearingValueTask
from trace_tasks.tasks.geometry.bearing_route.shared.state import SCENE_ID


TASK_CLASSES = (
    GeometryBearingRouteFinalBearingValueTask,
    GeometryBearingRouteEndpointPositionLabelTask,
)

QUERY_ID_BY_TASK = {
    GeometryBearingRouteFinalBearingValueTask: "single",
    GeometryBearingRouteEndpointPositionLabelTask: "single",
}

ANSWER_TYPE_BY_TASK = {
    GeometryBearingRouteFinalBearingValueTask: "option_letter",
    GeometryBearingRouteEndpointPositionLabelTask: "option_letter",
}

ANNOTATION_TYPE_BY_TASK = {
    GeometryBearingRouteFinalBearingValueTask: "point_map",
    GeometryBearingRouteEndpointPositionLabelTask: "point_map",
}

ANNOTATION_KEYS_BY_TASK = {
    GeometryBearingRouteFinalBearingValueTask: {
        "S",
        "F",
    },
}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_bearing_route_tasks_emit_public_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(78001, params={}, max_attempts=20)
    query_id = QUERY_ID_BY_TASK[task_cls]

    assert out.scene_id == SCENE_ID
    assert out.query_id == query_id
    assert out.answer_gt.type == ANSWER_TYPE_BY_TASK[task_cls]
    assert out.annotation_gt.type == ANNOTATION_TYPE_BY_TASK[task_cls]
    expected_annotation_keys = ANNOTATION_KEYS_BY_TASK.get(task_cls, {"S", str(out.answer_gt.value)})
    assert set(out.annotation_gt.value) == expected_annotation_keys
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    assert trace["query_spec"]["scene_id"] == SCENE_ID
    assert trace["scene_ir"]["scene_id"] == SCENE_ID
    assert trace["witness_symbolic"]["scene_id"] == SCENE_ID
    assert trace["query_spec"]["query_id"] == query_id
    assert trace["execution_trace"]["query_id"] == query_id
    assert trace["projected_annotation"]["type"] == ANNOTATION_TYPE_BY_TASK[task_cls]
    if out.annotation_gt.type == "bbox_map":
        assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_bbox_map"] == out.annotation_gt.value
    else:
        assert trace["projected_annotation"]["point_map"] == out.annotation_gt.value
        assert trace["projected_annotation"]["pixel_point_map"] == out.annotation_gt.value
    assert trace["render_spec"]["font_family"]["font_family"]


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_bearing_route_tasks_are_deterministic(task_cls) -> None:
    task = task_cls()
    out_a = task.generate(78011, params={}, max_attempts=20)
    out_b = task.generate(78011, params={}, max_attempts=20)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_bearing_route_final_bearing_uses_requested_bearing() -> None:
    task = GeometryBearingRouteFinalBearingValueTask()
    out = task.generate(78021, params={"target_bearing": 135}, max_attempts=20)
    trace = out.trace_payload["execution_trace"]

    assert out.answer_gt.value == trace["correct_option_label"]
    assert out.annotation_gt.type == "point_map"
    assert set(out.annotation_gt.value) == {"S", "F"}
    assert "compass_rose" not in out.annotation_gt.value
    assert int(trace["final_bearing"]) == 135
    assert int(trace["correct_option_value"]) == 135
    assert trace["answer_value"] == trace["correct_option_label"]
    assert trace["option_values"][trace["target_index"]] == 135
    assert len(trace["option_values"]) == 6
    assert len(set(trace["option_values"])) == 6
    assert trace["option_labels"] == ["A", "B", "C", "D", "E", "F"]
    assert out.answer_gt.value in trace["option_labels"]
    assert "bearing_options_bbox" in out.trace_payload["render_map"]
    assert int(trace["bearing_a"]) in {90, 180}
    assert int(trace["bearing_b"]) in {90, 180}


def test_bearing_endpoint_uses_selected_candidate_label() -> None:
    task = GeometryBearingRouteEndpointPositionLabelTask()
    out = task.generate(78031, params={"target_index": 2}, max_attempts=20)
    trace = out.trace_payload["execution_trace"]
    labels = trace["option_labels"]

    assert out.answer_gt.value == labels[2]
    assert trace["answer_value"] == labels[2]
    assert out.annotation_gt.type == "point_map"
    assert set(out.annotation_gt.value) == {"S", str(labels[2])}
    assert "route_instructions" not in out.annotation_gt.value
    assert "instruction_panel_bbox" in out.trace_payload["render_map"]
    assert "graph-paper candidate grid" in out.prompt
    graph_meta = out.trace_payload["render_map"]["candidate_graph_paper"]
    assert graph_meta["grid_unit"] == "one_square_equals_one_step"
    assert 20.0 <= float(graph_meta["grid_cell_px"]) <= 42.0
    assert 3 <= int(trace["leg_a"]) <= 7
    assert 3 <= int(trace["leg_b"]) <= 7
    assert abs(int(trace["leg_a"]) - int(trace["leg_b"])) >= 2
    selected_label_bbox = out.trace_payload["render_map"]["selected_candidate_label_bbox"]
    assert out.annotation_gt.value[str(labels[2])] != selected_label_bbox


def test_bearing_endpoint_candidate_labels_avoid_fixed_scene_labels() -> None:
    task = GeometryBearingRouteEndpointPositionLabelTask()
    reserved = {"N", "E", "S", "W", "F"}

    for seed in range(78050, 78100):
        out = task.generate(seed, params={}, max_attempts=20)
        labels = tuple(str(label) for label in out.trace_payload["execution_trace"]["option_labels"])

        assert len(labels) == len(set(labels))
        assert not reserved.intersection(labels)
        assert out.answer_gt.value in labels


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_bearing_route_annotation_stays_inside_canvas(task_cls) -> None:
    task = task_cls()
    out = task.generate(78041, params={}, max_attempts=20)
    width, height = out.image.size
    if out.annotation_gt.type == "bbox_map":
        for x0, y0, x1, y1 in out.annotation_gt.value.values():
            assert 0.0 <= x0 < x1 <= float(width)
            assert 0.0 <= y0 < y1 <= float(height)
            assert (x1 - x0) > 8.0
            assert (y1 - y0) > 8.0
    else:
        for x, y in out.annotation_gt.value.values():
            assert 0.0 <= x <= float(width)
            assert 0.0 <= y <= float(height)
