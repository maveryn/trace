"""Contracts for incircle tangent-segment geometry tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.incircle_tangents.incircle_radius_from_area_value import GeometryIncircleRadiusFromAreaValueTask
from trace_tasks.tasks.geometry.incircle_tangents.incircle_tangent_perimeter_value import (
    SCENE_ID,
    GeometryIncircleTangentPerimeterValueTask,
)

TASK_CLASSES = (
    GeometryIncircleTangentPerimeterValueTask,
    GeometryIncircleRadiusFromAreaValueTask,
)

QUERY_IDS_BY_TASK = {
    GeometryIncircleTangentPerimeterValueTask: ("single",),
    GeometryIncircleRadiusFromAreaValueTask: ("single",),
}

INTERNAL_QUERY_BY_TASK = {
    GeometryIncircleTangentPerimeterValueTask: "triangle_perimeter_from_tangent_segments",
    GeometryIncircleRadiusFromAreaValueTask: "inradius_from_tangent_segments",
}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_tangent_polygon_incircle_tasks_emit_public_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(57001, params={}, max_attempts=20)

    assert out.scene_id == SCENE_ID
    assert out.query_id == "single"
    if task_cls is GeometryIncircleTangentPerimeterValueTask:
        assert out.answer_gt.type == "integer"
        assert set(out.annotation_gt.value) == {"A", "B", "C", "D", "E", "F"}
    else:
        assert out.answer_gt.type == "number"
        assert set(out.annotation_gt.value) == {"A", "B", "C", "D", "E", "F", "O"}
    assert out.annotation_gt.type == "point_map"
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    assert trace["query_spec"]["scene_id"] == SCENE_ID
    assert trace["query_spec"]["query_id"] == out.query_id
    assert trace["execution_trace"]["query_id"] == out.query_id
    assert trace["execution_trace"]["internal_query_id"] == INTERNAL_QUERY_BY_TASK[task_cls]
    assert trace["projected_annotation"]["type"] == "point_map"
    assert trace["projected_annotation"]["point_map"] == out.annotation_gt.value
    assert trace["execution_trace"]["semiperimeter"] > 0
    assert trace["execution_trace"]["area"] > 0
    if task_cls is GeometryIncircleTangentPerimeterValueTask:
        assert out.answer_gt.value == int(round(2.0 * trace["execution_trace"]["semiperimeter"]))
    else:
        assert "area" not in out.annotation_gt.value
        assert trace["execution_trace"]["answer_rounding"] == "one_decimal"
        assert out.answer_gt.value == pytest.approx(
            round(trace["execution_trace"]["inradius"], 1)
        )
    triangle = trace["scene_ir"]["entities"][0]
    assert set(triangle["tangent_points"]) == {"D", "E", "F"}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_tangent_polygon_incircle_tasks_are_deterministic(task_cls) -> None:
    task = task_cls()
    params = {}
    out_a = task.generate(57011, params=params, max_attempts=20)
    out_b = task.generate(57011, params=params, max_attempts=20)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert out_a.image.tobytes() == out_b.image.tobytes()


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_tangent_polygon_incircle_tasks_support_every_explicit_query(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            57021 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        assert out.query_id == query_id
        assert out.answer_gt.type in {"integer", "number"}
        assert out.trace_payload["query_spec"]["params"][
            "query_id_probabilities"
        ] == {query_id: 1.0}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_tangent_polygon_incircle_annotation_stays_inside_canvas(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            57041 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        width, height = out.image.size
        for x, y in out.annotation_gt.value.values():
            assert 0.0 <= x <= float(width)
            assert 0.0 <= y <= float(height)


def test_tangent_polygon_incircle_tasks_reject_unknown_query_id() -> None:
    task = GeometryIncircleTangentPerimeterValueTask()
    with pytest.raises(ValueError):
        task.generate(57031, params={"query_id": "not_a_query"}, max_attempts=20)
