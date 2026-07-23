"""Contracts for cuboid orthographic-view geometry tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.cuboid_views.cuboid_projection_surface_area_value import (
    QUERY_ID,
    SCENE_ID,
    GeometryCuboidProjectionSurfaceAreaValueTask,
)

TASK_CLASSES = (GeometryCuboidProjectionSurfaceAreaValueTask,)

QUERY_IDS_BY_TASK = {
    GeometryCuboidProjectionSurfaceAreaValueTask: (QUERY_ID,),
}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_cuboid_orthographic_tasks_emit_public_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(61001, params={}, max_attempts=20)

    assert out.scene_id == SCENE_ID
    assert out.query_id
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_map"
    assert set(out.annotation_gt.value) == {"top_view", "front_view", "right_view"}
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    assert trace["query_spec"]["scene_id"] == SCENE_ID
    assert trace["query_spec"]["query_id"] == out.query_id
    assert trace["execution_trace"]["query_id"] == out.query_id
    assert trace["projected_annotation"]["type"] == "bbox_map"
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value

    length = int(trace["execution_trace"]["length"])
    width = int(trace["execution_trace"]["width"])
    height = int(trace["execution_trace"]["height"])
    assert length > 0
    assert width > 0
    assert height > 0
    assert trace["execution_trace"]["volume"] == length * width * height
    assert trace["execution_trace"]["surface_area"] == 2 * (
        (length * width) + (length * height) + (width * height)
    )
    assert trace["execution_trace"]["top_view_perimeter"] == 2 * (length + width)
    assert trace["execution_trace"]["front_view_perimeter"] == 2 * (length + height)
    assert trace["execution_trace"]["right_view_perimeter"] == 2 * (width + height)
    assert trace["execution_trace"]["formula_schema"] == "surface_area_from_orthographic_views"
    assert out.answer_gt.value == 2 * ((length * width) + (length * height) + (width * height))


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_cuboid_orthographic_tasks_are_deterministic(task_cls) -> None:
    task = task_cls()
    params = {}
    out_a = task.generate(61011, params=params, max_attempts=20)
    out_b = task.generate(61011, params=params, max_attempts=20)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert out_a.image.tobytes() == out_b.image.tobytes()


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_cuboid_orthographic_tasks_support_every_explicit_query(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            61021 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert out.trace_payload["query_spec"]["params"][
            "query_id_probabilities"
        ] == {query_id: 1.0}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_cuboid_orthographic_annotation_stays_inside_canvas(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            61041 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        width, height = out.image.size
        for x0, y0, x1, y1 in out.annotation_gt.value.values():
            assert 0.0 <= x0 < x1 <= float(width)
            assert 0.0 <= y0 < y1 <= float(height)
            assert (x1 - x0) > 8.0
            assert (y1 - y0) > 8.0


def test_cuboid_orthographic_tasks_reject_unknown_query_id() -> None:
    task = GeometryCuboidProjectionSurfaceAreaValueTask()
    with pytest.raises(ValueError):
        task.generate(61031, params={"query_id": "not_a_query"}, max_attempts=20)
