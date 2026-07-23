"""Contracts for solid cross-section geometry tasks."""

from __future__ import annotations

import math

import pytest

from trace_tasks.tasks.geometry.solid_cross_section.cone_parallel_slice_area import (
    SCENE_ID,
    GeometryConeParallelSliceAreaTask,
)
from trace_tasks.tasks.geometry.solid_cross_section.square_pyramid_parallel_slice_area import GeometrySquarePyramidParallelSliceAreaTask

TASK_CLASSES = (GeometryConeParallelSliceAreaTask, GeometrySquarePyramidParallelSliceAreaTask)

QUERY_IDS_BY_TASK = {
    GeometryConeParallelSliceAreaTask: ("single",),
    GeometrySquarePyramidParallelSliceAreaTask: ("single",),
}

@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_solid_cross_section_tasks_emit_public_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(65001, params={}, max_attempts=20)

    assert out.scene_id == SCENE_ID
    assert out.query_id == "single"
    assert out.answer_gt.type == "number"
    assert out.annotation_gt.type == "bbox"
    assert len(out.annotation_gt.value) == 4
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    assert trace["query_spec"]["scene_id"] == SCENE_ID
    assert trace["query_spec"]["query_id"] == out.query_id
    assert trace["execution_trace"]["query_id"] == out.query_id
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert trace["execution_trace"]["answer_rounding"] == "one_decimal"
    assert trace["execution_trace"]["answer_support_size"] >= 50


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_solid_cross_section_tasks_are_deterministic(task_cls) -> None:
    task = task_cls()
    params = {}
    out_a = task.generate(65011, params=params, max_attempts=20)
    out_b = task.generate(65011, params=params, max_attempts=20)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert (
        out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    )
    assert out_a.image.tobytes() == out_b.image.tobytes()


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_solid_cross_section_tasks_support_every_explicit_query(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            65021 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        trace = out.trace_payload["execution_trace"]
        assert out.query_id == query_id
        assert out.answer_gt.type == "number"
        assert out.trace_payload["query_spec"]["params"][
            "query_id_probabilities"
        ] == {query_id: 1.0}

        scale = float(trace["slice_distance_from_apex"]) / float(trace["solid_height"])
        assert trace["similarity_scale"] == pytest.approx(scale)
        if task_cls is GeometryConeParallelSliceAreaTask:
            base_radius = float(trace["base_radius"])
            slice_radius = base_radius * scale
            assert trace["slice_radius"] == pytest.approx(slice_radius)
            assert out.answer_gt.value == pytest.approx(round(math.pi * slice_radius**2, 1))
        elif task_cls is GeometrySquarePyramidParallelSliceAreaTask:
            base_side = float(trace["base_side"])
            exact_slice_side = base_side * scale
            assert trace["slice_side"] == pytest.approx(exact_slice_side)
            assert out.answer_gt.value == pytest.approx(round(exact_slice_side**2 + 1e-9, 1))


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_solid_cross_section_annotation_stays_inside_canvas(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            65041 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        width, height = out.image.size
        x0, y0, x1, y1 = out.annotation_gt.value
        assert 0.0 <= x0 < x1 <= float(width)
        assert 0.0 <= y0 < y1 <= float(height)
        assert (x1 - x0) > 8.0
        assert (y1 - y0) > 8.0


def test_solid_cross_section_tasks_reject_unknown_query_id() -> None:
    task = GeometryConeParallelSliceAreaTask()
    with pytest.raises(ValueError):
        task.generate(65031, params={"query_id": "not_a_query"}, max_attempts=20)
