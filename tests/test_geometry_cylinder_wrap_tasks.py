"""Contracts for cylinder-wrap geometry measurement tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.cylinder_wrap.surface_path_length_value import (
    GeometryCylinderWrapSurfacePathLengthValueTask,
    SCENE_ID,
)
from trace_tasks.tasks.geometry.cylinder_wrap.wrapped_mark_position_label import GeometryCylinderWrapWrappedMarkPositionLabelTask


TASK_CLASSES = (
    GeometryCylinderWrapSurfacePathLengthValueTask,
    GeometryCylinderWrapWrappedMarkPositionLabelTask,
)

QUERY_ID_BY_TASK = {
    GeometryCylinderWrapSurfacePathLengthValueTask: "single",
    GeometryCylinderWrapWrappedMarkPositionLabelTask: "single",
}

ANSWER_TYPE_BY_TASK = {
    GeometryCylinderWrapSurfacePathLengthValueTask: "integer",
    GeometryCylinderWrapWrappedMarkPositionLabelTask: "option_letter",
}

ANNOTATION_COUNT_BY_TASK = {
    GeometryCylinderWrapSurfacePathLengthValueTask: 3,
    GeometryCylinderWrapWrappedMarkPositionLabelTask: 2,
}

ANNOTATION_TYPE_BY_TASK = {
    GeometryCylinderWrapSurfacePathLengthValueTask: "bbox_map",
    GeometryCylinderWrapWrappedMarkPositionLabelTask: "point_map",
}

ANNOTATION_KEYS_BY_TASK = {
    GeometryCylinderWrapSurfacePathLengthValueTask: {
        "marked_surface_path",
        "circumference_dimension",
        "height_dimension",
    },
    GeometryCylinderWrapWrappedMarkPositionLabelTask: {
        "source_strip_mark",
        "matching_rim_candidate",
    },
}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_cylinder_wrap_tasks_emit_public_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(76001, params={}, max_attempts=20)
    query_id = QUERY_ID_BY_TASK[task_cls]

    assert out.scene_id == SCENE_ID
    assert out.query_id == query_id
    assert out.answer_gt.type == ANSWER_TYPE_BY_TASK[task_cls]
    assert out.annotation_gt.type == ANNOTATION_TYPE_BY_TASK[task_cls]
    assert len(out.annotation_gt.value) == ANNOTATION_COUNT_BY_TASK[task_cls]
    assert set(out.annotation_gt.value) == ANNOTATION_KEYS_BY_TASK[task_cls]
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    assert trace["query_spec"]["scene_id"] == SCENE_ID
    assert trace["scene_ir"]["scene_id"] == SCENE_ID
    assert trace["witness_symbolic"]["scene_id"] == SCENE_ID
    assert trace["query_spec"]["query_id"] == query_id
    assert trace["execution_trace"]["query_id"] == query_id
    assert trace["projected_annotation"]["type"] == ANNOTATION_TYPE_BY_TASK[task_cls]
    assert trace["render_spec"]["font_family"]["font_family"]


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_cylinder_wrap_tasks_are_deterministic(task_cls) -> None:
    task = task_cls()
    out_a = task.generate(76011, params={}, max_attempts=20)
    out_b = task.generate(76011, params={}, max_attempts=20)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_cylinder_surface_path_uses_pythagorean_case() -> None:
    task = GeometryCylinderWrapSurfacePathLengthValueTask()
    out = task.generate(76021, params={"target_path_length": 25}, max_attempts=20)
    trace = out.trace_payload["execution_trace"]

    assert out.answer_gt.value == 25
    assert int(trace["path_length"]) == 25
    assert int(trace["circumference"]) ** 2 + int(trace["height"]) ** 2 == 25**2


def test_wrapped_mark_position_uses_labeled_candidate_answer() -> None:
    task = GeometryCylinderWrapWrappedMarkPositionLabelTask()
    out = task.generate(76031, params={"target_index": 2}, max_attempts=20)
    trace = out.trace_payload["execution_trace"]
    labels = trace["option_labels"]

    assert out.answer_gt.value == labels[2]
    assert trace["answer_value"] == labels[2]


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_cylinder_wrap_annotation_stays_inside_canvas(task_cls) -> None:
    task = task_cls()
    out = task.generate(76041, params={}, max_attempts=20)
    width, height = out.image.size
    if out.annotation_gt.type == "bbox_map":
        for x0, y0, x1, y1 in out.annotation_gt.value.values():
            assert 0.0 <= x0 < x1 <= float(width)
            assert 0.0 <= y0 < y1 <= float(height)
            assert (x1 - x0) > 8.0
            assert (y1 - y0) > 8.0
    else:
        assert out.annotation_gt.type == "point_map"
        for x, y in out.annotation_gt.value.values():
            assert 0.0 <= x <= float(width)
            assert 0.0 <= y <= float(height)
