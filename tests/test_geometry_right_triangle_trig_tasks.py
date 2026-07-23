"""Contracts for right-triangle trigonometry geometry tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.triangle_relations.right_triangle_missing_side_value import (
    GeometryRightTriangleMissingSideValueTask,
    SCENE_ID,
)


TASK_CLASSES = (
    GeometryRightTriangleMissingSideValueTask,
)

QUERY_IDS_BY_TASK = {
    GeometryRightTriangleMissingSideValueTask: ("single",),
}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_right_triangle_trig_tasks_emit_public_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(57001, params={}, max_attempts=20)

    assert out.scene_id == SCENE_ID
    assert out.query_id == "single"
    assert out.answer_gt.type == "number"
    assert out.annotation_gt.type == "segment"
    assert len(out.annotation_gt.value) == 2
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    assert trace["query_spec"]["scene_id"] == SCENE_ID
    assert trace["query_spec"]["query_id"] == out.query_id
    assert trace["execution_trace"]["query_id"] == out.query_id
    assert trace["projected_annotation"]["type"] == out.annotation_gt.type
    assert trace["execution_trace"]["answer_rounding"] == "one_decimal"


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_right_triangle_trig_tasks_are_deterministic(task_cls) -> None:
    task = task_cls()
    params = {}
    out_a = task.generate(57011, params=params, max_attempts=20)
    out_b = task.generate(57011, params=params, max_attempts=20)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_right_triangle_trig_tasks_support_every_explicit_query(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            57021 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        assert out.query_id == query_id
        assert out.answer_gt.type == "number"
        assert out.trace_payload["query_spec"]["params"]["query_id_probabilities"] == {
            query_id: 1.0
        }


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_right_triangle_trig_annotation_stays_inside_canvas(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            57041 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        width, height = out.image.size
        for x, y in out.annotation_gt.value:
            assert 0.0 <= x <= float(width)
            assert 0.0 <= y <= float(height)


def test_right_triangle_trig_tasks_reject_unknown_query_id() -> None:
    task = GeometryRightTriangleMissingSideValueTask()
    with pytest.raises(ValueError):
        task.generate(57031, params={"query_id": "not_a_query"}, max_attempts=20)
