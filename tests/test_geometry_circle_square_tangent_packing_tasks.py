"""Contracts for circle/square tangent-packing geometry tasks."""

from __future__ import annotations

import math

import pytest

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.geometry.tangent_packing.circle_in_square_gap_area import (
    SCENE_ID,
    GeometryCircleInSquareGapAreaTask,
)
from trace_tasks.tasks.geometry.tangent_packing.circle_in_square_radius_from_gap_area import GeometryCircleInSquareRadiusFromGapAreaTask
from trace_tasks.tasks.geometry.tangent_packing.square_in_circle_gap_area import GeometrySquareInCircleGapAreaTask
from trace_tasks.tasks.geometry.tangent_packing.square_in_circle_side_from_gap_area import GeometrySquareInCircleSideFromGapAreaTask
from trace_tasks.tasks.geometry.tangent_packing.two_circles_in_rectangle_gap_area import GeometryTwoCirclesInRectangleGapAreaTask
from trace_tasks.tasks.geometry.tangent_packing.two_circles_in_rectangle_radius_from_gap_area import GeometryTwoCirclesInRectangleRadiusFromGapAreaTask

TASK_CLASSES = (
    GeometryCircleInSquareRadiusFromGapAreaTask,
    GeometrySquareInCircleSideFromGapAreaTask,
    GeometryTwoCirclesInRectangleRadiusFromGapAreaTask,
    GeometryCircleInSquareGapAreaTask,
    GeometrySquareInCircleGapAreaTask,
    GeometryTwoCirclesInRectangleGapAreaTask,
)

QUERY_IDS_BY_TASK = {
    GeometryCircleInSquareRadiusFromGapAreaTask: (SINGLE_QUERY_ID,),
    GeometrySquareInCircleSideFromGapAreaTask: (SINGLE_QUERY_ID,),
    GeometryTwoCirclesInRectangleRadiusFromGapAreaTask: (SINGLE_QUERY_ID,),
    GeometryCircleInSquareGapAreaTask: (SINGLE_QUERY_ID,),
    GeometrySquareInCircleGapAreaTask: (SINGLE_QUERY_ID,),
    GeometryTwoCirclesInRectangleGapAreaTask: (SINGLE_QUERY_ID,),
}

EXPECTED_FORMULA_BY_TASK = {
    GeometryCircleInSquareRadiusFromGapAreaTask: lambda radius: _round1(radius),
    GeometrySquareInCircleSideFromGapAreaTask: lambda radius: _round1(radius * math.sqrt(2.0)),
    GeometryTwoCirclesInRectangleRadiusFromGapAreaTask: lambda radius: _round1(radius),
    GeometryCircleInSquareGapAreaTask: lambda radius: _round1((2 * radius) ** 2 - math.pi * radius * radius),
    GeometrySquareInCircleGapAreaTask: lambda radius: _round1(math.pi * radius * radius - 2 * radius * radius),
    GeometryTwoCirclesInRectangleGapAreaTask: lambda radius: _round1(
        (4 * radius) * (2 * radius) - 2 * math.pi * radius * radius
    ),
}


def _round1(value: float) -> float:
    return round(float(value) + 1e-9, 1)


def _expected_answer_type(task_cls) -> str:
    return "number"


def _expected_answer_rounding(task_cls) -> str:
    return "one_decimal"


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_tangent_packing_tasks_emit_public_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(65001, params={}, max_attempts=20)

    assert out.scene_id == SCENE_ID
    assert out.query_id in QUERY_IDS_BY_TASK[task_cls]
    assert out.answer_gt.type == _expected_answer_type(task_cls)
    assert isinstance(out.answer_gt.value, float)
    assert out.annotation_gt.type == "bbox"
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert "target_cue" not in out.prompt_variants["answer_and_annotation"]
    assert "packing_region" not in out.prompt_variants["answer_and_annotation"]
    assert "support_measurement" not in out.prompt_variants["answer_and_annotation"]
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    assert trace["query_spec"]["scene_id"] == SCENE_ID
    assert trace["scene_ir"]["scene_id"] == SCENE_ID
    assert trace["witness_symbolic"]["scene_id"] == SCENE_ID
    assert trace["query_spec"]["query_id"] == out.query_id
    assert trace["execution_trace"]["query_id"] == out.query_id
    assert trace["execution_trace"]["answer_type"] == _expected_answer_type(task_cls)
    assert trace["execution_trace"]["answer_rounding"] == _expected_answer_rounding(task_cls)
    assert trace["execution_trace"]["answer_value"] == out.answer_gt.value
    assert trace["witness_symbolic"]["answer_value"] == out.answer_gt.value
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert trace["projected_annotation"]["pixel_bbox"] == out.annotation_gt.value
    assert trace["witness_symbolic"]["source_witness_type"] == "bbox"
    assert trace["scene_ir"]["relations"]["annotation_roles"] == ["diagram"]
    assert trace["execution_trace"]["annotation_roles"] == ["diagram"]
    assert trace["render_map"]["gap_shading_mode"] == "container_minus_packed_shape_mask"
    assert trace["render_map"]["packed_region_fill_mode"] == "background_unshaded"
    assert trace["render_map"]["gap_texture"] == "high_contrast_diagonal_hatch"

    radius = int(trace["execution_trace"]["radius"])
    square_side = int(trace["execution_trace"]["square_side"])
    container_width = int(trace["execution_trace"]["container_width"])
    container_height = int(trace["execution_trace"]["container_height"])

    assert square_side == 2 * radius
    if task_cls in {GeometryTwoCirclesInRectangleRadiusFromGapAreaTask, GeometryTwoCirclesInRectangleGapAreaTask}:
        assert container_width == 4 * radius
        assert container_height == 2 * radius

    expected_answer = EXPECTED_FORMULA_BY_TASK[task_cls](radius)
    assert out.answer_gt.value == pytest.approx(expected_answer)


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_tangent_packing_tasks_are_deterministic(task_cls) -> None:
    task = task_cls()
    params = {}
    out_a = task.generate(65011, params=params, max_attempts=20)
    out_b = task.generate(65011, params=params, max_attempts=20)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_tangent_packing_tasks_support_every_explicit_query(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            65021 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        assert out.query_id == query_id
        assert out.answer_gt.type == _expected_answer_type(task_cls)
        assert out.trace_payload["query_spec"]["params"][
            "query_id_probabilities"
        ] == {query_id: 1.0}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_tangent_packing_tasks_sample_all_query_ids(task_cls) -> None:
    task = task_cls()
    seen = set()
    for index in range(18):
        out = task.generate(
            65041 + index,
            params={"_sampling_index": index},
            max_attempts=20,
        )
        seen.add(out.query_id)

    assert seen == set(QUERY_IDS_BY_TASK[task_cls])


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_tangent_packing_annotation_stays_inside_canvas(task_cls) -> None:
    task = task_cls()
    for index, query_id in enumerate(QUERY_IDS_BY_TASK[task_cls]):
        out = task.generate(
            65061 + index,
            params={"query_id": query_id},
            max_attempts=20,
        )
        width, height = out.image.size
        x0, y0, x1, y1 = out.annotation_gt.value
        assert 0.0 <= x0 < x1 <= float(width)
        assert 0.0 <= y0 < y1 <= float(height)
        assert (x1 - x0) > 8.0
        assert (y1 - y0) > 8.0


def test_tangent_packing_tasks_reject_unknown_query_id() -> None:
    for task_cls in TASK_CLASSES:
        task = task_cls()
        with pytest.raises(ValueError):
            task.generate(65031, params={"query_id": "not_a_query"}, max_attempts=20)
