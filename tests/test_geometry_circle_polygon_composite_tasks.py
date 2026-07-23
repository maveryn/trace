"""Regression tests for circle-polygon composite geometry tasks."""

from __future__ import annotations

import json
import math

import pytest

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.geometry.circle_polygon_composite.tangent_angle_value import (
    QUERY_ID_TANGENT_ANGLE,
    TASK_ID as TANGENT_ANGLE_TASK_ID,
    GeometryCirclePolygonCompositeTangentAngleValueTask,
)
from trace_tasks.tasks.geometry.circle_polygon_composite.tangential_quadrilateral_side_length_value import (
    TASK_ID,
    GeometryCirclePolygonCompositeTangentialQuadrilateralSideLengthValueTask,
)


def _generate(seed: int, **params):
    task = GeometryCirclePolygonCompositeTangentialQuadrilateralSideLengthValueTask()
    return task.generate(seed, params=dict(params), max_attempts=80)


def _generate_angle(seed: int, **params):
    task = GeometryCirclePolygonCompositeTangentAngleValueTask()
    return task.generate(seed, params=dict(params), max_attempts=80)


def test_circle_polygon_composite_registered_public_task() -> None:
    assert TASK_ID in TASK_REGISTRY
    assert TASK_REGISTRY[TASK_ID] is GeometryCirclePolygonCompositeTangentialQuadrilateralSideLengthValueTask
    assert TANGENT_ANGLE_TASK_ID in TASK_REGISTRY
    assert TASK_REGISTRY[TANGENT_ANGLE_TASK_ID] is GeometryCirclePolygonCompositeTangentAngleValueTask


@pytest.mark.parametrize(
    ("missing_side", "expected"),
    [
        ("AB", 7),
        ("BC", 9),
        ("CD", 11),
        ("DA", 9),
    ],
)
def test_tangential_quadrilateral_side_length_formula(missing_side: str, expected: int) -> None:
    out = _generate(
        20260604,
        missing_side=missing_side,
        tangent_lengths=(3, 4, 5, 6),
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == "circle_polygon_composite"
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == expected == execution["answer"]
    side_lengths = execution["side_lengths"]
    assert side_lengths["AB"] + side_lengths["CD"] == side_lengths["BC"] + side_lengths["DA"]
    assert execution["answer"] == side_lengths[missing_side]
    assert execution["missing_side"] == missing_side

    assert out.annotation_gt.type == "point_map"
    annotation = out.annotation_gt.value
    assert set(annotation) == {
        "A",
        "B",
        "C",
        "D",
    }
    _assert_point_map_inside_image(annotation, out.image.size)
    assert "task_variant" not in json.dumps(trace)


def test_tangential_quadrilateral_generation_is_deterministic() -> None:
    params = {
        "missing_side": "AB",
        "tangent_lengths": (5, 6, 7, 8),
    }
    first = _generate(314159, **params)
    second = _generate(314159, **params)

    assert first.prompt == second.prompt
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]


def test_tangential_quadrilateral_rejects_invalid_params() -> None:
    task = GeometryCirclePolygonCompositeTangentialQuadrilateralSideLengthValueTask()
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": "bad_query"}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"missing_side": "AC"}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"tangent_lengths": (3, 4, 5)}, max_attempts=1)


@pytest.mark.parametrize("construction_kind", ["incircle", "semicircle"])
@pytest.mark.parametrize("side_sign", [-1, 1])
def test_circle_polygon_tangent_angle_contract(construction_kind: str, side_sign: int) -> None:
    out = _generate_angle(
        20260606,
        query_id=SINGLE_QUERY_ID,
        construction_kind=construction_kind,
        target_angle=45,
        side_sign=side_sign,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == "circle_polygon_composite"
    assert out.query_id == SINGLE_QUERY_ID
    assert execution["internal_query_id"] == QUERY_ID_TANGENT_ANGLE
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 45 == execution["answer"]
    assert execution["known_angle_degrees"] == execution["target_angle_degrees"] == 45
    assert execution["side_sign"] == side_sign
    assert execution["construction_kind"] == construction_kind
    assert trace["query_spec"]["params"]["construction_kind"] == construction_kind

    assert out.annotation_gt.type == "point_map"
    annotation = out.annotation_gt.value
    assert set(annotation) == {"A", "B", "C", "D", "O", "T"}
    render_map = trace["render_map"]
    assert annotation["O"] == render_map["circle_center"]
    assert annotation["T"] == render_map["tangent_point"]
    assert _angle_at(
        render_map["known_angle_vertex"],
        render_map["known_angle_reference_point"],
        render_map["tangent_point"],
    ) == pytest.approx(45.0, abs=0.5)
    assert _angle_at(
        render_map["target_angle_vertex"],
        render_map["target_reference_point"],
        render_map["tangent_point"],
    ) == pytest.approx(45.0, abs=0.5)
    _assert_point_map_inside_image(annotation, out.image.size)
    assert "task_variant" not in json.dumps(trace)


def test_circle_polygon_tangent_angle_generation_is_deterministic() -> None:
    params = {
        "query_id": SINGLE_QUERY_ID,
        "construction_kind": "semicircle",
        "target_angle": 60,
        "side_sign": 1,
    }
    first = _generate_angle(271828, **params)
    second = _generate_angle(271828, **params)

    assert first.prompt == second.prompt
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]


def test_circle_polygon_tangent_angle_rejects_invalid_params() -> None:
    task = GeometryCirclePolygonCompositeTangentAngleValueTask()
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": "bad_query"}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"construction_kind": "bad_construction"}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"target_angle": 17}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"side_sign": 0}, max_attempts=1)


def _assert_point_map_inside_image(annotation: dict[str, list[float]], image_size: tuple[int, int]) -> None:
    width, height = image_size
    for point in annotation.values():
        assert isinstance(point, list)
        assert len(point) == 2
        x, y = [float(value) for value in point]
        assert 0.0 <= x <= float(width)
        assert 0.0 <= y <= float(height)


def _angle_at(vertex: list[float], arm_a: list[float], arm_b: list[float]) -> float:
    ax = float(arm_a[0]) - float(vertex[0])
    ay = float(arm_a[1]) - float(vertex[1])
    bx = float(arm_b[0]) - float(vertex[0])
    by = float(arm_b[1]) - float(vertex[1])
    denom = math.hypot(ax, ay) * math.hypot(bx, by)
    assert denom > 0.0
    cosine = max(-1.0, min(1.0, ((ax * bx) + (ay * by)) / denom))
    return math.degrees(math.acos(cosine))
