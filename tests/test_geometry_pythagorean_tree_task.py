"""Regression tests for the Pythagorean tree geometry task."""

from __future__ import annotations

import json

import pytest

from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.geometry.pythagorean_tree.missing_square_area_value import (
    SUPPORTED_QUERY_IDS,
    TASK_ID,
    GeometryPythagoreanTreeMissingSquareAreaValueTask,
    _TRIPLES,
)


def _generate(seed: int, **params):
    task = GeometryPythagoreanTreeMissingSquareAreaValueTask()
    return task.generate(seed, params=dict(params), max_attempts=80)


def test_pythagorean_tree_registered_public_task() -> None:
    assert TASK_ID in TASK_REGISTRY
    assert TASK_REGISTRY[TASK_ID] is GeometryPythagoreanTreeMissingSquareAreaValueTask


def test_pythagorean_tree_supports_expected_public_queries() -> None:
    task = GeometryPythagoreanTreeMissingSquareAreaValueTask()

    assert tuple(task.supported_query_ids) == SUPPORTED_QUERY_IDS
    assert SUPPORTED_QUERY_IDS == ("hypotenuse_square_area", "leg_square_area")


def test_pythagorean_tree_default_pool_has_distinct_square_area_support() -> None:
    assert len(_TRIPLES) >= 10

    legs = [value for leg_a, leg_b, _hypotenuse in _TRIPLES for value in (leg_a, leg_b)]
    hypotenuses = [hypotenuse for _leg_a, _leg_b, hypotenuse in _TRIPLES]

    assert len(set(legs)) == len(legs)
    assert len(set(hypotenuses)) == len(hypotenuses)
    for leg_a, leg_b, hypotenuse in _TRIPLES:
        assert int(leg_a) ** 2 + int(leg_b) ** 2 == int(hypotenuse) ** 2


def test_pythagorean_tree_hypotenuse_square_area_formula() -> None:
    out = _generate(20260604, query_id="hypotenuse_square_area", triple=(3, 4, 5))
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == "pythagorean_tree"
    assert out.query_id == "hypotenuse_square_area"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == 25 == execution["answer"]
    assert execution["leg_square_1_area"] + execution["leg_square_2_area"] == execution["hypotenuse_square_area"]
    assert execution["target_role"] == "hypotenuse_square"
    assert out.trace_payload["render_spec"]["prompt"]["prompt_bundle_id"] == "geometry_pythagorean_tree_v1"

    assert out.annotation_gt.type == "bbox"
    _assert_bbox_inside_image(out.annotation_gt.value, out.image.size)
    _assert_annotation_matches_target_square(out)
    assert trace["projected_annotation"]["type"] == "bbox"
    assert execution["annotation_roles"] == ["target_square"]
    assert "task_variant" not in json.dumps(trace)


@pytest.mark.parametrize(
    ("target_role", "expected"),
    [
        ("leg_square_1", 9),
        ("leg_square_2", 16),
    ],
)
def test_pythagorean_tree_leg_square_area_formula(target_role: str, expected: int) -> None:
    out = _generate(20260605, query_id="leg_square_area", target_role=target_role, triple=(3, 4, 5))
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == "pythagorean_tree"
    assert out.query_id == "leg_square_area"
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == expected == execution["answer"]
    assert execution["target_role"] == target_role
    assert execution["hypotenuse_square_area"] - expected in {
        execution["leg_square_1_area"],
        execution["leg_square_2_area"],
    }
    assert out.trace_payload["render_spec"]["prompt"]["prompt_bundle_id"] == "geometry_pythagorean_tree_v1"

    assert out.annotation_gt.type == "bbox"
    _assert_bbox_inside_image(out.annotation_gt.value, out.image.size)
    _assert_annotation_matches_target_square(out)
    assert trace["projected_annotation"]["type"] == "bbox"
    assert execution["annotation_roles"] == ["target_square"]
    assert "task_variant" not in json.dumps(trace)


def test_pythagorean_tree_generation_is_deterministic() -> None:
    params = {"query_id": "leg_square_area", "target_role": "leg_square_2", "triple": (5, 12, 13)}
    first = _generate(314159, **params)
    second = _generate(314159, **params)

    assert first.prompt == second.prompt
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]


@pytest.mark.parametrize("query_id", SUPPORTED_QUERY_IDS)
def test_pythagorean_tree_explicit_queries_generate(query_id: str) -> None:
    out = _generate(20260606, query_id=query_id, triple=(5, 12, 13))

    assert out.query_id == query_id
    assert out.scene_id == "pythagorean_tree"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox"
    _assert_bbox_inside_image(out.annotation_gt.value, out.image.size)
    _assert_annotation_matches_target_square(out)


def _assert_bbox_inside_image(bbox: list[float], image_size: tuple[int, int]) -> None:
    width, height = image_size
    assert isinstance(bbox, list)
    assert len(bbox) == 4
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert 0.0 <= x0 < x1 <= float(width)
    assert 0.0 <= y0 < y1 <= float(height)


def _assert_annotation_matches_target_square(out) -> None:
    target_role = out.trace_payload["execution_trace"]["target_role"]
    expected = out.trace_payload["render_map"]["square_bboxes"][target_role]
    actual = out.annotation_gt.value
    assert [round(float(value), 3) for value in actual] == [
        round(float(value), 3) for value in expected
    ]
