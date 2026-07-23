"""Regression tests for circle-pair tangent geometry tasks."""

from __future__ import annotations

import json
import math

import pytest

from trace_tasks.core.taxonomy import lookup_task_taxonomy
from trace_tasks.tasks import TASK_REGISTRY, create_task
from trace_tasks.tasks.geometry.circle_pair_tangents.external_tangent_segment_length_value import (
    ANNOTATION_KEYS,
    GeometryCirclePairTangentsExternalTangentSegmentLengthValueTask,
    QUERY_ID_CENTER_DISTANCE_FROM_TANGENT_SEGMENT_LENGTH,
    QUERY_ID_TANGENT_SEGMENT_LENGTH_FROM_CENTER_DISTANCE,
    SCENE_ID,
    TASK_ID,
    TASK_ID_EXTERNAL_TANGENT_SEGMENT_LENGTH,
)
from trace_tasks.tasks.geometry.circle_pair_tangents.shared.construction import (
    TANGENT_CASES,
    validate_tangent_case,
)


def _generate(seed: int, *, task_id: str = TASK_ID, **params):
    task = create_task(task_id)
    return task.generate(seed, params=dict(params), max_attempts=80)


def test_circle_pair_tangent_length_registered() -> None:
    assert TASK_ID_EXTERNAL_TANGENT_SEGMENT_LENGTH in TASK_REGISTRY
    assert TASK_REGISTRY[TASK_ID_EXTERNAL_TANGENT_SEGMENT_LENGTH] is GeometryCirclePairTangentsExternalTangentSegmentLengthValueTask


def test_circle_pair_tangent_default_pool_has_broad_unique_answer_support() -> None:
    assert len(TANGENT_CASES) >= 64

    tangent_lengths = [int(case.tangent_length) for case in TANGENT_CASES]
    center_distances = [int(case.center_distance) for case in TANGENT_CASES]

    assert len(set(tangent_lengths)) == len(tangent_lengths)
    assert len(set(center_distances)) == len(center_distances)
    for case in TANGENT_CASES:
        validate_tangent_case(case)


@pytest.mark.parametrize(
    ("query_id", "expected_answer", "unknown_role"),
    [
        (
            QUERY_ID_TANGENT_SEGMENT_LENGTH_FROM_CENTER_DISTANCE,
            12,
            "tangent_length",
        ),
        (
            QUERY_ID_CENTER_DISTANCE_FROM_TANGENT_SEGMENT_LENGTH,
            13,
            "center_distance",
        ),
    ],
)
@pytest.mark.parametrize("larger_side", ["left", "right"])
@pytest.mark.parametrize("tangent_side", ["above", "below"])
def test_circle_pair_tangent_formula_and_annotation(
    query_id: str,
    expected_answer: int,
    unknown_role: str,
    larger_side: str,
    tangent_side: str,
) -> None:
    out = _generate(
        20260623,
        query_id=query_id,
        tangent_case=(3, 8, 13, 12),
        larger_circle_side=larger_side,
        tangent_side=tangent_side,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == SCENE_ID
    assert out.query_id == query_id
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == expected_answer == execution["answer"]
    assert execution["center_distance"] == 13
    assert execution["radius_difference"] == 5
    assert execution["tangent_length"] ** 2 == execution["center_distance"] ** 2 - execution["radius_difference"] ** 2
    assert execution["formula_family"] == "external_common_tangent_right_triangle"
    assert execution["larger_circle_side"] == larger_side
    assert execution["tangent_side"] == tangent_side
    assert execution["unknown_role"] == unknown_role
    assert trace["witness_symbolic"]["formula_family"] == "external_common_tangent_right_triangle"
    assert trace["witness_symbolic"]["unknown_role"] == unknown_role

    assert out.annotation_gt.type == "point_map"
    annotation = out.annotation_gt.value
    assert tuple(annotation.keys()) == ANNOTATION_KEYS
    assert trace["projected_annotation"]["point_map"] == annotation
    assert trace["projected_annotation"]["pixel_point_map"] == annotation
    assert trace["render_spec"]["prompt"]["prompt_variant"]["prompt_bundle_id"] == "geometry_circle_pair_tangents_v1"
    assert trace["render_spec"]["style"]["font_bold"] is False
    assert trace["render_spec"]["style"]["label_stroke_width"] == 0
    assert "task_variant" not in json.dumps(trace)
    _assert_point_map_inside_image(annotation, out.image.size)
    _assert_rendered_tangent_geometry(trace["render_map"])


def test_circle_pair_tangent_length_generation_is_deterministic() -> None:
    params = {
        "query_id": QUERY_ID_TANGENT_SEGMENT_LENGTH_FROM_CENTER_DISTANCE,
        "tangent_case": (4, 12, 17, 15),
        "larger_circle_side": "right",
        "tangent_side": "above",
    }
    first = _generate(314159, **params)
    second = _generate(314159, **params)

    assert first.prompt == second.prompt
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
    assert first.image.tobytes() == second.image.tobytes()


def test_circle_pair_center_distance_generation_is_deterministic() -> None:
    params = {
        "query_id": QUERY_ID_CENTER_DISTANCE_FROM_TANGENT_SEGMENT_LENGTH,
        "tangent_case": (4, 12, 17, 15),
        "larger_circle_side": "right",
        "tangent_side": "above",
    }
    first = _generate(314160, **params)
    second = _generate(314160, **params)

    assert first.prompt == second.prompt
    assert first.answer_gt.value == 17
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
    assert first.image.tobytes() == second.image.tobytes()


def test_circle_pair_tangent_length_rejects_invalid_params() -> None:
    task = create_task(TASK_ID)
    with pytest.raises(ValueError):
        task.generate(1, params={"query_id": "bad_query"}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"larger_circle_side": "middle"}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"tangent_side": "inside"}, max_attempts=1)
    with pytest.raises(ValueError):
        task.generate(1, params={"tangent_case": (3, 8, 12, 12)}, max_attempts=1)


def test_circle_pair_retired_task_ids_are_removed() -> None:
    retired_ids = (
        "task_geometry__circle_pair_tangents__center_distance_value",
        "task_geometry__circle_pair_tangents__common_tangent_length_value",
    )
    for task_id in retired_ids:
        assert lookup_task_taxonomy(task_id) is None


def _assert_point_map_inside_image(annotation: dict[str, list[float]], image_size: tuple[int, int]) -> None:
    width, height = image_size
    for key in ANNOTATION_KEYS:
        point = annotation[key]
        assert isinstance(point, list)
        assert len(point) == 2
        x, y = [float(value) for value in point]
        assert 0.0 <= x <= float(width)
        assert 0.0 <= y <= float(height)


def _assert_rendered_tangent_geometry(render_map: dict[str, object]) -> None:
    centers = render_map["centers"]
    tangent_points = render_map["tangent_points"]
    circle_bboxes = render_map["circle_bboxes"]
    label_bboxes = render_map["label_bboxes"]
    auxiliary = render_map["auxiliary_right_triangle"]

    c = _point(centers["C"])
    d = _point(centers["D"])
    a = _point(tangent_points["A"])
    b = _point(tangent_points["B"])
    e = _point(auxiliary["E"])
    tangent_vector = _sub(b, a)
    radius_c = _bbox_radius(circle_bboxes["C"])
    radius_d = _bbox_radius(circle_bboxes["D"])

    assert abs(_distance(c, a) - radius_c) <= 1.5
    assert abs(_distance(d, b) - radius_d) <= 1.5
    assert abs(_cosine(tangent_vector, _sub(a, c))) <= 1e-3
    assert abs(_cosine(tangent_vector, _sub(b, d))) <= 1e-3
    assert abs(_distance(e, d) - float(auxiliary["ED_length_units"]) * float(render_map["scale_px_per_unit"])) <= 2.0
    assert abs(_cosine(_sub(e, c), tangent_vector)) >= 0.999
    assert _bbox_outside_circle(label_bboxes["radius_o1"], center=c, radius=radius_c)
    assert _bbox_outside_circle(label_bboxes["radius_o2"], center=d, radius=radius_d)
    for radius_label in ("radius_o1", "radius_o2"):
        for point_label in ("A_label", "B_label", "C_label", "D_label"):
            assert not _bboxes_overlap(label_bboxes[radius_label], label_bboxes[point_label], pad=2.0)


def _point(value: list[float]) -> tuple[float, float]:
    return float(value[0]), float(value[1])


def _sub(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return a[0] - b[0], a[1] - b[1]


def _distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _dot(a: tuple[float, float], b: tuple[float, float]) -> float:
    return a[0] * b[0] + a[1] * b[1]


def _norm(value: tuple[float, float]) -> float:
    return math.hypot(value[0], value[1])


def _cosine(a: tuple[float, float], b: tuple[float, float]) -> float:
    return _dot(a, b) / (_norm(a) * _norm(b))


def _bbox_radius(bbox: list[float]) -> float:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return ((x1 - x0) + (y1 - y0)) / 4.0


def _bboxes_overlap(a: list[float], b: list[float], *, pad: float = 0.0) -> bool:
    ax0, ay0, ax1, ay1 = [float(value) for value in a]
    bx0, by0, bx1, by1 = [float(value) for value in b]
    return not (
        ax1 + float(pad) < bx0
        or bx1 + float(pad) < ax0
        or ay1 + float(pad) < by0
        or by1 + float(pad) < ay0
    )


def _bbox_outside_circle(bbox: list[float], *, center: tuple[float, float], radius: float) -> bool:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    closest_x = max(x0, min(float(center[0]), x1))
    closest_y = max(y0, min(float(center[1]), y1))
    return _distance((closest_x, closest_y), center) >= float(radius) - 2.0
