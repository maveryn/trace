"""Contracts for trapezoid-completion geometry tasks."""

from __future__ import annotations

import pytest

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks.geometry.trapezoid_extension.extension_from_parallelogram_area import (
    GeometryTrapezoidExtensionFromParallelogramAreaTask,
)
from trace_tasks.tasks.geometry.trapezoid_extension.extension_from_parallelogram_perimeter import (
    GeometryTrapezoidExtensionFromParallelogramPerimeterTask,
)
from trace_tasks.tasks.geometry.trapezoid_extension.trapezoid_area_from_extension_and_height import (
    GeometryTrapezoidAreaFromExtensionAndHeightTask,
)
from trace_tasks.tasks.geometry.trapezoid_extension.trapezoid_area_from_parallelogram_area import (
    GeometryTrapezoidAreaFromParallelogramAreaTask,
)
from trace_tasks.tasks.geometry.trapezoid_extension.trapezoid_area_from_parallelogram_perimeter import (
    GeometryTrapezoidAreaFromParallelogramPerimeterTask,
)
from trace_tasks.tasks.geometry.trapezoid_extension.shared.state import SCENE_ID

TASK_CLASSES = (
    GeometryTrapezoidExtensionFromParallelogramAreaTask,
    GeometryTrapezoidExtensionFromParallelogramPerimeterTask,
    GeometryTrapezoidAreaFromExtensionAndHeightTask,
    GeometryTrapezoidAreaFromParallelogramAreaTask,
    GeometryTrapezoidAreaFromParallelogramPerimeterTask,
)

EXTENSION_TASKS = (
    GeometryTrapezoidExtensionFromParallelogramAreaTask,
    GeometryTrapezoidExtensionFromParallelogramPerimeterTask,
)


def _point_in_bbox(point: tuple[float, float], bbox: list[float]) -> bool:
    x, y = float(point[0]), float(point[1])
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return x0 <= x <= x1 and y0 <= y <= y1


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return (float(b[0]) - float(a[0])) * (float(c[1]) - float(a[1])) - (
        float(b[1]) - float(a[1])
    ) * (float(c[0]) - float(a[0]))


def _segments_intersect(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    o1 = _orientation(a, b, c)
    o2 = _orientation(a, b, d)
    o3 = _orientation(c, d, a)
    o4 = _orientation(c, d, b)
    return (o1 <= 0.0 <= o2 or o2 <= 0.0 <= o1) and (o3 <= 0.0 <= o4 or o4 <= 0.0 <= o3)


def _segment_intersects_bbox(segment: tuple[tuple[float, float], tuple[float, float]], bbox: list[float]) -> bool:
    start, end = segment
    if _point_in_bbox(start, bbox) or _point_in_bbox(end, bbox):
        return True
    x0, y0, x1, y1 = [float(value) for value in bbox]
    corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
    edges = tuple(zip(corners, (*corners[1:], corners[0])))
    return any(_segments_intersect(start, end, edge_start, edge_end) for edge_start, edge_end in edges)


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_trapezoid_extension_tasks_emit_public_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(64001, params={}, max_attempts=20)

    assert out.scene_id == SCENE_ID
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "number"
    expected_annotation_type = "segment" if task_cls in EXTENSION_TASKS else "bbox"
    assert out.annotation_gt.type == expected_annotation_type
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    assert trace["query_spec"]["scene_id"] == SCENE_ID
    assert trace["scene_ir"]["scene_id"] == SCENE_ID
    assert trace["witness_symbolic"]["scene_id"] == SCENE_ID
    assert trace["query_spec"]["query_id"] == SINGLE_QUERY_ID
    assert trace["execution_trace"]["query_id"] == SINGLE_QUERY_ID
    assert trace["projected_annotation"]["type"] == expected_annotation_type
    assert trace["witness_symbolic"]["source_witness_type"] == expected_annotation_type
    assert trace["query_spec"]["prompt_variant"]["prompt_schema_version"] == "v1"
    assert set(trace["render_map"]["vertex_points"]) == {"A", "B", "C", "D", "E"}
    assert {f"point_{label}" for label in "ABCDE"}.issubset(trace["render_map"]["label_bboxes"])

    top_base = int(trace["execution_trace"]["top_base"])
    extension = int(trace["execution_trace"]["extension"])
    bottom_base = int(trace["execution_trace"]["bottom_base"])
    height = int(trace["execution_trace"]["height"])
    side = int(trace["execution_trace"]["side"])
    parallelogram_area = int(trace["execution_trace"]["parallelogram_area"])
    parallelogram_perimeter = int(trace["execution_trace"]["parallelogram_perimeter"])
    trapezoid_area = float(trace["execution_trace"]["trapezoid_area"])

    assert bottom_base == top_base + extension
    assert parallelogram_area == bottom_base * height
    assert parallelogram_perimeter == 2 * (bottom_base + side)
    assert trapezoid_area == pytest.approx((top_base + bottom_base) * height / 2.0)
    if task_cls is GeometryTrapezoidExtensionFromParallelogramAreaTask:
        assert out.answer_gt.value == pytest.approx(parallelogram_area / height - top_base)
    elif task_cls is GeometryTrapezoidExtensionFromParallelogramPerimeterTask:
        assert out.answer_gt.value == pytest.approx(parallelogram_perimeter / 2.0 - side - top_base)
    elif task_cls is GeometryTrapezoidAreaFromParallelogramPerimeterTask:
        derived_bottom_base = parallelogram_perimeter / 2.0 - side
        assert out.answer_gt.value == pytest.approx(height * (top_base + derived_bottom_base) / 2.0)
    else:
        assert out.answer_gt.value == pytest.approx(trapezoid_area)


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_trapezoid_extension_tasks_are_deterministic(task_cls) -> None:
    task = task_cls()
    params = {}
    out_a = task.generate(64011, params=params, max_attempts=20)
    out_b = task.generate(64011, params=params, max_attempts=20)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_trapezoid_extension_tasks_support_explicit_single_query(task_cls) -> None:
    task = task_cls()
    for param_key in ("query_id", "query_variant"):
        out = task.generate(
            64021,
            params={param_key: SINGLE_QUERY_ID},
            max_attempts=20,
        )
        assert out.query_id == SINGLE_QUERY_ID
        assert out.answer_gt.type == "number"
        assert out.trace_payload["query_spec"]["params"]["query_id_probabilities"] == {SINGLE_QUERY_ID: 1.0}


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_trapezoid_extension_annotation_stays_inside_canvas(task_cls) -> None:
    task = task_cls()
    out = task.generate(64061, params={"query_id": SINGLE_QUERY_ID}, max_attempts=20)
    width, height = out.image.size
    if out.annotation_gt.type == "segment":
        for x, y in out.annotation_gt.value:
            assert 0.0 <= float(x) <= float(width)
            assert 0.0 <= float(y) <= float(height)
        (x0, y0), (x1, y1) = out.annotation_gt.value
        assert abs(float(x1) - float(x0)) > 8.0 or abs(float(y1) - float(y0)) > 8.0
        return
    assert out.annotation_gt.type == "bbox"
    x0, y0, x1, y1 = out.annotation_gt.value
    for x0, y0, x1, y1 in (out.annotation_gt.value,):
        assert 0.0 <= x0 < x1 <= float(width)
        assert 0.0 <= y0 < y1 <= float(height)
        assert (x1 - x0) > 8.0
        assert (y1 - y0) > 8.0


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_trapezoid_extension_labels_stay_inside_canvas(task_cls) -> None:
    task = task_cls()
    for seed in range(64070, 64090):
        out = task.generate(seed, params={"query_id": SINGLE_QUERY_ID}, max_attempts=20)
        width, height = out.image.size
        for role, bbox in out.trace_payload["render_map"]["label_bboxes"].items():
            x0, y0, x1, y1 = [float(value) for value in bbox]
            assert x0 > 2.0, (seed, role, bbox)
            assert y0 > 2.0, (seed, role, bbox)
            assert x1 < float(width) - 2.0, (seed, role, bbox)
            assert y1 < float(height) - 2.0, (seed, role, bbox)
        side_bbox = out.trace_payload["render_map"]["label_bboxes"].get("side")
        if side_bbox is not None:
            trapezoid_points = out.trace_payload["render_map"]["original_trapezoid"]["points"]
            ad_segment = (tuple(trapezoid_points[0]), tuple(trapezoid_points[3]))
            assert not _segment_intersects_bbox(ad_segment, side_bbox), (seed, side_bbox, ad_segment)


def test_trapezoid_extension_tasks_reject_unknown_query_id() -> None:
    for task_cls in TASK_CLASSES:
        task = task_cls()
        with pytest.raises(ValueError):
            task.generate(64031, params={"query_id": "not_a_query"}, max_attempts=20)
