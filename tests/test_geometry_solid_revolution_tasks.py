"""Contracts for solid-revolution geometry tasks."""

from __future__ import annotations

import math

import pytest

from trace_tasks.tasks.geometry.solid_revolution.revolution_cone_volume_value import (
    ANNOTATION_KEYS as CONE_ANNOTATION_KEYS,
)
from trace_tasks.tasks.geometry.solid_revolution.revolution_cone_volume_value import (
    GeometrySolidRevolutionConeVolumeValueTask,
)
from trace_tasks.tasks.geometry.solid_revolution.revolution_cylinder_volume_value import (
    ANNOTATION_KEYS as CYLINDER_ANNOTATION_KEYS,
)
from trace_tasks.tasks.geometry.solid_revolution.revolution_cylinder_volume_value import (
    SCENE_ID,
    GeometrySolidRevolutionCylinderVolumeValueTask,
)
from trace_tasks.tasks.geometry.solid_revolution.revolution_cylinder_volume_from_diagonal_value import (
    ANNOTATION_KEYS as CYLINDER_DIAGONAL_ANNOTATION_KEYS,
)
from trace_tasks.tasks.geometry.solid_revolution.revolution_cylinder_volume_from_diagonal_value import (
    GeometrySolidRevolutionCylinderVolumeFromDiagonalValueTask,
)
from trace_tasks.tasks.geometry.solid_revolution.revolution_double_cone_volume_value import (
    ANNOTATION_KEYS as DOUBLE_CONE_ANNOTATION_KEYS,
)
from trace_tasks.tasks.geometry.solid_revolution.revolution_double_cone_volume_value import (
    GeometrySolidRevolutionDoubleConeVolumeValueTask,
)
from trace_tasks.tasks.geometry.solid_revolution.revolution_frustum_volume_value import (
    ANNOTATION_KEYS as FRUSTUM_ANNOTATION_KEYS,
)
from trace_tasks.tasks.geometry.solid_revolution.revolution_frustum_volume_value import (
    GeometrySolidRevolutionFrustumVolumeValueTask,
)

TASK_CLASSES = (
    GeometrySolidRevolutionCylinderVolumeValueTask,
    GeometrySolidRevolutionCylinderVolumeFromDiagonalValueTask,
    GeometrySolidRevolutionConeVolumeValueTask,
    GeometrySolidRevolutionDoubleConeVolumeValueTask,
    GeometrySolidRevolutionFrustumVolumeValueTask,
)

ANNOTATION_KEYS_BY_TASK = {
    GeometrySolidRevolutionCylinderVolumeValueTask: set(CYLINDER_ANNOTATION_KEYS),
    GeometrySolidRevolutionCylinderVolumeFromDiagonalValueTask: set(CYLINDER_DIAGONAL_ANNOTATION_KEYS),
    GeometrySolidRevolutionConeVolumeValueTask: set(CONE_ANNOTATION_KEYS),
    GeometrySolidRevolutionDoubleConeVolumeValueTask: set(DOUBLE_CONE_ANNOTATION_KEYS),
    GeometrySolidRevolutionFrustumVolumeValueTask: set(FRUSTUM_ANNOTATION_KEYS),
}


def _rounded(value: float) -> float:
    return round(float(value) + 1e-9, 1)


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_solid_revolution_tasks_emit_public_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(67001, params={}, max_attempts=20)

    assert out.scene_id == SCENE_ID
    assert out.query_id == "single"
    assert out.answer_gt.type == "number"
    assert out.annotation_gt.type == "bbox_map"
    assert set(out.annotation_gt.value) == ANNOTATION_KEYS_BY_TASK[task_cls]
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    assert trace["query_spec"]["scene_id"] == SCENE_ID
    assert trace["query_spec"]["query_id"] == "single"
    assert trace["query_spec"]["template_id"] == "geometry_solid_revolution_v1"
    assert trace["execution_trace"]["query_id"] == "single"
    assert trace["projected_annotation"]["type"] == "bbox_map"
    assert trace["projected_annotation"]["bbox_map"] == out.annotation_gt.value
    assert trace["execution_trace"]["answer_rounding"] == "one_decimal"
    assert trace["execution_trace"]["answer_support_size"] >= 50


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_solid_revolution_tasks_are_deterministic(task_cls) -> None:
    task = task_cls()
    out_a = task.generate(67011, params={}, max_attempts=20)
    out_b = task.generate(67011, params={}, max_attempts=20)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_solid_revolution_tasks_support_single_query(task_cls) -> None:
    task = task_cls()
    out = task.generate(67021, params={"query_id": "single"}, max_attempts=20)
    trace = out.trace_payload["execution_trace"]

    assert out.query_id == "single"
    assert out.trace_payload["query_spec"]["params"]["query_id_probabilities"] == {
        "single": 1.0
    }

    if task_cls is GeometrySolidRevolutionCylinderVolumeValueTask:
        diameter = float(trace["diameter"])
        height = float(trace["height"])
        assert trace["radial_input_kind"] == "diameter"
        assert trace["diagonal"] is None
        answer = math.pi * (diameter / 2.0) ** 2 * height
    elif task_cls is GeometrySolidRevolutionCylinderVolumeFromDiagonalValueTask:
        diameter = float(trace["diameter"])
        height = float(trace["height"])
        diagonal = float(trace["diagonal"])
        assert trace["radial_input_kind"] == "diagonal"
        assert diameter**2 + height**2 == pytest.approx(diagonal**2)
        answer = math.pi * (diameter / 2.0) ** 2 * height
    elif task_cls is GeometrySolidRevolutionConeVolumeValueTask:
        radius = float(trace["radius"])
        height = float(trace["height"])
        slant_height = float(trace["slant_height"])
        assert radius**2 + height**2 == pytest.approx(slant_height**2)
        answer = math.pi * radius**2 * height / 3.0
    elif task_cls is GeometrySolidRevolutionDoubleConeVolumeValueTask:
        radius = float(trace["radius"])
        half_height = float(trace["half_height"])
        assert trace["total_height"] == pytest.approx(2.0 * half_height)
        answer = 2.0 * math.pi * radius**2 * half_height / 3.0
    elif task_cls is GeometrySolidRevolutionFrustumVolumeValueTask:
        top_radius = float(trace["top_radius"])
        bottom_radius = float(trace["bottom_radius"])
        height = float(trace["height"])
        assert top_radius < bottom_radius
        answer = math.pi * height * (bottom_radius**2 + bottom_radius * top_radius + top_radius**2) / 3.0
    else:
        raise AssertionError(f"unhandled task class: {task_cls}")
    assert out.answer_gt.value == pytest.approx(_rounded(answer))


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_solid_revolution_annotation_stays_inside_canvas(task_cls) -> None:
    task = task_cls()
    out = task.generate(67041, params={"query_id": "single"}, max_attempts=20)
    width, height = out.image.size
    for x0, y0, x1, y1 in out.annotation_gt.value.values():
        assert 0.0 <= x0 < x1 <= float(width)
        assert 0.0 <= y0 < y1 <= float(height)
        assert (x1 - x0) > 8.0
        assert (y1 - y0) > 8.0


def test_solid_revolution_tasks_reject_unknown_query_id() -> None:
    task = GeometrySolidRevolutionCylinderVolumeValueTask()
    with pytest.raises(ValueError):
        task.generate(67031, params={"query_id": "not_a_query"}, max_attempts=20)
