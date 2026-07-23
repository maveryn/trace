"""Contracts for sector formula geometry tasks."""

from __future__ import annotations

import math

import pytest

from trace_tasks.tasks.geometry.sector.arc_length_from_sector_area_value import (
    GeometrySectorArcLengthFromSectorAreaValueTask,
)
from trace_tasks.tasks.geometry.sector.arc_length_from_supplement_angle_value import (
    GeometrySectorArcLengthFromSupplementAngleValueTask,
)
from trace_tasks.tasks.geometry.sector.central_angle_from_sector_measure_value import (
    GeometrySectorCentralAngleFromSectorMeasureValueTask,
)
from trace_tasks.tasks.geometry.sector.sector_area_from_complement_angle_value import (
    GeometrySectorAreaFromComplementAngleValueTask,
)
from trace_tasks.tasks.geometry.sector.related_angle_from_sector_measure_value import (
    GeometrySectorRelatedAngleFromSectorMeasureValueTask,
)


SCENE_ID = "sector"
SINGLE_QUERY_ID = "single"

TASK_CLASSES = (
    GeometrySectorAreaFromComplementAngleValueTask,
    GeometrySectorArcLengthFromSectorAreaValueTask,
    GeometrySectorArcLengthFromSupplementAngleValueTask,
    GeometrySectorCentralAngleFromSectorMeasureValueTask,
    GeometrySectorRelatedAngleFromSectorMeasureValueTask,
)
TASK_QUERY_CASES = tuple((task_cls, query_id) for task_cls in TASK_CLASSES for query_id in task_cls.supported_query_ids)

EXPECTED_ANNOTATION_ROLE_BY_TASK = {
    GeometrySectorAreaFromComplementAngleValueTask: "target_sector_region",
    GeometrySectorArcLengthFromSectorAreaValueTask: "target_arc",
    GeometrySectorArcLengthFromSupplementAngleValueTask: "target_arc",
    GeometrySectorCentralAngleFromSectorMeasureValueTask: "target_sector_angle_region",
    GeometrySectorRelatedAngleFromSectorMeasureValueTask: "target_related_angle_arc",
}

RETIRED_TASK_IDS = {
    "task_geometry__sector__arc_length_value",
    "task_geometry__sector__related_angle_value",
    "task_geometry__sector__sector_angle_value",
    "task_geometry__sector__sector_area_value",
    "task_geometry__sector__angle_from_sector_measure_angle_from_arc_length_and_radius",
    "task_geometry__sector__angle_from_sector_measure_angle_from_area_and_radius",
    "task_geometry__sector__arc_length_value_arc_length_from_area_and_radius",
    "task_geometry__sector__arc_length_value_arc_length_from_radius_and_supplement_angle",
    "task_geometry__sector__central_angle_from_arc_length_value",
    "task_geometry__sector__central_angle_from_sector_area_value",
    "task_geometry__sector__complement_angle_from_arc_length_value",
    "task_geometry__sector__remaining_angle_from_arc_length_value",
    "task_geometry__sector__related_angle_from_sector_measure_complement_angle_from_arc_length",
    "task_geometry__sector__related_angle_from_sector_measure_remaining_angle_from_sector_measure",
    "task_geometry__sector__related_angle_from_sector_measure_supplement_angle_from_area",
    "task_geometry__sector__sector_area_from_arc_length_value",
    "task_geometry__sector__sector_area_value_area_from_arc_length_and_radius",
    "task_geometry__sector__sector_area_value_area_from_radius_and_complement_angle",
    "task_geometry__sector__supplement_angle_from_sector_area_value",
}


def _round1(value: float) -> float:
    return round(float(value) + 1e-9, 1)


def _expected_answer(task_cls: type, trace: dict) -> float:
    radius = float(trace["radius_units"])
    theta = float(trace["theta_degrees"])
    arc_length = float(trace["arc_length"])
    sector_area = float(trace["sector_area"])
    angle_from_arc = _round1((360.0 * arc_length) / (2.0 * math.pi * radius))
    angle_from_area = _round1((360.0 * sector_area) / (math.pi * radius**2))
    query_id = str(trace["query_id"])

    if task_cls is GeometrySectorAreaFromComplementAngleValueTask:
        return _round1((theta / 360.0) * math.pi * radius**2)
    if task_cls is GeometrySectorArcLengthFromSectorAreaValueTask:
        return _round1((2.0 * sector_area) / radius)
    if task_cls is GeometrySectorArcLengthFromSupplementAngleValueTask:
        return _round1((theta / 360.0) * 2.0 * math.pi * radius)
    if task_cls is GeometrySectorCentralAngleFromSectorMeasureValueTask:
        if query_id == "from_arc_length":
            return angle_from_arc
        if query_id == "from_sector_area":
            return angle_from_area
    if task_cls is GeometrySectorRelatedAngleFromSectorMeasureValueTask:
        if query_id == "complement_from_arc_length":
            return _round1(90.0 - angle_from_arc)
        if query_id == "supplement_from_sector_area":
            return _round1(180.0 - angle_from_area)
        if query_id == "remaining_from_arc_length":
            return _round1(360.0 - angle_from_arc)
    raise AssertionError(f"unhandled task class: {task_cls}")


@pytest.mark.parametrize(("task_cls", "query_id"), TASK_QUERY_CASES)
def test_sector_formula_tasks_emit_public_contract(task_cls, query_id) -> None:
    task = task_cls()
    out = task.generate(55001, params={"query_id": query_id}, max_attempts=20)

    assert out.scene_id == SCENE_ID
    assert out.query_id == query_id
    assert out.answer_gt.type == "number"
    assert out.annotation_gt.type == "bbox"
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert '"answer"' in out.prompt_variants["answer_only"]

    trace = out.trace_payload
    annotation_roles = trace["execution_trace"]["annotation_roles"]
    assert annotation_roles == [EXPECTED_ANNOTATION_ROLE_BY_TASK[task_cls]]
    assert all("label" not in role and "readout" not in role for role in annotation_roles)
    assert trace["query_spec"]["params"]["scene_id"] == SCENE_ID
    assert trace["query_spec"]["query_id"] == query_id
    assert trace["execution_trace"]["query_id"] == query_id
    assert trace["projected_annotation"]["type"] == "bbox"
    assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert trace["witness_symbolic"]["source_witness_type"] == "bbox"
    target_bboxes = trace["render_map"]["target_bboxes"]
    assert target_bboxes[annotation_roles[0]] == out.annotation_gt.value
    construction_points = trace["render_map"]["construction_points"]
    assert {"O", "A", "B"}.issubset(set(construction_points))
    assert all(len(construction_points[label]) == 2 for label in ("O", "A", "B"))
    if task_cls is GeometrySectorRelatedAngleFromSectorMeasureValueTask and query_id != "remaining_from_arc_length":
        assert "C" in construction_points
    assert "construction_label_bboxes" in trace["render_map"]


@pytest.mark.parametrize("task_cls", TASK_CLASSES)
def test_sector_formula_tasks_are_deterministic(task_cls) -> None:
    task = task_cls()
    params = {}
    out_a = task.generate(55011, params=params, max_attempts=20)
    out_b = task.generate(55011, params=params, max_attempts=20)

    assert out_a.prompt == out_b.prompt
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()


@pytest.mark.parametrize(("task_cls", "query_id"), TASK_QUERY_CASES)
def test_sector_formula_tasks_support_query_ids(task_cls, query_id) -> None:
    task = task_cls()
    out = task.generate(
        55021,
        params={"query_id": query_id},
        max_attempts=20,
    )
    assert out.query_id == query_id
    assert out.answer_gt.type == "number"
    assert out.annotation_gt.type == "bbox"
    probabilities = out.trace_payload["query_spec"]["params"]["query_id_probabilities"]
    assert probabilities == {str(branch): (1.0 if str(branch) == str(query_id) else 0.0) for branch in task.supported_query_ids}


@pytest.mark.parametrize(("task_cls", "query_id"), TASK_QUERY_CASES)
def test_sector_formula_annotation_stays_inside_canvas(task_cls, query_id) -> None:
    task = task_cls()
    out = task.generate(
        55041,
        params={"query_id": query_id},
        max_attempts=20,
    )
    width, height = out.image.size
    x0, y0, x1, y1 = out.annotation_gt.value
    assert 0.0 <= x0 < x1 <= float(width)
    assert 0.0 <= y0 < y1 <= float(height)
    assert (x1 - x0) > 18.0
    assert (y1 - y0) > 18.0


@pytest.mark.parametrize(("task_cls", "query_id"), TASK_QUERY_CASES)
def test_sector_formula_answers_match_trace_formula(task_cls, query_id) -> None:
    task = task_cls()
    out = task.generate(55061, params={"query_id": query_id}, max_attempts=20)
    expected = _expected_answer(task_cls, out.trace_payload["execution_trace"])
    assert float(out.answer_gt.value) == pytest.approx(expected)


@pytest.mark.parametrize(("task_cls", "query_id"), TASK_QUERY_CASES)
def test_sector_formula_prompts_do_not_repeat_sampled_readouts(task_cls, query_id) -> None:
    task = task_cls()
    out = task.generate(55071, params={"query_id": query_id}, max_attempts=20)
    question_text = out.prompt.split("Return a JSON object", 1)[0]
    trace = out.trace_payload["execution_trace"]
    forbidden = {
        f"{float(trace['radius_units']):.1f}",
        f"{float(trace['arc_length']):.1f}",
        f"{float(trace['sector_area']):.1f}",
    }
    assert not any(value in question_text for value in forbidden)
    assert "visible measurement readouts" not in out.prompt
    assert "circular sector diagram" not in out.prompt.split(".", 1)[0]
    assert "Use the shown measurements" not in out.prompt
    assert "Use the visible measurements" not in out.prompt
    assert "Use the measurements shown" not in out.prompt
    assert "Use the labeled values" not in out.prompt
    assert "Use the visible diagram values" not in out.prompt
    if task_cls in {
        GeometrySectorAreaFromComplementAngleValueTask,
        GeometrySectorArcLengthFromSectorAreaValueTask,
        GeometrySectorArcLengthFromSupplementAngleValueTask,
        GeometrySectorCentralAngleFromSectorMeasureValueTask,
    }:
        assert ("AOB" in question_text) or ("arc AB" in question_text) or ("from A to B" in question_text)
    if task_cls is GeometrySectorRelatedAngleFromSectorMeasureValueTask:
        assert " at O" in question_text


def test_sector_formula_tasks_reject_unknown_query_id() -> None:
    task = GeometrySectorAreaFromComplementAngleValueTask()
    with pytest.raises(ValueError):
        task.generate(55031, params={"query_id": "not_a_query"}, max_attempts=20)


def test_sector_formula_retired_public_ids_are_not_registered() -> None:
    from trace_tasks.tasks.registry import TASK_REGISTRY, ensure_scene_tasks_registered

    ensure_scene_tasks_registered("geometry", SCENE_ID)
    for task_id in RETIRED_TASK_IDS:
        assert task_id not in TASK_REGISTRY
