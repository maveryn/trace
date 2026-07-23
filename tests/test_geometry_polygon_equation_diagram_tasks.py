"""Regression tests for polygon equation diagram geometry tasks."""

from __future__ import annotations

import json
import re

import pytest

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.tasks import create_task


TASK_IDS = (
    "task_geometry__polygon_equation_diagram__equal_side_variable_value",
    "task_geometry__polygon_equation_diagram__equal_side_length_value",
    "task_geometry__polygon_equation_diagram__equal_angle_variable_value",
    "task_geometry__polygon_equation_diagram__equal_angle_measure_value",
    "task_geometry__polygon_equation_diagram__interior_angle_sum_variable_value",
    "task_geometry__polygon_equation_diagram__interior_angle_sum_angle_value",
    "task_geometry__polygon_equation_diagram__side_expression_perimeter_value",
)


def _eval_linear_label(label: str, *, variable_name: str, variable_value: int) -> int:
    text = str(label).strip().removesuffix("°")
    if text.startswith("(") and text.endswith(")"):
        text = text[1:-1]
    pattern = rf"(?P<sign>-?)(?P<coeff>\d*){re.escape(str(variable_name))}(?P<offset>[+-]\d+)?"
    match = re.fullmatch(pattern, text)
    assert match is not None, text
    coefficient = int(match.group("coeff") or "1")
    if match.group("sign") == "-":
        coefficient *= -1
    offset = int(match.group("offset") or "0")
    return int(coefficient * int(variable_value) + offset)


def _eval_side_label(label: str, *, variable_name: str, variable_value: int) -> int:
    text = str(label).strip()
    if re.fullmatch(r"-?\d+", text):
        return int(text)
    return _eval_linear_label(text, variable_name=variable_name, variable_value=variable_value)


@pytest.mark.parametrize("task_id", TASK_IDS)
def test_polygon_equation_diagram_registered_task(task_id: str) -> None:
    assert create_task(task_id).task_id == task_id


@pytest.mark.parametrize("task_id", TASK_IDS)
@pytest.mark.parametrize("side_count", [3, 4, 5, 6])
def test_polygon_equation_diagram_generates_each_side_count(task_id: str, side_count: int) -> None:
    out = create_task(task_id).generate(
        20260627 + side_count,
        params={"side_count": side_count},
        max_attempts=40,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert out.scene_id == "polygon_equation_diagram"
    assert out.query_id == SINGLE_QUERY_ID
    assert out.answer_gt.type == "integer"
    assert out.answer_gt.value == execution["answer"]
    assert execution["side_count"] == side_count
    assert execution["formula_schema"]
    assert execution["relation"]

    assert out.annotation_gt.type == "point_map"
    annotation = out.annotation_gt.value
    expected_keys = {chr(ord("A") + index) for index in range(side_count)}
    assert set(annotation) == expected_keys
    width, height = out.image.size
    for point in annotation.values():
        assert 0 <= point[0] <= width
        assert 0 <= point[1] <= height

    if "interior_angle_sum" in task_id:
        assert sum(execution["numeric_angle_values"]) == (side_count - 2) * 180
    if task_id.endswith("side_expression_perimeter_value"):
        assert execution["answer"] == execution["perimeter_value"]
        assert execution["perimeter_value"] == sum(execution["perimeter_side_values"].values())
        assert len(execution["side_labels"]) == side_count
        assert set(execution["side_labels"]) == set(execution["perimeter_side_values"])
        for side_label, side_value in execution["perimeter_side_values"].items():
            assert _eval_side_label(
                execution["side_labels"][side_label],
                variable_name=execution["variable_name"],
                variable_value=execution["variable_value"],
            ) == side_value
        for side_label in execution["equal_sides"]:
            assert execution["side_mark_counts"][side_label] == 2
        assert execution["distractor_mode"] == bool(execution["side_distractors"])
        for side_label, side_label_text in execution["side_labels"].items():
            if side_label not in execution["equal_sides"] and side_label not in {
                distractor["side"] for distractor in execution["side_distractors"]
            }:
                assert side_label_text.isdigit()
    if "equal_side" in task_id:
        for side_label in execution["equal_sides"]:
            assert execution["side_mark_counts"][side_label] == 2
        assert execution["side_distractors"]
        for distractor in execution["side_distractors"]:
            assert execution["side_mark_counts"][distractor["side"]] in {1, 3}
            assert execution["side_mark_counts"][distractor["side"]] != 2
            if distractor["variable_name"] == execution["variable_name"]:
                assert _eval_linear_label(
                    distractor["label"],
                    variable_name=execution["variable_name"],
                    variable_value=execution["variable_value"],
                ) == distractor["numeric_value_under_x"]
                assert distractor["numeric_value_under_x"] != execution["equal_side_length"]
    if "equal_angle" in task_id:
        for vertex_label in execution["equal_angles"]:
            assert execution["angle_mark_counts"][vertex_label] == 2
        assert execution["distractor_mode"] == bool(execution["angle_distractors"])
        for distractor in execution["angle_distractors"]:
            assert execution["angle_mark_counts"][distractor["vertex"]] in {1, 3}
            assert execution["angle_mark_counts"][distractor["vertex"]] != 2
            if distractor["variable_name"] == execution["variable_name"]:
                assert _eval_linear_label(
                    distractor["label"],
                    variable_name=execution["variable_name"],
                    variable_value=execution["variable_value"],
                ) == distractor["numeric_value_under_x"]
                assert distractor["numeric_value_under_x"] != execution["equal_angle_measure"]
    label_blob = json.dumps(
        {
            "side_labels": execution.get("side_labels", {}),
            "angle_labels": execution.get("angle_labels", {}),
        }
    )
    assert "1x" not in label_blob
    assert "task_variant" not in json.dumps(trace)
    assert "query_variant" not in json.dumps(trace)


def test_polygon_equation_diagram_generation_is_deterministic() -> None:
    task_id = "task_geometry__polygon_equation_diagram__interior_angle_sum_variable_value"
    params = {"side_count": 6}
    first = create_task(task_id).generate(123456, params=params, max_attempts=40)
    second = create_task(task_id).generate(123456, params=params, max_attempts=40)

    assert first.prompt == second.prompt
    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
