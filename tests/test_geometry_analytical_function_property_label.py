"""Contract tests for geometry function-panel property label tasks."""

from __future__ import annotations

from collections import Counter

import pytest

from trace_tasks.core.seed import hash64
from trace_tasks.tasks.geometry.function_panels.function_status_label import (
    TASK_ID as FUNCTION_TASK_ID,
    GeometryFunctionPanelsFunctionStatusLabelTask,
)
from trace_tasks.tasks.geometry.function_panels.one_to_one_status_label import GeometryFunctionPanelsOneToOneStatusLabelTask
from trace_tasks.tasks.geometry.function_panels.range_match_label import GeometryFunctionPanelsRangeMatchLabelTask
from trace_tasks.tasks.geometry.function_panels.x_axis_symmetry_label import GeometryFunctionPanelsXAxisSymmetryLabelTask


def _matching_labels(rule: str, relations_by_label: dict[str, dict], answer_label: str) -> list[str]:
    if rule == "function_test":
        return [label for label, relation in relations_by_label.items() if bool(relation["is_function"])]
    if rule == "injective_function_test":
        return [
            label
            for label, relation in relations_by_label.items()
            if bool(relation["is_function"]) and bool(relation["is_one_to_one"])
        ]
    if rule == "horizontal_axis_symmetry":
        return [label for label, relation in relations_by_label.items() if bool(relation["symmetric_about_x_axis"])]
    if rule == "range_interval_match":
        target = relations_by_label[str(answer_label)]["range"]
        return [label for label, relation in relations_by_label.items() if relation["range"] == target]
    raise AssertionError(f"unsupported rule in test: {rule}")


@pytest.mark.parametrize(
    "task_cls",
    (
        GeometryFunctionPanelsFunctionStatusLabelTask,
        GeometryFunctionPanelsOneToOneStatusLabelTask,
        GeometryFunctionPanelsRangeMatchLabelTask,
        GeometryFunctionPanelsXAxisSymmetryLabelTask,
    ),
)
def test_geometry_function_panel_property_label_contract(task_cls) -> None:
    task = task_cls()
    out = task.generate(
        hash64(93210, task.task_id, 3),
        params={"query_id": "single", "winner_label": "C"},
        max_attempts=10,
    )

    assert out.query_id == "single"
    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox"
    assert len(out.annotation_gt.value) == 4
    assert out.trace_payload["projected_annotation"]["bbox"] == out.annotation_gt.value
    assert out.trace_payload["query_spec"]["template_id"] == "geometry_analytical_function_property_v1"
    assert out.trace_payload["query_spec"]["prompt_variant"]["prompt_schema_version"] == "v1"
    assert out.image.size == (1024, 720)

    relations = out.trace_payload["execution_trace"]["relations_by_label"]
    assert 4 <= len(relations) <= 6
    assert "C" in relations
    assert set(relations).issubset({"A", "B", "C", "D", "E", "F"})
    rule = out.trace_payload["execution_trace"]["property_rule"]
    assert _matching_labels(str(rule), relations, "C") == ["C"]


def test_geometry_function_panel_property_label_balances_answers() -> None:
    task = GeometryFunctionPanelsFunctionStatusLabelTask()
    labels = Counter()

    for index in range(60):
        out = task.generate(hash64(93220, FUNCTION_TASK_ID, index), params={}, max_attempts=10)
        labels[str(out.answer_gt.value)] += 1

    assert set(labels.keys()).issubset({"A", "B", "C", "D", "E", "F"})
    assert len(labels) >= 5
    assert max(labels.values()) <= 18


def test_geometry_function_panel_property_label_randomizes_relation_geometry() -> None:
    task = GeometryFunctionPanelsOneToOneStatusLabelTask()
    observed_winner_points = set()
    observed_domains = set()

    for index in range(12):
        out = task.generate(
            hash64(93230, task.task_id, index),
            params={"query_id": "single", "winner_label": "A"},
            max_attempts=10,
        )
        winner = out.trace_payload["execution_trace"]["winner_relation"]
        observed_winner_points.add(tuple(tuple(point) for point in winner["points"]))
        observed_domains.add(tuple(winner["domain"]))

    assert len(observed_winner_points) >= 6
    assert len(observed_domains) >= 4
