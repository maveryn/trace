from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from trace_tasks.tasks.geometry.polar_graph_paper.readout_value import (
    QUERY_IDS,
    TASK_ID,
    PolarGraphPaperReadoutValueTask,
)
from trace_tasks.tasks.geometry.polar_graph_paper.coordinate_difference_value import (
    QUERY_IDS as DIFFERENCE_QUERY_IDS,
    TASK_ID as DIFFERENCE_TASK_ID,
    PolarGraphPaperCoordinateDifferenceValueTask,
)
from trace_tasks.tasks.geometry.polar_graph_paper.coordinate_value_point_count import (
    QUERY_IDS as COUNT_QUERY_IDS,
    TASK_ID as COUNT_TASK_ID,
    PolarGraphPaperCoordinateValuePointCountTask,
)


def _task() -> PolarGraphPaperReadoutValueTask:
    return PolarGraphPaperReadoutValueTask()


def _difference_task() -> PolarGraphPaperCoordinateDifferenceValueTask:
    return PolarGraphPaperCoordinateDifferenceValueTask()


def _count_task() -> PolarGraphPaperCoordinateValuePointCountTask:
    return PolarGraphPaperCoordinateValuePointCountTask()


@pytest.mark.parametrize("query_id", QUERY_IDS)
def test_polar_graph_paper_readout_generates_each_query(query_id: str) -> None:
    output = _task().generate(
        17,
        params={"query_id": query_id, "radius": 5, "theta_degrees": 120},
        max_attempts=1,
    )

    assert output.trace_payload["scene_ir"]["task_id"] == TASK_ID
    assert output.answer_gt.type == "integer"
    expected = 5 if query_id == "radius_readout_value" else 120
    assert output.answer_gt.value == expected
    assert output.annotation_gt.type == "point"
    assert len(output.annotation_gt.value) == 2
    assert output.query_id == query_id
    assert output.scene_id == "polar_graph_paper"

    render_map = output.trace_payload["render_map"]
    assert "option_values_by_label" not in render_map
    assert output.trace_payload["execution_trace"]["correct_value"] == output.answer_gt.value


@pytest.mark.parametrize(
    ("query_id", "expected"),
    (
        ("radius_difference_value", 5),
        ("angle_difference_value", 60),
    ),
)
def test_polar_graph_paper_coordinate_difference_generates_each_query(query_id: str, expected: int) -> None:
    output = _difference_task().generate(
        19,
        params={
            "query_id": query_id,
            "radius_p": 3,
            "radius_q": 8,
            "theta_degrees_p": 330,
            "theta_degrees_q": 30,
        },
        max_attempts=1,
    )

    assert output.trace_payload["scene_ir"]["task_id"] == DIFFERENCE_TASK_ID
    assert output.answer_gt.type == "integer"
    assert output.answer_gt.value == expected
    assert output.annotation_gt.type == "point_map"
    assert set(output.annotation_gt.value) == {"P", "Q"}
    assert output.query_id == query_id
    assert output.scene_id == "polar_graph_paper"
    assert output.trace_payload["execution_trace"]["correct_value"] == output.answer_gt.value


@pytest.mark.parametrize(
    ("query_id", "params"),
    (
        ("radius_value_point_count", {"target_radius": 4}),
        ("angle_value_point_count", {"target_angle_degrees": 120}),
    ),
)
def test_polar_graph_paper_coordinate_value_point_count_generates_each_query(
    query_id: str,
    params: dict[str, int],
) -> None:
    output = _count_task().generate(
        31,
        params={
            "query_id": query_id,
            "answer_count": 3,
            "total_point_count": 10,
            **params,
        },
        max_attempts=1,
    )

    assert output.trace_payload["scene_ir"]["task_id"] == COUNT_TASK_ID
    assert output.answer_gt.type == "integer"
    assert output.answer_gt.value == 3
    assert output.annotation_gt.type == "point_set"
    assert len(output.annotation_gt.value) == output.answer_gt.value
    assert output.query_id == query_id
    assert output.scene_id == "polar_graph_paper"

    execution_trace = output.trace_payload["execution_trace"]
    assert execution_trace["correct_value"] == output.answer_gt.value
    assert execution_trace["total_point_count"] == 10
    assert len(execution_trace["matching_labels"]) == output.answer_gt.value
    assert len(output.trace_payload["render_map"]["points_by_label"]) == 10


def test_polar_graph_paper_generation_is_deterministic() -> None:
    first = _task().generate(23, params={"query_id": "angle_readout_value"}, max_attempts=1)
    second = _task().generate(23, params={"query_id": "angle_readout_value"}, max_attempts=1)

    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]


def test_polar_graph_paper_difference_generation_is_deterministic() -> None:
    first = _difference_task().generate(23, params={"query_id": "angle_difference_value"}, max_attempts=1)
    second = _difference_task().generate(23, params={"query_id": "angle_difference_value"}, max_attempts=1)

    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]


def test_polar_graph_paper_coordinate_value_point_count_is_deterministic() -> None:
    first = _count_task().generate(23, params={"query_id": "angle_value_point_count"}, max_attempts=1)
    second = _count_task().generate(23, params={"query_id": "angle_value_point_count"}, max_attempts=1)

    assert first.answer_gt == second.answer_gt
    assert first.annotation_gt == second.annotation_gt
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]


def test_polar_graph_paper_invalid_query_id_raises() -> None:
    with pytest.raises(ValueError):
        _task().generate(1, params={"query_id": "x_component"}, max_attempts=1)

    with pytest.raises(ValueError):
        _difference_task().generate(1, params={"query_id": "x_component"}, max_attempts=1)

    with pytest.raises(ValueError):
        _count_task().generate(1, params={"query_id": "x_component"}, max_attempts=1)


def test_polar_graph_paper_config_has_no_query_routing() -> None:
    config = yaml.safe_load(Path("src/trace_tasks/resources/configs/domains/geometry/polar_graph_paper.yaml").read_text())
    assert "query_weights" not in str(config)
    assert "queries" not in config.get("prompt", {}).get("shared", {})
    assert set(DIFFERENCE_QUERY_IDS) == {"radius_difference_value", "angle_difference_value"}
    assert set(COUNT_QUERY_IDS) == {"radius_value_point_count", "angle_value_point_count"}
