"""Contract tests for function-panel interval-sign label tasks."""

from __future__ import annotations

import pytest

from trace_tasks.tasks.geometry.function_panels.function_status_label import GeometryFunctionPanelsFunctionStatusLabelTask
from trace_tasks.tasks.geometry.function_panels.sign_interval_label import GeometryFunctionPanelsSignIntervalLabelTask


@pytest.mark.parametrize("query_id", ("sign_interval_positive_label", "sign_interval_negative_label"))
def test_geometry_function_panel_interval_tasks_emit_expected_contract(query_id: str) -> None:
    out = GeometryFunctionPanelsSignIntervalLabelTask().generate(
        24011,
        params={"query_id": query_id, "winner_label": "C"},
        max_attempts=20,
    )

    assert out.answer_gt.type == "option_letter"
    assert out.answer_gt.value == "C"
    assert out.annotation_gt.type == "bbox"
    assert out.query_id == query_id
    assert out.trace_payload["query_spec"]["params"]["query_id"] == query_id
    expected_interval = "[-4, 4]"
    assert out.trace_payload["query_spec"]["params"]["target_interval"] == expected_interval
    assert out.trace_payload["execution_trace"]["target_interval"] == expected_interval
    assert out.trace_payload["projected_annotation"]["bbox"] == out.annotation_gt.value


def test_geometry_function_panel_single_query_task_rejects_unknown_branch() -> None:
    with pytest.raises(ValueError):
        GeometryFunctionPanelsFunctionStatusLabelTask().generate(
            24012,
            params={"query_id": "positive_interval_label"},
            max_attempts=20,
        )
