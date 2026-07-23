"""Behavior tests for standard Sankey chart tasks."""

from __future__ import annotations

from collections import Counter

import pytest

from tests.helpers import assert_counter_support_within, extract_prompt_json_example
from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.charts.sankey.node_side_total_value import (
    SOURCE_OUTGOING_QUERY_ID,
    TARGET_INCOMING_QUERY_ID,
    SUPPORTED_QUERY_IDS as NODE_SIDE_QUERY_IDS,
    ChartsFlowSankeyNodeSideTotalValuePublicTask,
)
from trace_tasks.tasks.charts.sankey.path_bottleneck_value import ChartsFlowSankeyPathBottleneckValuePublicTask
from trace_tasks.tasks.charts.sankey.source_to_target_total_flow import ChartsFlowSankeySourceToTargetTotalFlowPublicTask
from trace_tasks.tasks.registry import list_default_task_ids


def _assert_point_inside_canvas(point: list[float], *, width: int, height: int) -> None:
    assert len(point) == 2
    x, y = [float(value) for value in point]
    assert 0 <= x <= width
    assert 0 <= y <= height


def _expected_source_target_total(execution: dict) -> int:
    return sum(min(int(path["first_value"]), int(path["second_value"])) for path in execution["query_path_details"])


def _expected_bottleneck(execution: dict) -> int:
    assert len(execution["query_path_details"]) == 1
    path = dict(execution["query_path_details"][0])
    return min(int(path["first_value"]), int(path["second_value"]))


def _expected_node_total(execution: dict) -> int:
    if str(execution["query_id"]) == SOURCE_OUTGOING_QUERY_ID:
        return sum(int(path["first_value"]) for path in execution["query_path_details"])
    if str(execution["query_id"]) == TARGET_INCOMING_QUERY_ID:
        return sum(int(path["second_value"]) for path in execution["query_path_details"])
    raise AssertionError(f"unsupported Sankey node-side branch: {execution['query_id']}")


@pytest.mark.parametrize(
    ("task_cls", "seed", "expected_answer", "expected_annotation_type"),
    [
        (ChartsFlowSankeySourceToTargetTotalFlowPublicTask, 69100, _expected_source_target_total, "point_set"),
        (ChartsFlowSankeyPathBottleneckValuePublicTask, 69140, _expected_bottleneck, "point"),
    ],
)
def test_charts_sankey_single_branch_tasks_match_contract(task_cls, seed: int, expected_answer, expected_annotation_type: str) -> None:
    task = task_cls()
    out = task.generate(seed, params={"query_id": SINGLE_QUERY_ID}, max_attempts=100)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render = trace["render_spec"]
    render_map = trace["render_map"]
    assert task.task_id in list_default_task_ids()
    assert task.supported_query_ids == (SINGLE_QUERY_ID,)
    assert out.scene_id == "sankey"
    assert out.query_id == SINGLE_QUERY_ID
    assert str(execution["query_id"]) == SINGLE_QUERY_ID
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == expected_annotation_type
    assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
    assert 2 <= int(execution["source_count"]) <= 3
    assert 2 <= int(execution["middle_count"]) <= 3
    assert 2 <= int(execution["target_count"]) <= 3
    assert 3 <= int(execution["path_count"]) <= 4
    assert int(out.answer_gt.value) == int(expected_answer(execution))
    refs = [str(value) for value in execution["annotation_segment_ids"]]
    expected_points = [render_map["segment_centers_px"][ref] for ref in refs]
    if str(expected_annotation_type) == "point":
        assert len(expected_points) == 1
        assert out.annotation_gt.value == expected_points[0]
        assert trace["projected_annotation"]["point"] == out.annotation_gt.value
        points_to_check = [out.annotation_gt.value]
    else:
        assert out.annotation_gt.value == expected_points
        assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
        points_to_check = list(out.annotation_gt.value)
    assert trace["projected_annotation"]["type"] == expected_annotation_type
    assert trace["query_spec"]["params"]["query_id"] == SINGLE_QUERY_ID
    assert str(render["font_assets"]["font_asset_version"])
    assert str(render["font_assets"]["chart_font_family"])
    for point in points_to_check:
        _assert_point_inside_canvas([float(value) for value in point], width=int(render["canvas_width"]), height=int(render["canvas_height"]))


@pytest.mark.parametrize("query_id", NODE_SIDE_QUERY_IDS)
def test_charts_sankey_node_side_total_matches_contract(query_id: str) -> None:
    task = ChartsFlowSankeyNodeSideTotalValuePublicTask()
    out = task.generate(69250 + NODE_SIDE_QUERY_IDS.index(query_id), params={"query_id": query_id}, max_attempts=100)
    trace = out.trace_payload
    execution = trace["execution_trace"]
    render = trace["render_spec"]
    render_map = trace["render_map"]
    assert task.task_id in list_default_task_ids()
    assert out.scene_id == "sankey"
    assert out.query_id == query_id
    assert str(execution["query_id"]) == query_id
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "point_set"
    assert int(out.answer_gt.value) == int(_expected_node_total(execution))
    assert trace["query_spec"]["params"]["query_id"] == query_id
    refs = [str(value) for value in execution["annotation_segment_ids"]]
    expected_points = [render_map["segment_centers_px"][ref] for ref in refs]
    assert out.annotation_gt.value == expected_points
    for point in out.annotation_gt.value:
        _assert_point_inside_canvas([float(value) for value in point], width=int(render["canvas_width"]), height=int(render["canvas_height"]))


def test_charts_sankey_prompt_examples_match_contract() -> None:
    for task_cls in (
        ChartsFlowSankeySourceToTargetTotalFlowPublicTask,
        ChartsFlowSankeyPathBottleneckValuePublicTask,
        ChartsFlowSankeyNodeSideTotalValuePublicTask,
    ):
        out = task_cls().generate(69300 + len(task_cls.__name__), params={}, max_attempts=100)
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert isinstance(answer_and_annotation["answer"], int)
        assert isinstance(answer_only["answer"], int)
        annotation = answer_and_annotation["annotation"]
        assert isinstance(annotation, list)
        if task_cls is ChartsFlowSankeyPathBottleneckValuePublicTask:
            assert len(annotation) == 2 and all(isinstance(value, int) for value in annotation)
        else:
            assert annotation and all(len(point) == 2 for point in annotation)


def test_charts_sankey_node_side_sampling_covers_branches() -> None:
    task = ChartsFlowSankeyNodeSideTotalValuePublicTask()
    counts: Counter[str] = Counter()
    for index in range(40):
        out = task.generate(hash64(69400, "charts_sankey_node_side", index), params={}, max_attempts=100)
        counts[str(out.query_id)] += 1
    assert_counter_support_within(counts, NODE_SIDE_QUERY_IDS, expected_per_key=20, tolerance=5)


def test_charts_sankey_generation_is_deterministic() -> None:
    params = {"query_id": TARGET_INCOMING_QUERY_ID}
    first = ChartsFlowSankeyNodeSideTotalValuePublicTask().generate(69500, params=params, max_attempts=100)
    second = ChartsFlowSankeyNodeSideTotalValuePublicTask().generate(69500, params=params, max_attempts=100)
    assert first.prompt == second.prompt
    assert first.answer_gt.to_dict() == second.answer_gt.to_dict()
    assert first.annotation_gt.to_dict() == second.annotation_gt.to_dict()
    assert first.trace_payload["execution_trace"] == second.trace_payload["execution_trace"]
