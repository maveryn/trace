"""Behavior tests for pages process-flow diagram tasks."""

from __future__ import annotations

from collections import Counter

from trace_tasks.core.seed import hash64
from trace_tasks.core.taxonomy import resolve_task_taxonomy
from trace_tasks.tasks.pages.process_flow.condition_path_endpoint_label import (
    PROMPT_QUERY_KEY as CONDITION_PATH_ENDPOINT_PROMPT_QUERY_KEY,
)
from trace_tasks.tasks.pages.process_flow.condition_path_endpoint_label import (
    TASK_ID as CONDITION_PATH_ENDPOINT_TASK_ID,
)
from trace_tasks.tasks.pages.process_flow.condition_path_endpoint_label import (
    PagesProcessFlowConditionPathEndpointLabelTask,
)
from trace_tasks.tasks.pages.process_flow.filtered_node_count import (
    TASK_ID as FILTERED_NODE_COUNT_TASK_ID,
)
from trace_tasks.tasks.pages.process_flow.filtered_node_count import (
    PagesProcessFlowFilteredNodeCountTask,
)
from trace_tasks.tasks.pages.process_flow.lane_filtered_handoff_count import (
    TASK_ID as LANE_FILTERED_HANDOFF_COUNT_TASK_ID,
)
from trace_tasks.tasks.pages.process_flow.lane_filtered_handoff_count import (
    PagesProcessFlowLaneFilteredHandoffCountTask,
)


def _assert_bboxes_inside_image(out) -> None:
    width, height = out.image.size
    annotation_value = out.annotation_gt.value
    bboxes = annotation_value.values() if isinstance(annotation_value, dict) else annotation_value
    for bbox in bboxes:
        x0, y0, x1, y1 = [float(value) for value in bbox]
        assert 0.0 <= x0 <= x1 <= float(width)
        assert 0.0 <= y0 <= y1 <= float(height)


def _assert_segments_inside_image(out) -> None:
    width, height = out.image.size
    for segment in out.annotation_gt.value:
        assert len(segment) == 2
        for point in segment:
            x, y = [float(value) for value in point]
            assert 0.0 <= x <= float(width)
            assert 0.0 <= y <= float(height)


def test_pages_process_flow_tasks_are_registered_in_public_taxonomy() -> None:
    for task_id in [
        FILTERED_NODE_COUNT_TASK_ID,
        CONDITION_PATH_ENDPOINT_TASK_ID,
        LANE_FILTERED_HANDOFF_COUNT_TASK_ID,
    ]:
        taxonomy = resolve_task_taxonomy(task_id)
        assert taxonomy.domain == "pages"
        assert taxonomy.scene_id == "process_flow"
        assert taxonomy.source_scene_id == ""


def test_pages_process_flow_filtered_node_count_contract() -> None:
    task = PagesProcessFlowFilteredNodeCountTask()
    for query_id in ("shape_node_count", "status_node_count", "role_node_count"):
        out = task.generate(83100, params={"query_id": query_id, "layout_variant": "horizontal_swimlane"}, max_attempts=10)
        trace = out.trace_payload
        query = trace["execution_trace"]["query"]
        annotation_ids = [str(item) for item in query["annotation_node_ids"]]
        expected = [trace["render_map"]["node_bboxes_px"][node_id] for node_id in annotation_ids]

        assert out.scene_id == "process_flow"
        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "bbox_set"
        assert int(out.answer_gt.value) == int(query["answer"])
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)
        assert out.annotation_gt.value == expected
        assert sorted(out.prompt_variants) == ["answer_and_annotation", "answer_only"]
        _assert_bboxes_inside_image(out)


def test_pages_process_flow_condition_path_endpoint_contract() -> None:
    task = PagesProcessFlowConditionPathEndpointLabelTask()
    out = task.generate(83120, params={"layout_variant": "horizontal_swimlane"}, max_attempts=10)
    trace = out.trace_payload
    query = trace["execution_trace"]["query"]
    render_map = trace["render_map"]
    expected = {}
    for role in query["annotation_roles"]:
        if str(role["kind"]) == "node":
            expected[str(role["key"])] = render_map["node_bboxes_px"][str(role["id"])]
        else:
            expected[str(role["key"])] = render_map["edge_label_bboxes_px"][str(role["id"])]

    assert out.scene_id == "process_flow"
    assert out.query_id == "single"
    assert trace["execution_trace"]["prompt_query_key"] == CONDITION_PATH_ENDPOINT_PROMPT_QUERY_KEY
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "bbox_map"
    assert str(out.answer_gt.value) == str(query["answer"])
    assert out.annotation_gt.value == expected
    assert list(out.annotation_gt.value) == [
        "start_step",
        "first_decision_label",
        "second_decision_label",
        "endpoint_step",
    ]
    assert len(query["condition_labels"]) == 2
    for label in query["condition_labels"]:
        assert f'"{label}"' in out.prompt
    _assert_bboxes_inside_image(out)


def test_pages_process_flow_handoff_count_contract() -> None:
    task = PagesProcessFlowLaneFilteredHandoffCountTask()
    for query_id in ("lane_outgoing_handoff_count", "lane_involved_handoff_count"):
        out = task.generate(83140, params={"query_id": query_id, "layout_variant": "horizontal_swimlane"}, max_attempts=10)
        trace = out.trace_payload
        query = trace["execution_trace"]["query"]
        annotation_ids = [str(item) for item in query["annotation_edge_ids"]]
        expected = [trace["render_map"]["edge_segments_px"][edge_id] for edge_id in annotation_ids]

        assert out.scene_id == "process_flow"
        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "segment_set"
        assert int(out.answer_gt.value) == int(query["answer"])
        assert len(out.annotation_gt.value) == int(out.answer_gt.value)
        assert out.annotation_gt.value == expected
        _assert_segments_inside_image(out)


def test_pages_process_flow_generation_is_deterministic() -> None:
    task = PagesProcessFlowConditionPathEndpointLabelTask()
    params = {"layout_variant": "horizontal_swimlane", "style_variant": "warm_memo"}
    out_a = task.generate(83180, params=params, max_attempts=10)
    out_b = task.generate(83180, params=params, max_attempts=10)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_pages_process_flow_sampling_covers_visual_and_text_axes() -> None:
    task = PagesProcessFlowFilteredNodeCountTask()
    layouts: Counter[str] = Counter()
    styles: Counter[str] = Counter()
    scenes: Counter[str] = Counter()
    queries: Counter[str] = Counter()

    for index in range(24):
        out = task.generate(hash64(83220, "process_flow_axes", index), params={}, max_attempts=10)
        execution = out.trace_payload["execution_trace"]
        layouts[str(execution["layout_variant"])] += 1
        styles[str(execution["style_variant"])] += 1
        scenes[str(execution["scene_variant"])] += 1
        queries[str(execution["query_id"])] += 1

    assert set(layouts) == {"horizontal_swimlane"}
    assert set(styles) == {"blueprint", "pastel_cards", "graphite", "warm_memo"}
    assert len(scenes) >= 5
    assert set(queries) == {"shape_node_count", "status_node_count", "role_node_count"}
