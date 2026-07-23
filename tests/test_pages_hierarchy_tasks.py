"""Behavior tests for pages hierarchy org-chart tasks."""

from __future__ import annotations

import importlib
from collections import Counter, defaultdict

import pytest

from trace_tasks.core.query_ids import SINGLE_QUERY_ID
from trace_tasks.core.seed import hash64
from trace_tasks.tasks.pages.hierarchy.manager_most_direct_reports_label import (
    PagesHierarchyManagerMostDirectReportsLabelTask,
)
from trace_tasks.tasks.pages.hierarchy.manager_most_total_reports_label import (
    PagesHierarchyManagerMostTotalReportsLabelTask,
)
from trace_tasks.tasks.pages.hierarchy.subtree_descendant_count import (
    PagesHierarchySubtreeDescendantCountTask,
)
from tests.helpers import extract_prompt_json_example


def _direct_children(edge_specs: list[dict]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for edge in edge_specs:
        children[str(edge["source_node_id"])].append(str(edge["target_node_id"]))
    return dict(children)


def _descendants(node_id: str, children_by_parent: dict[str, list[str]]) -> list[str]:
    collected: list[str] = []

    def _walk(parent_id: str) -> None:
        for child_id in children_by_parent.get(str(parent_id), []):
            collected.append(str(child_id))
            _walk(str(child_id))

    _walk(str(node_id))
    return collected


def test_pages_hierarchy_active_org_chart_contracts() -> None:
    task_cases = (
        (PagesHierarchySubtreeDescendantCountTask(), "subtree_descendant_count", "integer", "bbox_set"),
        (PagesHierarchyManagerMostTotalReportsLabelTask(), "manager_most_total_reports_label", "string", "bbox"),
        (PagesHierarchyManagerMostDirectReportsLabelTask(), "manager_most_direct_reports_label", "string", "bbox"),
    )

    for query_id_index, (task, source_query_id, answer_type, annotation_type) in enumerate(task_cases):
        out = task.generate(
            61400 + query_id_index,
            params={"query_id": SINGLE_QUERY_ID, "scene_variant": "org_chart", "pages_context_text_enabled": False},
            max_attempts=10,
        )
        trace = out.trace_payload
        execution = trace["execution_trace"]
        render = trace["render_spec"]
        render_map = trace["render_map"]

        assert out.answer_gt.type == answer_type
        assert out.annotation_gt.type == annotation_type
        assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]
        assert str(out.query_id) == SINGLE_QUERY_ID
        assert str(execution["query_id"]) == SINGLE_QUERY_ID
        assert str(execution["source_query_id"]) == source_query_id
        assert str(execution["prompt_query_key"]) == source_query_id
        assert str(execution["scene_variant"]) == "org_chart"
        assert str(execution["question_format"]) == "hierarchy_org_chart"
        assert str(execution["view_family"]) == "org_chart_diagram"
        assert "CEO" in {str(node["node_label"]) for node in execution["node_specs"]}
        assert out.image.size == (int(render["canvas_width"]), int(render["canvas_height"]))
        assert 16 <= int(execution["tree_node_count"]) <= 30
        assert 4 <= int(execution["tree_depth"]) <= 8
        assert len(execution["node_specs"]) == int(execution["tree_node_count"])
        assert len(render_map["node_bboxes_px"]) == int(execution["tree_node_count"])
        assert len(render_map["edge_bboxes_px"]) == len(execution["edge_specs"])
        assert "node" not in out.prompt.lower()
        assert "leaf" not in out.prompt.lower()
        assert "subtree" not in out.prompt.lower()
        assert "path" not in out.prompt.lower()
        assert "hop" not in out.prompt.lower()

        if source_query_id == "subtree_descendant_count":
            annotation_bboxes = [[float(value) for value in bbox] for bbox in out.annotation_gt.value]
            annotation_bbox_ids = [str(bbox_id) for bbox_id in execution["annotation_node_bbox_ids"]]
            expected_bboxes = [
                [float(value) for value in render_map["node_bboxes_px"][str(bbox_id)]]
                for bbox_id in annotation_bbox_ids
            ]
            prompt_slots = dict(execution["query_prompt_slots"])
            for label in execution["query_node_labels"]:
                assert str(label) in {str(value) for value in prompt_slots.values()}
                assert f'"{label}"' in out.prompt
            assert trace["projected_annotation"]["bbox_set"] == annotation_bboxes
            assert annotation_bboxes == expected_bboxes
            assert int(out.answer_gt.value) == int(execution["answer_count"])
            assert int(execution["answer_count"]) == int(execution["descendant_count"])
            assert 3 <= int(execution["answer_count"]) <= 7
            assert len(execution["annotation_node_ids"]) == int(execution["answer_count"])
            assert str(execution["annotation_semantics"]) == "all_reports_under_named_manager"
        else:
            answer_bbox_id = str(execution["answer_node_bbox_id"])
            expected_bbox = [float(value) for value in render_map["node_bboxes_px"][answer_bbox_id]]
            assert [float(value) for value in out.annotation_gt.value] == expected_bbox
            assert trace["projected_annotation"]["bbox"] == out.annotation_gt.value
            assert str(out.answer_gt.value) == str(execution["answer_node_label"])
            assert str(execution["answer_value"]) == str(execution["answer_node_label"])
            assert str(execution["answer_node_id"]) != str(execution["root_node_id"])
            assert str(execution["annotation_semantics"]) == "selected_manager_node"
            candidate_counts = [dict(row) for row in execution["candidate_manager_counts"]]
            assert candidate_counts
            max_count = max(int(row["count"]) for row in candidate_counts)
            winners = [row for row in candidate_counts if int(row["count"]) == int(max_count)]
            assert len(winners) == 1
            assert str(winners[0]["node_id"]) == str(execution["answer_node_id"])
            assert int(winners[0]["count"]) == int(execution["answer_metric_count"])


def test_pages_hierarchy_prompt_examples_match_answer_contracts() -> None:
    expected = (
        (PagesHierarchySubtreeDescendantCountTask(), "subtree_descendant_count", int),
        (PagesHierarchyManagerMostTotalReportsLabelTask(), "manager_most_total_reports_label", str),
        (PagesHierarchyManagerMostDirectReportsLabelTask(), "manager_most_direct_reports_label", str),
    )

    for index, (task, source_query_id, answer_cls) in enumerate(expected, start=61460):
        out = task.generate(index, params={"query_id": SINGLE_QUERY_ID, "scene_variant": "org_chart"}, max_attempts=10)
        assert out.trace_payload["execution_trace"]["source_query_id"] == source_query_id
        answer_and_annotation = extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
        answer_only = extract_prompt_json_example(out.prompt_variants["answer_only"])
        assert isinstance(answer_and_annotation["answer"], answer_cls)
        assert isinstance(answer_only["answer"], answer_cls)
        assert isinstance(answer_and_annotation["annotation"], list)
        if source_query_id == "subtree_descendant_count":
            assert all(isinstance(item, list) for item in answer_and_annotation["annotation"])
        else:
            assert len(answer_and_annotation["annotation"]) == 4


def test_pages_hierarchy_org_chart_is_deterministic() -> None:
    task = PagesHierarchyManagerMostTotalReportsLabelTask()
    params = {"query_id": SINGLE_QUERY_ID, "scene_variant": "org_chart"}
    out_a = task.generate(61510, params=params, max_attempts=10)
    out_b = task.generate(61510, params=params, max_attempts=10)

    assert out_a.answer_gt.to_dict() == out_b.answer_gt.to_dict()
    assert out_a.annotation_gt.to_dict() == out_b.annotation_gt.to_dict()
    assert out_a.trace_payload["execution_trace"] == out_b.trace_payload["execution_trace"]
    assert out_a.trace_payload["query_spec"]["prompt_variant"] == out_b.trace_payload["query_spec"]["prompt_variant"]
    assert out_a.prompt == out_b.prompt
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_pages_hierarchy_total_reports_retries_infeasible_tree_size() -> None:
    task = PagesHierarchyManagerMostTotalReportsLabelTask()
    out_a = task.generate(500714, params={}, max_attempts=100)
    out_b = task.generate(500714, params={}, max_attempts=100)

    execution = out_a.trace_payload["execution_trace"]
    assert int(execution["generation_attempt_index"]) == 1
    assert int(execution["generation_attempt_seed"]) == hash64(
        500714,
        f"{task.task_id}.hierarchy_retry",
        1,
    )
    assert 16 <= int(execution["tree_node_count"]) <= 30
    assert out_a.answer_gt == out_b.answer_gt
    assert out_a.annotation_gt == out_b.annotation_gt
    assert execution == out_b.trace_payload["execution_trace"]
    assert out_a.image.tobytes() == out_b.image.tobytes()


def test_pages_hierarchy_org_chart_sampling_covers_active_tasks() -> None:
    tasks = (
        PagesHierarchySubtreeDescendantCountTask(),
        PagesHierarchyManagerMostTotalReportsLabelTask(),
        PagesHierarchyManagerMostDirectReportsLabelTask(),
    )
    source_ids: Counter[str] = Counter()
    answers_by_variant: dict[str, set[str]] = defaultdict(set)

    for task in tasks:
        for index in range(15):
            out = task.generate(
                hash64(61540, task.task_id, index),
                params={},
                max_attempts=10,
            )
            execution = out.trace_payload["execution_trace"]
            assert str(execution["query_id"]) == SINGLE_QUERY_ID
            assert str(execution["scene_variant"]) == "org_chart"
            source_query_id = str(execution["source_query_id"])
            source_ids[source_query_id] += 1
            answers_by_variant[source_query_id].add(str(execution["answer_value"]))

    assert set(source_ids.keys()) == {
        "manager_most_direct_reports_label",
        "manager_most_total_reports_label",
        "subtree_descendant_count",
    }
    assert all(count == 15 for count in source_ids.values())
    assert all(len(values) >= 3 for values in answers_by_variant.values())


def test_pages_hierarchy_org_chart_layout_keeps_same_depth_nodes_separated() -> None:
    task = PagesHierarchySubtreeDescendantCountTask()

    for index in range(100):
        out = task.generate(
            hash64(61580, "pages_hierarchy_org_chart_layout_v0", index),
            params={},
            max_attempts=10,
        )
        execution = out.trace_payload["execution_trace"]
        node_bboxes = out.trace_payload["render_map"]["node_bboxes_px"]
        nodes_by_depth: dict[int, list[tuple[str, list[float]]]] = defaultdict(list)
        for node_spec in execution["node_specs"]:
            bbox = [float(value) for value in node_bboxes[str(node_spec["node_bbox_id"])]]
            nodes_by_depth[int(node_spec["depth"])].append((str(node_spec["node_id"]), bbox))

        for depth_nodes in nodes_by_depth.values():
            ordered = sorted(depth_nodes, key=lambda item: item[1][0])
            for left, right in zip(ordered, ordered[1:]):
                assert float(right[1][0]) - float(left[1][2]) >= 8.0


def test_pages_hierarchy_retired_graph_like_tasks_are_removed() -> None:
    for module_name in (
        "trace_tasks.tasks.pages.hierarchy.path_length_count",
        "trace_tasks.tasks.pages.hierarchy.subtree_leaf_count",
    ):
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(module_name)
