"""Behavior tests for graph binary-tree diagram tasks."""

from __future__ import annotations

from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.graph.binary_tree.child_structure_node_count import (
    GraphCountingBinaryTreeChildStructureNodeCountTask,
)
from trace_tasks.tasks.graph.binary_tree.depth_level_node_count import (
    GraphCountingBinaryTreeDepthLevelNodeCountTask,
)
from trace_tasks.tasks.graph.binary_tree.traversal_kth_label import (
    GraphOrderBinaryTreeTraversalLabelTask,
)
from trace_tasks.tasks.graph.binary_tree.local_relative_node_label import (
    GraphRelationBinaryTreeLocalRelativeNodeLabelTask,
)
from trace_tasks.tasks.graph.binary_tree.lowest_common_ancestor_label import (
    GraphRelationBinaryTreeLowestCommonAncestorLabelTask,
)
from trace_tasks.tasks.graph.binary_tree.bst_path_operation_label import (
    GraphRelationBstPathOperationLabelTask,
)
from trace_tasks.tasks.graph.binary_tree.heap_property_violation_label import (
    GraphRelationHeapPropertyViolationLabelTask,
)


def _binary_tree_nodes(trace_payload: dict) -> list[dict]:
    return [
        entity
        for entity in trace_payload["scene_ir"]["entities"]
        if entity["entity_kind"] == "binary_tree_node"
    ]


def _binary_tree_edges(trace_payload: dict) -> list[dict]:
    return [
        entity
        for entity in trace_payload["scene_ir"]["entities"]
        if entity["entity_kind"] == "binary_tree_edge"
    ]


_BROAD_BINARY_TREE_ANSWER_HINT = (
    "requested count as an integer or the answer node label"
)
_COUNT_ANSWER_HINT = 'set "answer" to the requested count as an integer'
_NODE_LABEL_ANSWER_HINT = (
    'set "answer" to the answer node label as a string exactly as shown'
)
_NUMERIC_KEY_ANSWER_HINT = (
    'set "answer" to the answer node key as a string exactly as shown'
)


def _answer_hint_slot(trace_payload: dict) -> str:
    return str(
        trace_payload["query_spec"]["prompt_variants"]["answer_only"]["metadata"][
            "slot_values"
        ]["answer_hint"]
    )


def _assert_answer_hint(out, expected: str) -> None:
    assert _answer_hint_slot(out.trace_payload) == expected
    for prompt in out.prompt_variants.values():
        assert expected in prompt
        assert _BROAD_BINARY_TREE_ANSWER_HINT not in prompt


def test_graph_counting_binary_tree_node_count_contracts() -> None:
    child_task = GraphCountingBinaryTreeChildStructureNodeCountTask()
    depth_task = GraphCountingBinaryTreeDepthLevelNodeCountTask()
    cases = (
        ("leaf_node_count", 4, None),
        ("internal_node_count", 5, None),
        ("single_child_node_count", 3, None),
        ("two_child_node_count", 3, None),
        ("depth_level_node_count", 2, 3),
    )

    assert "task_graph__binary_tree__child_structure_node_count" in TASK_REGISTRY
    assert "task_graph__binary_tree__depth_level_node_count" in TASK_REGISTRY
    for offset, (query_id, target_count, target_depth) in enumerate(cases):
        params = {
            "query_id": query_id,
            "target_count": target_count,
            "label_variant": "letters",
            "scene_variant": "classic_tree",
        }
        if target_depth is not None:
            params["target_depth"] = target_depth
            params.pop("query_id")
        task = depth_task if query_id == "depth_level_node_count" else child_task
        out = task.generate(23000 + offset, params=params, max_attempts=300)
        trace = out.trace_payload
        nodes = _binary_tree_nodes(trace)
        edges = _binary_tree_edges(trace)

        assert out.scene_id == "binary_tree"
        assert out.query_id == ("single" if query_id == "depth_level_node_count" else query_id)
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "point_set"
        assert int(out.answer_gt.value) == target_count
        assert len(out.annotation_gt.value) == target_count
        assert trace["scene_ir"]["scene_kind"] == "binary_tree"
        assert trace["execution_trace"]["query_id"] == ("single" if query_id == "depth_level_node_count" else query_id)
        assert trace["execution_trace"]["internal_query_id"] == query_id
        assert len(nodes) == int(trace["execution_trace"]["node_count"])
        assert len(edges) == len(nodes) - 1
        assert any(
            node["left_label"] is not None or node["right_label"] is not None
            for node in nodes
        )
        assert trace["projected_annotation"]["type"] == "point_set"
        assert trace["projected_annotation"]["point_set"] == out.annotation_gt.value
        assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
        assert "[x,y]" in out.prompt_variants["answer_and_annotation"]
        _assert_answer_hint(out, _COUNT_ANSWER_HINT)


def test_graph_order_binary_tree_traversal_label_contracts() -> None:
    task = GraphOrderBinaryTreeTraversalLabelTask()
    traversal_keys = {
        "preorder_kth_node_label": "preorder_labels",
        "inorder_kth_node_label": "inorder_labels",
        "postorder_kth_node_label": "postorder_labels",
        "level_order_kth_node_label": "level_order_labels",
    }

    assert "task_graph__binary_tree__traversal_kth_label" in TASK_REGISTRY
    for offset, (query_id, traversal_key) in enumerate(traversal_keys.items()):
        out = task.generate(
            23100 + offset,
            params={
                "query_id": query_id,
                "traversal_position": 4,
                "label_variant": "letters",
                "scene_variant": "paper_tree",
            },
            max_attempts=300,
        )
        trace = out.trace_payload
        nodes = _binary_tree_nodes(trace)
        edges = _binary_tree_edges(trace)
        traversal_labels = trace["scene_ir"]["relations"][traversal_key]

        assert out.scene_id == "binary_tree"
        assert out.query_id == query_id
        assert out.answer_gt.type == "string"
        assert out.annotation_gt.type == "point_sequence"
        assert str(out.answer_gt.value) == str(traversal_labels[3])
        assert len(out.annotation_gt.value) == 4
        assert len(nodes) == int(trace["execution_trace"]["node_count"])
        assert len(edges) == len(nodes) - 1
        assert trace["projected_annotation"]["type"] == "point_sequence"
        assert trace["projected_annotation"]["point_sequence"] == out.annotation_gt.value
        assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
        assert "ordered JSON array" in out.prompt_variants["answer_and_annotation"]
        _assert_answer_hint(out, _NODE_LABEL_ANSWER_HINT)


def test_graph_relation_binary_tree_node_label_contracts() -> None:
    local_task = GraphRelationBinaryTreeLocalRelativeNodeLabelTask()
    lca_task = GraphRelationBinaryTreeLowestCommonAncestorLabelTask()
    expected_role_keys = {
        "parent_label": {"child", "parent"},
        "left_child_label": {"parent", "left_child"},
        "right_child_label": {"parent", "right_child"},
        "sibling_label": {"node", "sibling"},
        "lowest_common_ancestor_label": {"node_a", "node_b", "lowest_common_ancestor"},
    }

    assert "task_graph__binary_tree__local_relative_node_label" in TASK_REGISTRY
    assert "task_graph__binary_tree__lowest_common_ancestor_label" in TASK_REGISTRY
    for offset, query_id in enumerate(expected_role_keys):
        task = lca_task if query_id == "lowest_common_ancestor_label" else local_task
        params = {
            "label_variant": "letters",
            "scene_variant": "boxed_tree",
        }
        if query_id != "lowest_common_ancestor_label":
            params["query_id"] = query_id
        out = task.generate(
            23200 + offset,
            params=params,
            max_attempts=300,
        )
        trace = out.trace_payload
        nodes = _binary_tree_nodes(trace)
        edges = _binary_tree_edges(trace)
        execution = trace["execution_trace"]

        assert out.scene_id == "binary_tree"
        assert out.query_id == ("single" if query_id == "lowest_common_ancestor_label" else query_id)
        assert out.answer_gt.type == "string"
        assert out.annotation_gt.type == "point_map"
        assert str(out.answer_gt.value) == str(execution["answer_label"])
        assert set(out.annotation_gt.value) == expected_role_keys[query_id]
        assert set(execution["annotation_role_to_label"]) == expected_role_keys[query_id]
        assert len(out.annotation_gt.value) == (
            3 if query_id == "lowest_common_ancestor_label" else 2
        )
        assert len(nodes) == int(execution["node_count"])
        assert len(edges) == len(nodes) - 1
        assert trace["projected_annotation"]["type"] == "point_map"
        assert trace["projected_annotation"]["point_map"] == out.annotation_gt.value
        assert (
            trace["projected_annotation"]["pixel_point_map"] == out.annotation_gt.value
        )
        assert sum(1 for node in nodes if node["is_answer_node"]) == 1
        assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
        assert "JSON object" in out.prompt_variants["answer_and_annotation"]
        assert "[x,y]" in out.prompt_variants["answer_and_annotation"]
        _assert_answer_hint(out, _NODE_LABEL_ANSWER_HINT)


def test_graph_binary_tree_lca_answer_scope_can_be_forced() -> None:
    task = GraphRelationBinaryTreeLowestCommonAncestorLabelTask()
    for scope, expected_answer_node_id in (("non_root", "L"), ("root", "")):
        out = task.generate(
            2000,
            params={
                "label_variant": "letters",
                "scene_variant": "boxed_tree",
                "relation_answer_scope": scope,
            },
            max_attempts=500,
        )
        execution = out.trace_payload["execution_trace"]
        params = out.trace_payload["query_spec"]["params"]
        assert execution["answer_scope"] == scope
        assert execution["answer_node_id"] == expected_answer_node_id
        assert params["relation_answer_scope"] == scope
        assert params["relation_answer_scope_probabilities"][scope] == 1.0


def test_graph_relation_bst_path_operation_label_contracts() -> None:
    task = GraphRelationBstPathOperationLabelTask()
    query_ids = (
        "bst_search_terminal_label",
        "bst_insert_parent_label",
    )

    assert "task_graph__binary_tree__bst_path_operation_label" in TASK_REGISTRY
    for offset, query_id in enumerate(query_ids):
        out = task.generate(
            23300 + offset,
            params={
                "query_id": query_id,
                "node_count": 9,
                "scene_variant": "classic_tree",
            },
            max_attempts=300,
        )
        trace = out.trace_payload
        nodes = _binary_tree_nodes(trace)
        edges = _binary_tree_edges(trace)
        execution = trace["execution_trace"]

        assert out.scene_id == "binary_tree"
        assert out.query_id == query_id
        assert out.answer_gt.type == "string"
        assert out.annotation_gt.type == "point_sequence"
        assert str(out.answer_gt.value) == str(execution["answer_label"])
        assert len(out.annotation_gt.value) >= 2
        assert len(nodes) == int(execution["node_count"])
        assert len(edges) == len(nodes) - 1
        assert trace["scene_ir"]["scene_kind"] == "search_tree_operation_diagram"
        assert trace["projected_annotation"]["type"] == "point_sequence"
        assert trace["projected_annotation"]["point_sequence"] == out.annotation_gt.value
        assert sum(1 for node in nodes if node["is_answer_node"]) == 1
        assert execution["target_key"] is not None
        assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
        assert "ordered JSON array" in out.prompt_variants["answer_and_annotation"]
        _assert_answer_hint(out, _NUMERIC_KEY_ANSWER_HINT)


def test_graph_relation_heap_property_violation_label_contracts() -> None:
    task = GraphRelationHeapPropertyViolationLabelTask()

    assert "task_graph__binary_tree__heap_property_violation_label" in TASK_REGISTRY
    out = task.generate(
        23350,
        params={
            "node_count": 9,
            "scene_variant": "classic_tree",
        },
        max_attempts=300,
    )
    trace = out.trace_payload
    nodes = _binary_tree_nodes(trace)
    edges = _binary_tree_edges(trace)
    execution = trace["execution_trace"]

    assert out.scene_id == "binary_tree"
    assert out.query_id == "single"
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "point_map"
    assert str(out.answer_gt.value) == str(execution["answer_label"])
    assert set(out.annotation_gt.value) == {"parent", "child"}
    assert len(out.annotation_gt.value) == 2
    assert len(nodes) == int(execution["node_count"])
    assert len(edges) == len(nodes) - 1
    assert trace["scene_ir"]["scene_kind"] == "search_tree_operation_diagram"
    assert trace["projected_annotation"]["type"] == "point_map"
    assert trace["projected_annotation"]["point_map"] == out.annotation_gt.value
    assert execution["annotation_role_to_label"]["child"] == str(out.answer_gt.value)
    assert sum(1 for node in nodes if node["is_answer_node"]) == 1
    assert execution["target_key"] is None
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert '"parent"' in out.prompt_variants["answer_and_annotation"]
    assert '"child"' in out.prompt_variants["answer_and_annotation"]
    _assert_answer_hint(out, _NUMERIC_KEY_ANSWER_HINT)
