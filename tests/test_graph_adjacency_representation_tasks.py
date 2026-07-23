"""Behavior tests for graph adjacency-representation tasks."""

from __future__ import annotations

from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.graph.adjacency.directed_pair_reciprocity_count import GraphCountingAdjacencyDirectedPairReciprocityCountTask
from trace_tasks.tasks.graph.adjacency.directed_strong_component_count import GraphCountingAdjacencyDirectedStrongComponentCountTask
from trace_tasks.tasks.graph.adjacency.mst_weight import GraphOptimizationAdjacencyMatrixMSTWeightTask
from trace_tasks.tasks.graph.adjacency.traversal_kth_label import GraphOrderAdjacencyTraversalLabelTask
from trace_tasks.tasks.graph.adjacency.undirected_component_count import GraphCountingAdjacencyUndirectedComponentCountTask


def _assert_bbox_in_image(bbox: list[float], size: tuple[int, int]) -> None:
    width, height = size
    x0, y0, x1, y1 = [float(value) for value in bbox]
    assert 0 <= x0 < x1 <= width
    assert 0 <= y0 < y1 <= height


def test_graph_order_adjacency_traversal_label_contracts() -> None:
    task = GraphOrderAdjacencyTraversalLabelTask()

    assert "task_graph__adjacency__traversal_kth_label" in TASK_REGISTRY
    for offset, query_id in enumerate(("bfs_kth_visit_label", "dfs_kth_visit_label")):
        out = task.generate(
            41100 + offset,
            params={
                "query_id": query_id,
                "node_count": 6,
                "traversal_position": 4,
                "label_variant": "letters",
            },
            max_attempts=100,
        )
        trace = out.trace_payload
        execution = trace["execution_trace"]

        assert out.scene_id == "adjacency"
        assert out.query_id == query_id
        assert out.answer_gt.type == "string"
        assert out.annotation_gt.type == "bbox_sequence"
        assert str(out.answer_gt.value) == str(execution["visit_order"][3])
        assert len(out.annotation_gt.value) == 4
        assert trace["projected_annotation"]["bbox_sequence"] == out.annotation_gt.value
        assert trace["scene_ir"]["relations"]["representation_variant"] == "adjacency_list_panel"
        assert len(trace["scene_ir"]["relations"]["adjacency"]) == 6
        for bbox in out.annotation_gt.value:
            _assert_bbox_in_image(bbox, out.image.size)
        assert "annotation format" in out.prompt_variants["answer_and_annotation"].lower()
        assert "ordered JSON array" in out.prompt_variants["answer_and_annotation"]


def test_graph_counting_adjacency_component_count_contracts() -> None:
    cases = (
        (GraphCountingAdjacencyUndirectedComponentCountTask(), "single", "adjacency_list_panel", "connected component"),
        (GraphCountingAdjacencyDirectedStrongComponentCountTask(), "single", "adjacency_matrix_panel", "strongly connected component"),
    )

    assert "task_graph__adjacency__undirected_component_count" in TASK_REGISTRY
    assert "task_graph__adjacency__directed_strong_component_count" in TASK_REGISTRY
    for offset, (task, query_id, scene_variant, prompt_phrase) in enumerate(cases):
        out = task.generate(
            41200 + offset,
            params={
                "query_id": query_id,
                "scene_variant": scene_variant,
                "node_count": 8,
                "component_count": 3,
                "label_variant": "letters",
            },
            max_attempts=100,
        )
        trace = out.trace_payload
        execution = trace["execution_trace"]

        assert out.scene_id == "adjacency"
        assert out.query_id == query_id
        assert out.answer_gt.type == "integer"
        assert out.annotation_gt.type == "bbox_set"
        assert int(out.answer_gt.value) == 3
        assert len(out.annotation_gt.value) == 3
        assert len(execution["components"]) == 3
        assert len(execution["component_topmost_row_labels"]) == 3
        assert execution["representation_variant"] == scene_variant
        assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
        assert prompt_phrase in str(out.prompt).lower()
        prompt_line = str(out.prompt).splitlines()[0].lower()
        if scene_variant == "adjacency_matrix_panel":
            assert "adjacency matrix" in prompt_line
            assert "adjacency list or adjacency matrix" not in prompt_line
        else:
            assert "adjacency list" in prompt_line
            assert "adjacency list or adjacency matrix" not in prompt_line
        for bbox in out.annotation_gt.value:
            _assert_bbox_in_image(bbox, out.image.size)
        annotation_text = out.prompt_variants["answer_and_annotation"]
        if "strongly" in prompt_phrase:
            assert "topmost row label in each strongly connected component" in annotation_text
        else:
            assert "topmost row label in each connected component" in annotation_text


def test_graph_counting_adjacency_pair_reciprocity_contracts() -> None:
    task = GraphCountingAdjacencyDirectedPairReciprocityCountTask()

    assert "task_graph__adjacency__directed_pair_reciprocity_count" in TASK_REGISTRY
    out = task.generate(
        41270,
        params={
            "query_id": "single",
            "node_count": 6,
            "target_count": 3,
            "label_variant": "letters",
        },
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]

    assert task.supported_query_ids == ("single",)
    assert out.scene_id == "adjacency"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "segment_set"
    assert int(out.answer_gt.value) == 3
    assert execution["representation_variant"] == "adjacency_matrix_panel"
    assert execution["target_pair_state"] == "mutual"
    assert len(execution["counted_pairs"]) == int(out.answer_gt.value)
    assert len(out.annotation_gt.value) == int(out.answer_gt.value)
    assert len(execution["annotation_cell_keys"]) == 2 * len(out.annotation_gt.value)
    assert trace["projected_annotation"]["segment_set"] == out.annotation_gt.value
    assert all("||" in key for key in execution["annotation_cell_keys"])
    for key in execution["annotation_cell_keys"]:
        row_label, column_label = str(key).split("||")
        assert row_label != column_label
    width, height = out.image.size
    for point_pair in out.annotation_gt.value:
        assert len(point_pair) == 2
        for point in point_pair:
            assert 0 <= float(point[0]) <= width
            assert 0 <= float(point[1]) <= height
    assert "mirrored matrix-cell centers" in out.prompt_variants["answer_and_annotation"]


def test_graph_counting_adjacency_pair_reciprocity_allows_zero_answer() -> None:
    task = GraphCountingAdjacencyDirectedPairReciprocityCountTask()
    try:
        task.generate(
            41290,
            params={
            "query_id": "one_way_pair_count",
                "node_count": 5,
                "target_count": 0,
                "label_variant": "letters",
            },
            max_attempts=100,
        )
    except ValueError as exc:
        assert "query_id" in str(exc)
    else:
        raise AssertionError("one_way_pair_count should no longer be a supported query")

    zero = task.generate(
        41291,
        params={
            "query_id": "single",
            "node_count": 5,
            "target_count": 0,
            "label_variant": "letters",
        },
        max_attempts=100,
    )
    assert zero.answer_gt.value == 0
    assert zero.annotation_gt.value == []
    assert zero.trace_payload["projected_annotation"]["segment_set"] == []


def test_graph_optimization_adjacency_matrix_mst_weight_contracts() -> None:
    task = GraphOptimizationAdjacencyMatrixMSTWeightTask()

    assert "task_graph__adjacency__mst_weight" in TASK_REGISTRY
    out = task.generate(
        41300,
        params={
            "query_id": "single",
            "node_count": 5,
            "extra_edge_count": 2,
            "edge_weight_min": 1,
            "edge_weight_max": 12,
            "label_variant": "letters",
        },
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    relation_edges = {tuple(edge) for edge in trace["scene_ir"]["relations"]["minimum_spanning_tree_edges"]}
    weights = {
        tuple(entry["edge"]): int(entry["weight"])
        for entry in trace["scene_ir"]["relations"]["weights"]
    }

    assert out.scene_id == "adjacency"
    assert out.query_id == "single"
    assert out.answer_gt.type == "integer"
    assert out.annotation_gt.type == "bbox_set"
    assert execution["representation_variant"] == "adjacency_matrix_panel"
    assert len(out.annotation_gt.value) == int(execution["node_count"]) - 1
    assert int(out.answer_gt.value) == sum(weights[edge] for edge in relation_edges)
    assert trace["projected_annotation"]["bbox_set"] == out.annotation_gt.value
    for bbox in out.annotation_gt.value:
        _assert_bbox_in_image(bbox, out.image.size)
    assert "matrix cell" in out.prompt_variants["answer_and_annotation"]
