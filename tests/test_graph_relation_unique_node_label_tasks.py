"""Behavior tests for graph unique node-label relation task."""

from __future__ import annotations

import json
from collections import Counter

from trace_tasks.core.seed import hash64
from trace_tasks.tasks import TASK_REGISTRY
from trace_tasks.tasks.graph.node_link.unique_related_node_label import GraphRelationUniqueNodeLabelTask


def _extract_prompt_json_example(prompt: str) -> dict:
    marker = "Example JSON:\n"
    assert marker in str(prompt)
    payload = str(prompt).split(marker, 1)[1].strip()
    return json.loads(payload)


def test_graph_relation_unique_node_label_contract_matches_trace() -> None:
    task = GraphRelationUniqueNodeLabelTask()
    out = task.generate(
        21001,
        params={
            "query_id": "unique_successor_label",
            "node_count": 7,
            "label_variant": "named",
            "edge_routing_variant": "mixed_arc",
            "layout_variant": "shell",
        },
        max_attempts=100,
    )
    trace = out.trace_payload
    execution = trace["execution_trace"]
    node_entities = [
        entity for entity in trace["scene_ir"]["entities"] if entity["entity_kind"] == "graph_node"
    ]

    assert "task_graph__node_link__unique_related_node_label" in TASK_REGISTRY
    assert out.scene_id == "node_link"
    assert out.query_id == "unique_successor_label"
    assert out.answer_gt.type == "string"
    assert out.annotation_gt.type == "point"
    assert trace["scene_ir"]["scene_kind"] == "graph_unique_related_node_label"
    assert execution["query_id"] == "unique_successor_label"
    assert execution["graph_directionality"] == "directed"
    assert execution["relation_mode"] == "directed_unique_successor"
    assert execution["edge_routing_variant"] == "mixed_arc"
    assert execution["label_variant"] == "named"
    assert 'node "' in str(out.prompt)

    query_label = str(execution["query_label"])
    answer_label = str(execution["answer_label"])
    assert query_label
    assert tuple(execution["target_labels"]) == (answer_label,)
    assert tuple(execution["matching_labels"]) == (answer_label,)
    assert out.answer_gt.value == answer_label
    assert trace["witness_symbolic"]["labels"] == [answer_label]
    assert trace["projected_annotation"]["type"] == "point"
    assert trace["projected_annotation"]["point"] == out.annotation_gt.value
    answer_nodes = [node for node in node_entities if str(node["label"]) == answer_label]
    assert len(answer_nodes) == 1
    answer_entity = answer_nodes[0]
    assert answer_entity["center_px"] == out.annotation_gt.value
    assert sorted(out.prompt_variants.keys()) == ["answer_and_annotation", "answer_only"]


def test_graph_relation_unique_node_label_all_query_contracts() -> None:
    task = GraphRelationUniqueNodeLabelTask()
    cases = (
        ("unique_neighbor_label", "undirected"),
        ("unique_successor_label", "directed"),
        ("unique_predecessor_label", "directed"),
    )
    for offset, (query_id, directionality) in enumerate(cases):
        out = task.generate(
            21010 + offset,
            params={"query_id": query_id, "label_variant": "letters"},
            max_attempts=100,
        )
        execution = out.trace_payload["execution_trace"]
        query_label = str(execution["query_label"])
        answer_label = str(execution["answer_label"])
        assert out.query_id == query_id
        assert execution["graph_directionality"] == directionality
        assert query_label
        assert tuple(execution["target_labels"]) == (answer_label,)
        assert tuple(execution["matching_labels"]) == (answer_label,)
        assert out.answer_gt.value == answer_label
        assert out.annotation_gt.type == "point"
        assert len(out.annotation_gt.value) == 2


def test_graph_relation_unique_node_label_prompt_example_contract() -> None:
    task = GraphRelationUniqueNodeLabelTask()
    out = task.generate(
        21020,
        params={"query_id": "unique_neighbor_label", "label_variant": "numbers"},
        max_attempts=100,
    )

    answer_only = _extract_prompt_json_example(out.prompt_variants["answer_only"])
    answer_and_annotation = _extract_prompt_json_example(out.prompt_variants["answer_and_annotation"])
    assert answer_only == {"answer": "B"}
    assert list(answer_and_annotation.keys()) == ["annotation", "answer"]
    assert answer_and_annotation["annotation"] == [303, 187]
    assert answer_and_annotation["answer"] == "B"
    assert "Annotation format:" in out.prompt_variants["answer_and_annotation"]
    assert "[x,y]" in out.prompt_variants["answer_and_annotation"]


def test_graph_relation_unique_node_label_balanced_sampling_covers_queries() -> None:
    task = GraphRelationUniqueNodeLabelTask()
    query_ids: Counter[str] = Counter()
    directionality: Counter[str] = Counter()
    label_variants: Counter[str] = Counter()
    edge_routing_variants: Counter[str] = Counter()
    answers: Counter[str] = Counter()
    for index in range(120):
        out = task.generate(
            hash64(21030, "graph_relation_unique_node_label", index),
            params={},
            max_attempts=100,
        )
        execution = out.trace_payload["execution_trace"]
        query_ids[str(out.query_id)] += 1
        directionality[str(execution["graph_directionality"])] += 1
        label_variants[str(execution["label_variant"])] += 1
        edge_routing_variants[str(execution["edge_routing_variant"])] += 1
        answers[str(out.answer_gt.value)] += 1
        assert str(out.answer_gt.value) == str(execution["answer_label"])
        assert out.annotation_gt.type == "point"
        assert len(out.annotation_gt.value) == 2

    assert set(query_ids) == set(("unique_neighbor_label", "unique_successor_label", "unique_predecessor_label"))
    assert all(20 <= count <= 60 for count in query_ids.values())
    assert set(directionality) == {"undirected", "directed"}
    assert set(label_variants) == {"letters", "numbers", "named"}
    assert set(edge_routing_variants) == {"straight", "mixed_arc"}
    assert len(answers) > 12
